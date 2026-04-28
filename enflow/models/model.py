from typing import Any, Dict, List, Optional, TypeVar, Literal

import numpy as np
import torch
from pytorch_lightning import seed_everything
from torch import Tensor
from torch_geometric.data import Batch

from enflow.commons.covmat import set_multiple_rdmol_positions
from enflow.commons.featurization import MoleculeFeaturizer, get_mol_from_smiles
from enflow.commons.utils import signed_volume
from enflow.models.base import BaseModel
from enflow.models.loss import batchwise_l2_loss
from enflow.models.utils import (
    HarmonicSampler,
    center_of_mass,
    extend_bond_index,
    rmsd_align,
    unsqueeze_like,
)
from enflow.networks.torchmd_net import TorchMDDynamics

__all__ = ["BaseFlow"]

Config = TypeVar("Config", str, Dict[str, Any])


def dummy_time(t: Tensor, value: float = 0.5) -> Tensor:
    """
    Dummy time function.
    :param t: time tensor
    :param value: value to return
    :return: time tensor
    """
    return t.new_full(t.shape, value)


class BaseFlow(BaseModel):
    """LightningModule for Flow Matching"""

    __prior_types__ = ["gaussian", "harmonic"]
    __interpolation_types__ = ["linear", "gvp", "gvp_w_sigma", "gvp_squared"]

    def __init__(
        self,
        # flow matching network args
        network_type: str = "TorchMDDynamics",
        hidden_channels: int = 128,
        num_layers: int = 8,
        num_rbf: int = 64,
        rbf_type: str = "expnorm",
        trainable_rbf: bool = False,
        activation: str = "silu",
        neighbor_embedding: int = True,
        cutoff_lower: float = 0.0,
        cutoff_upper: float = 10.0,
        max_z: int = 100,
        node_attr_dim: int = 0,
        edge_attr_dim: int = 0,
        attn_activation: str = "silu",
        num_heads: int = 8,
        distance_influence: str = "both",
        reduce_op: str = "sum",
        qk_norm: bool = False,
        output_layer_norm: bool = False,
        clip_during_norm: bool = False,
        max_num_neighbors: int = 32,
        so3_equivariant: bool = False,
        # flow matching args
        sigma: float = 0.1,
        prior_type: str = "gaussian",
        sample_time_dist: str = "uniform",
        harmonic_alpha: float = 1.0,
        parity_switch: Optional[str] = None,
        # make edge_type one_hot
        edge_one_hot: bool = False,
        edge_one_hot_types: int = 5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # setup network
        if network_type == "TorchMDDynamics":
            self.vector_network = TorchMDDynamics(
                hidden_channels=hidden_channels,
                num_layers=num_layers,
                num_rbf=num_rbf,
                rbf_type=rbf_type,
                trainable_rbf=trainable_rbf,
                activation=activation,
                neighbor_embedding=neighbor_embedding,
                cutoff_lower=cutoff_lower,
                cutoff_upper=cutoff_upper,
                max_z=max_z,
                node_attr_dim=node_attr_dim,
                edge_attr_dim=edge_attr_dim,
                attn_activation=attn_activation,
                num_heads=num_heads,
                distance_influence=distance_influence,
                reduce_op=reduce_op,
                qk_norm=qk_norm,
                output_layer_norm=output_layer_norm,
                clip_during_norm=clip_during_norm,
                so3_equivariant=so3_equivariant,
                reture_x_or_v="v",
            )
            self.energy_network = TorchMDDynamics(
                hidden_channels=hidden_channels,
                num_layers=num_layers,
                num_rbf=num_rbf,
                rbf_type=rbf_type,
                trainable_rbf=trainable_rbf,
                activation=activation,
                neighbor_embedding=neighbor_embedding,
                cutoff_lower=cutoff_lower,
                cutoff_upper=cutoff_upper,
                max_z=max_z,
                node_attr_dim=node_attr_dim,
                edge_attr_dim=edge_attr_dim,
                attn_activation=attn_activation,
                num_heads=num_heads,
                distance_influence=distance_influence,
                reduce_op=reduce_op,
                qk_norm=qk_norm,
                output_layer_norm=output_layer_norm,
                clip_during_norm=clip_during_norm,
                so3_equivariant=so3_equivariant,
                reture_x_or_v="x",
            )
        else:
            raise NotImplementedError(f"Network {network_type} not implemented.")

        self.sigma = sigma
        self.cutoff = cutoff_upper
        self.parity_switch = parity_switch
        self.prior_type = prior_type
        self.sample_time_dist = sample_time_dist
        self.edge_one_hot = edge_one_hot
        self.edge_one_hot_types = edge_one_hot_types
        self.max_num_neighbors = max_num_neighbors
        # flow loss weighted by time (cut off)
        self.energy_finetune = kwargs.get("energy_finetune", False)
        self.energy_finetune_weight = kwargs.get(
            "energy_finetune_weight", 0.0 if not self.energy_finetune else 0.01
        )
        # reflow config
        self.reflow = kwargs.get("reflow", False)

        if self.energy_finetune:
            # frozen vector_network
            for param in self.vector_network.parameters():
                param.requires_grad = False

        if self.reflow:
            # frozen energy_network
            for param in self.energy_network.parameters():
                param.requires_grad = False

        if parity_switch is not None:
            assert (
                parity_switch == "post_hoc"
            ), f"Parity switch {parity_switch} not implemented"

        assert (
            self.prior_type in self.__prior_types__
        ), f"""\nPrior type {prior_type} not available.
            This is the list of implemented prior types {self.__prior_types__}.\n"""

        if prior_type == "harmonic":
            self.harmonic_sampler = HarmonicSampler(alpha=harmonic_alpha)

    @classmethod
    def from_config(cls, cfg: Config):
        import yaml

        if isinstance(cfg, str):
            cfg = yaml.safe_load(open(cfg))
        if isinstance(cfg, dict):
            return cls(**cfg["model_args"])
        else:
            raise ValueError("cfg should be a dictionary or a path to a yaml file")

    def sigma_t(self, t):
        return self.sigma * torch.sqrt(t * (1 - t))

    def sigma_dot_t(self, t):
        return self.sigma * 0.5 * (1 - 2 * t) / torch.sqrt(t * (1 - t))

    def sample_conditional_pt(self, x0: Tensor, x1: Tensor, t: Tensor, batch: Tensor):
        # Have this here in case sample_conditional_pt
        # is used outside of compute_conditional_vector_field
        # center both x0 and pos (x1: data distribution)
        x0 = center_of_mass(x0, batch=batch)
        x1 = center_of_mass(x1, batch=batch)

        # unsqueeze t and then reshape to number of atoms
        t = t[batch] if batch is not None else t
        t = unsqueeze_like(t, target=x0)

        # linear interpolation between x0 and x1
        # mu_t = self.interpolation_fn(x0, x1, t)
        eps = torch.randn_like(x1)

        # center each around center of mass
        eps = center_of_mass(eps, batch=batch)
        x_t_prime = t * x1 + (1 - t) * x0

        # no noise at t = 0 or t = 1
        x_t = x_t_prime + self.sigma_t(t) * eps

        return x_t_prime, x_t, eps

    def compute_conditional_vector_field(self, x0, x1, t, batch=None):
        if batch is None:
            batch = torch.zeros((x1.size(0),)).to(self.device)

        # sample a gaussian centered around the interpolation of x1, x0
        x_t_prime, x_t, eps = self.sample_conditional_pt(x0, x1, t, batch=batch)
        t = unsqueeze_like(t[batch], x1)

        # derivative of interpolate plus derivative of sigma function * noise
        s_t = x1 - x0
        v_t = s_t + self.sigma_dot_t(t) * eps

        return x_t, v_t, x_t_prime, s_t

    def switch_parity_of_pos(
        self, pos, chiral_index, chiral_nbr_index, chiral_tag, batch
    ):
        assert all(
            [
                key is not None
                for key in [chiral_index, chiral_nbr_index, chiral_tag, batch]
            ]
        )
        num_graphs = batch.max().item() + 1
        sv = signed_volume(
            pos[chiral_nbr_index.view(chiral_index.shape[1], 4)].unsqueeze(2)
        ).squeeze()
        ct = chiral_tag
        z_flip = sv * ct

        graph_diag = torch.ones(num_graphs, device=self.device)
        graph_diag[batch[chiral_index][:, (z_flip == -1.0)].squeeze()] = -1.0
        node_factor = graph_diag[batch].unsqueeze(1)

        return pos * node_factor

    def sample_base_dist(
        self,
        size: torch.Size,
        edge_index: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
        smiles: Optional[str] = None,
    ) -> Tensor:
        """Sample from prior distribution (Either harmonic or gaussian)"""
        if self.prior_type == "harmonic":
            assert (edge_index is not None) and (batch is not None)
            x0 = self.harmonic_sampler.sample(
                size=size, edge_index=edge_index, batch=batch, smiles=smiles
            ).to(self.device)

            # check if x0 is nan
            if torch.isnan(x0).any():
                raise ValueError("x0 is NaN. Check edge_index for disconnected graphs!")

            return x0

        # gaussian prior if not harmonic
        return torch.randn(size=size, device=self.device)

    def sample_time(
        self,
        num_samples: int,
        low: float = 1e-4,
        high: float = 0.9999,
        stage: str = "train",
    ):
        """Sample flow-matching time steps for training or validation"""
        if self.sample_time_dist == "uniform" or stage == "val":
            return torch.zeros(size=(num_samples, 1), device=self.device).uniform_(
                low, high
            )
        elif self.sample_time_dist == "logit_norm":
            return torch.sigmoid(torch.randn(size=(num_samples, 1), device=self.device))

        raise NotImplementedError(
            f"Time sampling with {self.sample_time_dist} not implemented"
        )

    def forward(
        self,
        z: Tensor,
        t: Tensor,
        pos: Tensor,
        bond_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        node_attr: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
        forward_type: Literal["vector", "energy"] = "energy",
    ):
        # center the positions at 0
        pos = center_of_mass(pos, batch=batch)
        # compute extended bond index
        edge_index, edge_type = extend_bond_index(
            pos=pos,
            bond_index=bond_index,
            batch=batch,
            bond_attr=edge_attr,
            device=self.device,
            one_hot=self.edge_one_hot,
            one_hot_types=self.edge_one_hot_types,
            cutoff=self.cutoff,
            max_num_neighbors=self.max_num_neighbors,
        )

        if forward_type == "energy":  # time-independent energy
            t = dummy_time(t, value=0.0)
            # compute energy and score from network
            E = self.energy_network(
                z=z,
                t=t[batch],
                pos=pos,
                edge_index=edge_index,
                edge_attr=edge_type,
                node_attr=node_attr,
                batch=batch,
            )
            return E

        elif forward_type == "vector":  # time-dependent vector field
            # compute vector field from network
            v_t = self.vector_network(
                z=z,
                t=t[batch],
                pos=pos,
                edge_index=edge_index,
                edge_attr=edge_type,
                node_attr=node_attr,
                batch=batch,
            )
            return v_t

        else:
            raise NotImplementedError(f"Forward type {forward_type} not implemented")

    def dEdx(
        self,
        z: Tensor,
        t: Tensor,
        pos: Tensor,
        bond_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        node_attr: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
        create_graph: bool = True,
        return_energy: bool = False,
    ):
        """
        Compute the gradient of the energy with respect to the positions
        """
        with torch.set_grad_enabled(True):
            # make pos differentiable
            pos = torch.autograd.Variable(pos.clone(), requires_grad=True)
            # compute energy
            E = self.forward(
                z=z,
                t=t,
                pos=pos,
                bond_index=bond_index,
                edge_attr=edge_attr,
                node_attr=node_attr,
                batch=batch,
                forward_type="energy",
            )

            dEdx = torch.autograd.grad(
                outputs=E,
                inputs=pos,
                grad_outputs=torch.ones_like(E),
                create_graph=create_graph,
            )[0]

            if return_energy:
                return dEdx, E

            return dEdx

    def velocity(
        self,
        z: Tensor,
        t: Tensor,
        pos: Tensor,
        bond_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        node_attr: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
        create_graph: bool = True,
    ):
        """
        Compute the velocity at time t given the positions and other parameters
        """
        v_t = self.forward(
            z=z,
            t=t,
            pos=pos,
            bond_index=bond_index,
            edge_attr=edge_attr,
            node_attr=node_attr,
            batch=batch,
            forward_type="vector",
        )

        return v_t

    def generic_step(self, batched_data, batch_idx: int, stage: str):
        # atomic numbers
        (
            z,
            pos,
            bond_index,
            node_attr,
            edge_attr,
            batch,
            chiral_index,
            chiral_nbr_index,
            chiral_tag,
            energy,
            energy_mean,
            energy_std,
        ) = (
            batched_data["atomic_numbers"],
            batched_data["pos"],
            batched_data["edge_index"],
            batched_data.get("node_attr", None),  # optional
            batched_data.get("edge_attr", None),  # optional
            batched_data.get("batch", None),  # optional
            batched_data.get("chiral_index", None),  # optional
            batched_data.get("chiral_nbr_index", None),  # optional
            batched_data.get("chiral_tag", None),  # optional
            batched_data["energy"],
            batched_data["energy_mean"],
            batched_data["energy_std"],
        )
        batch_size = batch.max().item() + 1 if batch is not None else 1

        # sample base distribution, either from harmonic or gaussian
        # x0 is sampling distribution and not data distribution
        x0 = self.sample_base_dist(
            pos.shape,
            edge_index=bond_index,
            batch=batch,
            smiles=batched_data.get("smiles", None),
        )

        # sample time steps equal to number of molecules in a batch
        t = self.sample_time(num_samples=batch_size, stage=stage)

        # sample conditional vector field for positions

        if not self.reflow:  # flow matching & energy matching

            x0 = (
                rmsd_align(pos=x0, ref_pos=pos, batch=batch)
                if self.prior_type == "harmonic"
                else x0
            )  # alignment for OT simulation.
            """
            sigma(t)  = sigma * sqrt(t * (1 - t))
            sigma_dot(t) = sigma * (1 - 2t) / (2 * sqrt(t * (1 - t)))
            --- ---
            x_t: (1 - t) * x0 + t * x1 + sigma(t) * eps
            v_t: x1 - x0 + sigma_dot(t) * eps
            x_t_prime: (1 - t) * x0 + t * x1
            s_t: x1 - x0
            """
            x_t, v_t, x_t_prime, s_t = self.compute_conditional_vector_field(
                x0=x0, x1=pos, t=t, batch=batch
            )

            if not self.energy_finetune:  # just do flow matching in pretraing phrase

                # NOTE: Flow Matching
                v_t_pred = self.velocity(
                    z=z,
                    t=t,
                    pos=x_t,
                    bond_index=bond_index,
                    edge_attr=edge_attr,
                    node_attr=node_attr,
                    batch=batch,
                )
                # compute fm loss
                l_fm = batchwise_l2_loss(v_t_pred, v_t, batch=batch, reduce="mean")

                assert (
                    torch.isfinite(l_fm).all().item()
                ), "Loss fm is NaN or Inf, fix bug"
                self.log_helper(
                    f"{stage}/flow_matching_loss", l_fm, batch_size=batch_size
                )

                # NOTE: Energy Matching
                # compute dEdx
                dEdx = self.dEdx(
                    z=z,
                    t=t,
                    pos=x_t_prime,
                    bond_index=bond_index,
                    edge_attr=edge_attr,
                    node_attr=node_attr,
                    batch=batch,
                )
                # compute energy matching loss
                l_em = batchwise_l2_loss(-dEdx, s_t, batch=batch, reduce="mean")

                assert (
                    torch.isfinite(l_em).all().item()
                ), "Loss em is NaN or Inf, fix bug"
                self.log_helper(
                    f"{stage}/energy_matching_loss", l_em, batch_size=batch_size
                )

                loss = l_fm + l_em

            else:  # just do energy matching and contrastive divergence in finetune phrase
                # NOTE: Energy Matching
                dEdx = self.dEdx(
                    z=z,
                    t=t,
                    pos=x_t_prime,
                    bond_index=bond_index,
                    edge_attr=edge_attr,
                    node_attr=node_attr,
                    batch=batch,
                )
                # compute energy matching loss
                l_em = batchwise_l2_loss(-dEdx, s_t, batch=batch, reduce="mean")

                assert (
                    torch.isfinite(l_em).all().item()
                ), "Loss em is NaN or Inf, fix bug"
                self.log_helper(
                    f"{stage}/energy_matching_loss", l_em, batch_size=batch_size
                )

                # NOTE: Contrastive Divergence
                # compute positive energy
                E_pos = self(
                    z=z,
                    t=t,
                    pos=pos,
                    bond_index=bond_index,
                    edge_attr=edge_attr,
                    node_attr=node_attr,
                    batch=batch,
                    forward_type="energy",
                )

                l_energy = torch.nn.functional.l1_loss(E_pos, energy)
                self.log_helper(f"{stage}/energy_loss", l_energy, batch_size=batch_size)
                # loss = l_energy
                loss = l_em + l_energy * self.energy_finetune_weight

        else:
            x0_prime, x1_prime = batched_data["pos_x0"], batched_data["pos_x1"]
            x0_prime = center_of_mass(x0_prime, batch=batch)

            x0_prime = rmsd_align(
                pos=x0_prime, ref_pos=x1_prime, batch=batch
            )  # alignment for OT simulation.

            # sample time steps equal to number of molecules in a batch
            t = self.sample_time(num_samples=batch_size, stage=stage)

            x_t, v_t, x_t_prime, s_t = self.compute_conditional_vector_field(
                x0=x0_prime, x1=x1_prime, t=t, batch=batch
            )

            v_t_pred = self.velocity(
                z=z,
                t=t,
                pos=x_t,
                bond_index=bond_index,
                edge_attr=edge_attr,
                node_attr=node_attr,
                batch=batch,
            )
            l_reflow = batchwise_l2_loss(v_t_pred, s_t, batch=batch, reduce="mean")

            assert (
                torch.isfinite(l_reflow).all().item()
            ), "Loss reflow is NaN or Inf, fix bug"
            self.log_helper(f"{stage}/reflow_loss", l_reflow, batch_size=batch_size)

            loss = l_reflow

        self.log_helper(f"{stage}/loss", loss, batch_size=batch_size)
        return loss

    def _compute_delta_t(self, t_schedule: Tensor, t: Tensor):
        if t + 1 >= t_schedule.size(0):
            return 0.0

        t_curr, t_next = t_schedule[t : t + 2]
        return t_next - t_curr

    @torch.no_grad()
    def simple_ode_sample(
        self,
        x0: Tensor,
        z: Tensor,
        bond_index: Tensor,
        batch: Tensor,
        node_attr: Tensor = None,
        edge_attr: Tensor = None,
        n_timesteps: int = 50,
    ):

        t_schedule = torch.linspace(0, 1.0, steps=n_timesteps + 1, device=self.device)
        x = center_of_mass(x0, batch=batch)

        n = t_schedule.size(0) - 1
        for i in range(n):
            t = t_schedule[i].repeat(x.size(0))
            t = unsqueeze_like(t, x)
            delta_t = self._compute_delta_t(t_schedule, t=i)

            v_t = self.velocity(
                z=z,
                t=t,
                pos=x,
                bond_index=bond_index,
                edge_attr=edge_attr,
                node_attr=node_attr,
                batch=batch,
            )
            x = x + delta_t * v_t

        return x

    @torch.no_grad()
    def sample(
        self,
        z: Tensor,
        bond_index: Tensor,
        batch: Tensor,
        node_attr: Tensor = None,
        edge_attr: Tensor = None,
        chiral_index: Tensor = None,
        chiral_nbr_index: Tensor = None,
        chiral_tag: Tensor = None,
        n_timesteps: int = 50,
        s_churn: float = 1.0,
        t_min: float = 1.0,
        t_max: float = 1.0,
        std: float = 1.0,
        sampler_type: str = "ode",
        guide_scale: float = 1.0,
        energy_guided: bool = False,
        return_energy: bool = False,
        sample_debug: bool = False,
    ):
        """
        By default performs ODE (sampler_type="ode") sampling
        If sampler_type is set to "stochastic", then it performs stochastic sampling
        """
        t_schedule = torch.linspace(0, 1.0, steps=n_timesteps + 1, device=self.device)
        x = center_of_mass(
            self.sample_base_dist((z.size(0), 3), bond_index, batch), batch=batch
        )
        gamma = torch.tensor(s_churn / n_timesteps).to(self.device)

        n = t_schedule.size(0) - 1
        for i in range(n):
            t = t_schedule[i].repeat(x.size(0))
            t = unsqueeze_like(t, x)
            delta_t = self._compute_delta_t(t_schedule, t=i)

            # We do ODE when t is outside of [s_min, s_max]
            if (
                t_schedule[i] < t_min or t_schedule[i] >= t_max
            ) or sampler_type == "ode":
                v_t = self.velocity(
                    z=z,
                    t=t,
                    pos=x,
                    bond_index=bond_index,
                    edge_attr=edge_attr,
                    node_attr=node_attr,
                    batch=batch,
                )

                if not energy_guided:
                    x = x + delta_t * v_t

                else:
                    x1_pred = x + (1 - t) * v_t

                    dEdx = self.dEdx(
                        z=z,
                        t=t,
                        pos=x1_pred,
                        bond_index=bond_index,
                        edge_attr=edge_attr,
                        node_attr=node_attr,
                        batch=batch,
                        create_graph=False,
                    )

                    f_t = 0.1 * (1 - t) ** 2
                    f_t *= guide_scale
                    v_t_prime = v_t - f_t * dEdx
                    if sample_debug:
                        v_norm = torch.linalg.vector_norm(v_t).item()
                        dEdx_norm = torch.linalg.vector_norm(dEdx).item()
                        ft_dedx = f_t * dEdx
                        ft_dedx_norm = torch.linalg.vector_norm(ft_dedx).item()
                        ratio = ft_dedx_norm / (v_norm + 1e-12)
                        t_scalar = t_schedule[i].item()
                        print(
                            "[sample-debug][ode] "
                            f"step={i}/{n - 1} "
                            f"t={t_scalar:.6f} "
                            f"||v_t||={v_norm:.6e} "
                            f"||dEdx||={dEdx_norm:.6e} "
                            f"||f_t*dEdx||={ft_dedx_norm:.6e} "
                            f"ratio={ratio:.6e}",
                            flush=True,
                        )

                    x = x + delta_t * v_t_prime

            # Stochastic sampling
            else:
                # TODO: energy guidance in stochastic sampling
                # delta_hat = gamma*delta_t
                delta_hat = gamma * (1 - t_schedule[i])
                t_prev_int = t_schedule[i] - delta_hat
                t_prev = t_prev_int.repeat(x.size(0))
                t_prev = unsqueeze_like(t_prev, x)
                """linear noise"""
                sig_t_sq = t_schedule[i] ** 2
                sig_t_prev_sq = t_prev_int**2
                mean = torch.zeros_like(x)
                noise = torch.normal(mean=mean, std=std)
                noise = center_of_mass(noise, batch=batch)
                x_prev = (
                    x
                    + torch.sqrt(torch.abs(sig_t_sq - sig_t_prev_sq))
                    * noise
                    * delta_hat
                )  # quadratic + linear decay

                v_t_prev = self.velocity(
                    z=z,
                    t=t_prev,
                    pos=x_prev,
                    bond_index=bond_index,
                    edge_attr=edge_attr,
                    node_attr=node_attr,
                    batch=batch,
                )
                # update step
                x = x_prev + v_t_prev * (delta_t + delta_hat)

        if self.parity_switch == "post_hoc":
            x = self.switch_parity_of_pos(
                x, chiral_index, chiral_nbr_index, chiral_tag, batch
            )

        if return_energy:
            E = self.forward(
                z=z,
                t=t,
                pos=x,
                bond_index=bond_index,
                edge_attr=edge_attr,
                node_attr=node_attr,
                batch=batch,
                forward_type="energy",
            )
            return x, E

        return x


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
