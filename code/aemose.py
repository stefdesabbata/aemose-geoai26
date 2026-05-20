# -------------------------------------------------------------------------
# Autoencoder Mixture of Spatial Experts (AutoencoderMoSE) model
# for Geodemographic classification
#
# version: 4.4
# date:    2026-05-17
#
# -------------------------------------------------------------------------

aemose_version = '4.4'

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch_geometric.nn import GATConv, SAGEConv, StdAggregation, MeanAggregation, SimpleConv
from torch_geometric.data import Data


# Plain MLP
class MLP(nn.Module):
    def __init__(self, 
            size_sequence:  list[int],
            use_neg_slope:  float=0.01,
            use_batchnorm:  bool=False,
            use_layernorm:  bool=False,
            use_dropout:    float=0.0,
            use_init:       str=None,
            use_final_bias: bool=True,
            use_final_act:  bool=False,
            then_continues: bool=False
            ) -> None:
        super().__init__()
        self.size_sequence  = size_sequence
        self.num_of_layers  = len(self.size_sequence) - 1
        self.use_neg_slope  = use_neg_slope
        self.use_batchnorm  = use_batchnorm
        self.use_layernorm  = use_layernorm
        self.use_dropout    = use_dropout
        self.use_init       = use_init
        self.use_final_bias = use_final_bias
        # If then_continues is True, force final activation off to avoid double activation
        self.use_final_act  = use_final_act and not then_continues
        # Checks
        if use_batchnorm and use_layernorm:
            raise ValueError('Cannot use both batchnorm and layernorm simultaneously.')
        # Create Multi-Layer Perceptron based on size sequence
        self.mlp = torch.nn.Sequential()
        use_bias = not (self.use_batchnorm or self.use_layernorm)
        for i in range(self.num_of_layers):
            # Determine if bias is used for this layer
            this_use_bias = use_bias if i < (self.num_of_layers - 1) else self.use_final_bias
            # Add linear layer
            self.mlp.append(
                torch.nn.Linear(
                    self.size_sequence[i], 
                    self.size_sequence[i + 1],
                    bias=this_use_bias))
            if i < self.num_of_layers - 1 or then_continues:
                # Add batchnorm/layernorm
                if self.use_batchnorm:
                    self.mlp.append(
                        torch.nn.BatchNorm1d(
                            self.size_sequence[i + 1]))
                elif self.use_layernorm:
                    self.mlp.append(
                        torch.nn.LayerNorm(
                            self.size_sequence[i + 1]))
                # Add activation
                self.mlp.append(
                    torch.nn.LeakyReLU(
                        negative_slope=self.use_neg_slope))
            # Add dropout
            if i == self.num_of_layers - 2:
                if self.use_dropout > 0.0 and self.use_dropout <= 1.0:
                    self.mlp.append(
                        torch.nn.Dropout(p=self.use_dropout))
            # Add final activation
            if i == (self.num_of_layers - 1) and self.use_final_act:
                self.mlp.append(
                    torch.nn.ReLU())
        # Weight initialisation
        if use_init=='normal':
            for m in self.mlp:
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=1.0)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)
        elif use_init=='kaiming':
            for m in self.mlp:
                if isinstance(m, nn.Linear):
                    torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu', a=self.use_neg_slope)
                    if m.bias is not None:
                        torch.nn.init.constant_(m.bias, 0.0)
        elif use_init=='orthogonal':
            for m in self.mlp:
                if isinstance(m, nn.Linear):
                    torch.nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('leaky_relu', self.use_neg_slope))
                    if m.bias is not None:
                        torch.nn.init.constant_(m.bias, 0.0)
        elif use_init=='normal_small':
            use_init_std = 0.1
            print(f'Using small normal initialization: {use_init_std=}')
            for m in self.mlp:
                if isinstance(m, nn.Linear):
                    torch.nn.init.normal_(m.weight, mean=0.0, std=use_init_std)
                    if m.bias is not None:
                        torch.nn.init.constant_(m.bias, 0.0)
    
    def forward(
            self, x: torch.Tensor
            ) -> torch.Tensor:
        return self.mlp(x)


# Gating network to select experts
class SpatialGatingNetwork(nn.Module):
    def __init__(self,
            input_dim:      int,
            embedding_dim:  int,
            num_experts:    list[int]=[4, 4],
            gating_dims:    list[list[int]]=None,
            gtn_neg_slope:  float=0.01,
            gtn_batchnorm:  bool=False,
            gtn_layernorm:  bool=False,
            gtn_dropout:    float=0.0,
            gtn_init:       str=None,
            gtn_final_bias: bool=True,
            gtn_final_act:  bool=False,
            gconv_type:     str='SAGE',
            gconv_dims:     list[int]=None,
            gat_heads:      int=4,
            gat_concat:     bool=True,
            gat_residual:   bool=False
            ) -> None:
        super().__init__()
        # Checks
        if gconv_type not in ['SAGE', 'GAT']:
            raise ValueError(f"gconv_type '{gconv_type}' not recognized. Supported types are: 'SAGE', 'GAT'.")
        if gconv_dims is None:
            raise ValueError(f"gconv_dims is None. Please provide a list of integers for gconv_dims.")
        self.gating_groups  = gating_dims[0]
        self.gating_pergrp  = gating_dims[1]
        self.num_exp_groups = num_experts[0]
        self.num_exp_pergrp = num_experts[1]
        self.gat_heads      = gat_heads
        self.gat_concat     = gat_concat
        self.gat_residual   = gat_residual
        self.gconv_type     = gconv_type
        self.gconv_dims     = [input_dim] + gconv_dims
        self.gconv_out_dim  = (self.gconv_dims[-1] * self.gat_heads) if (self.gat_concat and gconv_type == 'GAT') else self.gconv_dims[-1]
        self.gconv_use_bias = not (gtn_batchnorm or gtn_layernorm)
        # ---
        # Checks
        if gtn_batchnorm and gtn_layernorm:
            raise ValueError('Cannot use both batchnorm and layernorm simultaneously.')
        # GNN convolutional layers
        self.gconvs = torch.nn.Sequential()
        for i in range(len(self.gconv_dims)-1):
            this_gconv_in_dim  = self.gconv_dims[0] if i == 0 else -1
            this_gconv_out_dim = (self.gconv_dims[i+1] * self.gat_heads) if (self.gat_concat and self.gconv_type == 'GAT') else self.gconv_dims[i+1]
            if self.gconv_type == 'SAGE':
                self.gconvs.append(SAGEConv(
                        this_gconv_in_dim,
                        self.gconv_dims[i+1],
                        aggr=MeanAggregation(),
                        # aggr=StdAggregation(),
                        # aggr=[MeanAggregation(), StdAggregation()],
                        normalize=False,
                        root_weight=True,
                        bias=self.gconv_use_bias
                    ))
            if self.gconv_type == 'GAT':
                self.gconvs.append(GATConv(
                        this_gconv_in_dim,
                        self.gconv_dims[i+1],
                        heads=         self.gat_heads,
                        concat=        self.gat_concat,
                        negative_slope=gtn_neg_slope,
                        dropout=       gtn_dropout,
                        residual=      self.gat_residual,
                        bias=          self.gconv_use_bias
                    ))
            if gtn_batchnorm:
                self.gconvs.append(torch.nn.BatchNorm1d(this_gconv_out_dim))
            elif gtn_layernorm:
                self.gconvs.append(torch.nn.LayerNorm(this_gconv_out_dim))
            self.gconvs.append(
                torch.nn.LeakyReLU(
                    negative_slope=gtn_neg_slope))
        # ---
        # # Group MLP Pre-GNN
        # self.mlp_pre = MLP(
        #     [input_dim, self.gconv_dims[0]],
        #     use_neg_slope= gtn_neg_slope,
        #     use_batchnorm= gtn_batchnorm,
        #     use_layernorm= gtn_layernorm,
        #     use_dropout=   gtn_dropout,
        #     use_init=      gtn_init,
        #     use_final_bias=gtn_final_bias,
        #     use_final_act= gtn_final_act,
        #     then_continues= True
        #     )
        # ---
        # Group MLP Post-GNN 
        self.group_mlp = MLP(
            size_sequence= [self.gconv_out_dim] + self.gating_groups + [self.num_exp_groups],
            use_neg_slope= gtn_neg_slope,
            use_batchnorm= gtn_batchnorm,
            use_layernorm= gtn_layernorm,
            use_dropout=   gtn_dropout,
            use_init=      gtn_init,
            use_final_bias=gtn_final_bias,
            use_final_act= gtn_final_act,
            )
        # ---
        # Expert-per-group MLPs
        self.gr_exp_mlp = nn.ModuleList([MLP(
            size_sequence= [embedding_dim] + self.gating_pergrp + [self.num_exp_pergrp], 
            use_neg_slope= gtn_neg_slope,
            use_batchnorm= gtn_batchnorm,
            use_layernorm= gtn_layernorm,
            use_dropout=   gtn_dropout,
            use_init=      gtn_init,
            use_final_bias=gtn_final_bias,
            use_final_act= gtn_final_act,
            ) for _ in range(self.num_exp_groups)])

    def forward(self,
            x: torch.Tensor,
            z: torch.Tensor,
            edge_index: torch.Tensor = None,
            tau: float=None,
            hard_group:  bool=False,
            hard_expert: bool=False,
            batch_size: int=None
            ) -> torch.Tensor:
        # x must be the full sampled set (seeds + neighbours) for GNN message passing
        # z must already be restricted to the seed nodes
        for layer in self.gconvs:
            if isinstance(layer, SAGEConv) or isinstance(layer, GATConv):
                x = layer(x, edge_index)
            else:
                x = layer(x)
        # Restrict GNN output to seed nodes before the pointwise group MLP
        if batch_size is not None:
            x = x[:batch_size]
        # Get group weights and unsqueeze
        # [batch, num_groups, 1]
        group_weights = F.gumbel_softmax(self.group_mlp(x), tau=tau, hard=hard_group, dim=1)
        group_weights = group_weights.unsqueeze(2)
        # Get expert weights per group and stack them
        # [batch, num_groups, experts_per_group]
        expert_weights = []
        for g in range(self.num_exp_groups):
            expert_weights.append(
                F.gumbel_softmax(self.gr_exp_mlp[g](z), tau=tau, hard=hard_expert, dim=1))
        expert_weights = torch.stack(expert_weights, dim=1)
        # Overall gating weights
        # [batch, num_groups * experts_per_group]
        gating_weights = group_weights * expert_weights
        gating_weights = gating_weights.view(x.shape[0], -1)
        return gating_weights, group_weights.squeeze(-1)


# Mixture of Experts Decoder
class MoEDecoder(nn.Module):
    def __init__(self, 
            input_dim:      int,
            expert_dims:    list[int],
            num_experts:    int=8,
            dec_neg_slope:  float=0.01,
            dec_batchnorm:  bool=False,
            dec_layernorm:  bool=False,
            dec_dropout:    float=0.0,
            dec_init:       str=None,
            dec_final_bias: bool=True,
            dec_final_act:  bool=False,
            ) -> None:
        super().__init__()
        self.expert_dims   = expert_dims
        self.embedding_dim = expert_dims[0]
        self.output_dim    = expert_dims[-1]
        self.eps           = 1e-9
        self.experts       = nn.ModuleList([MLP(
                size_sequence= expert_dims + [input_dim],
                use_neg_slope= dec_neg_slope,
                use_batchnorm= dec_batchnorm,
                use_layernorm= dec_layernorm,
                use_dropout=   dec_dropout,
                use_init=      dec_init,
                use_final_bias=dec_final_bias,
                use_final_act= dec_final_act,
            ) for _ in range(num_experts)])

    def forward(self,
            embeddings: torch.Tensor
            ) -> tuple[torch.Tensor]:
        experts_outputs = [expert(embeddings) for expert in self.experts]
        return torch.stack(experts_outputs, dim=1)


# Main Autoencoder Mixture of Spatial Experts model
class AutoencoderMoSE(L.LightningModule):
    def __init__(self,
            input_dim:          int,
            encoder_dims:       list[int],
            gating_dims:        list[int]=None,
            gating_tau_ann_start_val: float=5.0,
            gating_tau_ann_end_val:   float=0.1,
            gating_tau_ann_start_at:  int=0,
            gating_tau_ann_len_at:    int=50_000,
            gating_hard_group:           bool=False,
            gating_hard_group_start_at:  int=100_000,
            gating_hard_expert:          bool=False,
            gating_hard_expert_start_at: int=100_000,
            expert_dims:        list[int]=None,
            num_experts:        list[int]=[4, 4], 
            enc_neg_slope:      float=0.01,
            enc_batchnorm:      bool=False,
            enc_layernorm:      bool=False,
            enc_dropout:        float=0.0,
            enc_init:           str=None,
            enc_final_bias:     bool=True,
            enc_final_act:      bool=False,
            enc_learning_rate:  float=1e-3,
            gtn_neg_slope:      float=0.01,
            gtn_batchnorm:      bool=False,
            gtn_layernorm:      bool=False,
            gtn_dropout:        float=0.0,
            gtn_init:           str=None,
            gtn_final_bias:     bool=True,
            gtn_final_act:      bool=False,
            gtn_learning_rate:  float=1e-3,
            dec_neg_slope:      float=0.01,
            dec_batchnorm:      bool=False,
            dec_layernorm:      bool=False,
            dec_dropout:        float=0.0,
            dec_init:           str=None,
            dec_final_bias:     bool=True,
            dec_final_act:      bool=False,
            dec_learning_rate:  float=1e-3,
            gconv_type:         str='SAGE',
            gconv_dims:         list[int]=None,
            gat_heads:          int=4,
            gat_concat:         bool=True,
            gat_residual:       bool=False,
            cv2_loss_coef_grp:  float=0.0,
            cv2_loss_coef_exp:  float=0.0,
            cv2_loss_tol_grp:   float=0.0,
            cv2_loss_tol_exp:   float=0.0,
            spec_loss_coef_grp: float=0.05,
            spec_loss_coef_exp: float=0.05,
            weight_decay:       float=1e-3,
            monitor_es:         str='val_loss',
            monitor_lr:         str='val_loss',
            scheduler_patience:      int=30,
            scheduler_factor:        float=0.5,
            scheduler_start_at_epoch: int=0,
            batch_size:         int=256,
            model_name:         str=None
        ):
        super().__init__()
        self.save_hyperparameters()
        # Currently set up for two-level hierarchy
        if len(num_experts) != 2:
            raise ValueError('num_experts must be a list of two integers representing [num_groups, experts_per_group].')
        self.model_name     = model_name
        self.batch_size     = batch_size
        self.num_exp_groups = num_experts[0]
        self.num_exp_total  = num_experts[0] * num_experts[1]
        expert_dims    = encoder_dims[::-1] if expert_dims is None else expert_dims
        # Define encoder and MoE decoder
        self.encoder       = MLP(
            size_sequence= [input_dim] + encoder_dims,
            use_neg_slope= enc_neg_slope,
            use_batchnorm= enc_batchnorm,
            use_layernorm= enc_layernorm,
            use_dropout=   enc_dropout,
            use_init=      enc_init,
            use_final_bias=enc_final_bias,
            use_final_act= enc_final_act,
            )
        self.gating_network = SpatialGatingNetwork(
            input_dim=     input_dim,
            embedding_dim= encoder_dims[-1],
            num_experts=   num_experts,
            gating_dims=   gating_dims,
            gtn_neg_slope= gtn_neg_slope,
            gtn_batchnorm= gtn_batchnorm,
            gtn_layernorm= gtn_layernorm,
            gtn_dropout=   gtn_dropout,
            gtn_init=      gtn_init,
            gtn_final_bias=gtn_final_bias,
            gtn_final_act= gtn_final_act,
            gconv_type=gconv_type,
            gconv_dims=gconv_dims,
            gat_heads=gat_heads,
            gat_concat=gat_concat,
            gat_residual=gat_residual
            )
        self.moe_decoder   = MoEDecoder(
            input_dim=     input_dim,
            expert_dims=   expert_dims,
            num_experts=   self.num_exp_total,
            dec_neg_slope= dec_neg_slope,
            dec_batchnorm= dec_batchnorm,
            dec_layernorm= dec_layernorm,
            dec_dropout=   dec_dropout,
            dec_init=      dec_init,
            dec_final_bias=dec_final_bias,
            dec_final_act= dec_final_act
            )
        # Loss coefficients
        self.cv2_loss_coef_grp = cv2_loss_coef_grp
        self.cv2_loss_coef_exp = cv2_loss_coef_exp
        self.cv2_loss_tol_grp  = cv2_loss_tol_grp
        self.cv2_loss_tol_exp  = cv2_loss_tol_exp
        self.spec_loss_coef_grp = spec_loss_coef_grp
        self.spec_loss_coef_exp = spec_loss_coef_exp
        # Training hyperparameters
        self.enc_learning_rate = enc_learning_rate
        self.gtn_learning_rate = gtn_learning_rate
        self.dec_learning_rate = dec_learning_rate
        self.weight_decay      = weight_decay
        self.monitor_es        = monitor_es
        self.monitor_lr        = monitor_lr
        if self.monitor_lr != self.monitor_es:
            print(f'Warning: monitor_lr ({self.monitor_lr}) != monitor_es ({self.monitor_es})')
        # LR scheduler
        self.scheduler_patience       = scheduler_patience
        self.scheduler_factor         = scheduler_factor
        self.scheduler_start_at_epoch = scheduler_start_at_epoch
        # ---
        # Constants
        self.eps = 1e-9
        self.spec_loss_grp_max = math.log(self.num_exp_groups)
        self.spec_loss_tot_max = math.log(self.num_exp_total)
        # Gating temperature annealing parameters
        self.tau_sta_vl = gating_tau_ann_start_val
        self.tau_end_vl = gating_tau_ann_end_val
        self.tau_sta_at = gating_tau_ann_start_at
        self.tau_len_at = gating_tau_ann_len_at
        # Straight-through (hard) Gumbel-softmax switches (per level)
        self.gating_hard_group           = gating_hard_group
        self.gating_hard_group_start_at  = gating_hard_group_start_at
        self.gating_hard_expert          = gating_hard_expert
        self.gating_hard_expert_start_at = gating_hard_expert_start_at

    # --- Gating temperature annealing ---
    def _tau(self) -> float:
        progress = self.trainer.global_step
        if progress <= self.tau_sta_at:
            return self.tau_sta_vl
        if progress >= self.tau_sta_at + self.tau_len_at:
            return self.tau_end_vl
        progress_window = float(max(1e-12, self.tau_len_at))
        progress_fractn = float(progress - self.tau_sta_at) / progress_window
        progress_tau    = progress_fractn * (self.tau_end_vl - self.tau_sta_vl)
        return self.tau_sta_vl + progress_tau

    # --- Straight-through Gumbel-softmax switches ---
    # Group and expert levels are hardened independently: each returns True
    # once global_step has reached its start threshold. Hard one-hot routing
    # matches inference while back-propagating through the soft Gumbel-softmax
    # via the straight-through estimator.
    def _gumbel_hard_group(self) -> bool:
        if not self.gating_hard_group:
            return False
        return self.trainer.global_step >= self.gating_hard_group_start_at

    def _gumbel_hard_expert(self) -> bool:
        if not self.gating_hard_expert:
            return False
        return self.trainer.global_step >= self.gating_hard_expert_start_at

    # --- Auxiliary loss functions ---
    # Specialisation loss based on entropy of the gating weights
    def _spec_loss(self,
            gating_weights: torch.Tensor
            ) -> torch.Tensor:
        gating_entropy = -(gating_weights * (gating_weights + self.eps).log()).sum(dim=1)
        return gating_entropy.mean()
    # ---
    # CV2 importance loss 
    # (Shazeer et al. 2017, "Outrageously Large Neural Networks"). 
    # Penalises the squared coefficient of variation of the per-expert importance 
    # (sum of gating weights over the batch).
    def _cv2_loss(self,
            weights: torch.Tensor,
            tol:     float
            ) -> torch.Tensor:
        importance = weights.sum(dim=0).float()
        cv2 = importance.var(unbiased=False) / (importance.mean() ** 2 + self.eps)
        return F.relu(cv2 - tol)
    # ---
    # MSE loss functions based on
    # https://github.com/openai/sparse_autoencoder
    # MIT license
    def weighted_normalized_mse(self,
            reconstruction: torch.Tensor,
            original_input: torch.Tensor,
            weights:        torch.Tensor
        ) -> torch.Tensor:
        expert_error = (((reconstruction - original_input) ** 2).mean(dim=2) / ((original_input**2).mean(dim=2) + self.eps))
        return (expert_error * weights).sum(dim=1).mean()
    
    # --- Training steps ---
    # Training, validation, and test steps
    def training_step(self, batch, batch_idx):
        return self._a_step(batch, log_prefix='train_')
    def validation_step(self, batch, batch_idx):
        return self._a_step(batch, log_prefix='val_')
    def test_step(self, batch, batch_idx):
        return self._a_step(batch, log_prefix='test_')

    # Shared step logic
    def _a_step(self, batch, log_prefix='train_'):
        # Extract node attributes and edges
        x          = batch.x
        edge_index = batch.edge_index
        # Encoder
        # pointwise, only needs seed nodes, skip the forward on neighbour rows
        # https://pytorch-geometric.readthedocs.io/en/latest/modules/loader.html#torch_geometric.loader.NeighborLoader
        # Sampled nodes are sorted based on the order in which they were sampled. In particular, the first batch_size nodes represent the set of original mini-batch nodes.
        embeddings  = self.encoder(x[:batch.batch_size])
        # Gating network
        # The gating GNN still receives the full raw x for its message passing.
        gating_tau  = self._tau() if self.training else self.tau_end_vl
        hard_group_now  = self._gumbel_hard_group()  if self.training else self.gating_hard_group
        hard_expert_now = self._gumbel_hard_expert() if self.training else self.gating_hard_expert
        gating_weights, group_weights = self.gating_network(
            x, embeddings, edge_index, 
            gating_tau, hard_group_now, hard_expert_now,
            batch_size=batch.batch_size)
        # Expert decoders
        expert_outputs = self.moe_decoder(embeddings)
        # ---
        # Reconstruction loss
        # pointwise, only needs seed nodes, skip the forward on neighbour rows
        x = x[:batch.batch_size]
        x_expanded = x.unsqueeze(1).expand(-1, self.num_exp_total, -1)
        recon_loss = self.weighted_normalized_mse(expert_outputs, x_expanded, gating_weights)
        # ---
        # Auxiliary losses
        spec_loss_grp = self._spec_loss(group_weights)  / self.spec_loss_grp_max
        spec_loss_exp = self._spec_loss(gating_weights) / self.spec_loss_tot_max
        cv2_loss_grp  = self._cv2_loss(group_weights,  self.cv2_loss_tol_grp)
        cv2_loss_exp  = self._cv2_loss(gating_weights, self.cv2_loss_tol_exp)
        # ---
        # Embeddings sparsity loss (L1)
        # sparsity_loss = torch.mean(torch.abs(embeddings))
        # Total loss
        loss = recon_loss  +\
            (self.cv2_loss_coef_grp  * cv2_loss_grp)  +\
            (self.cv2_loss_coef_exp  * cv2_loss_exp)  +\
            (self.spec_loss_coef_grp * spec_loss_grp) +\
            (self.spec_loss_coef_exp * spec_loss_exp)
        # ---
        # Log
        self.log(f'{log_prefix}loss',       loss,          on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch.batch_size)
        self.log(f'{log_prefix}recon_loss', recon_loss,    on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch.batch_size)
        self.log(f'{log_prefix}cv2_loss_grp', cv2_loss_grp, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch.batch_size)
        self.log(f'{log_prefix}cv2_loss_exp', cv2_loss_exp, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch.batch_size)
        self.log(f'{log_prefix}spec_loss_grp', spec_loss_grp, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch.batch_size)
        self.log(f'{log_prefix}spec_loss_exp', spec_loss_exp, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch.batch_size)
        self.log(f'{log_prefix}gating_tau', gating_tau,    on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch.batch_size)
        self.log(f'{log_prefix}gating_hard_group',  float(hard_group_now),  on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch.batch_size)
        self.log(f'{log_prefix}gating_hard_expert', float(hard_expert_now), on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch.batch_size)
        # ---
        top_expert_idx = torch.argmax(gating_weights, dim=1)
        dead_experts   = self.num_exp_total - top_expert_idx.unique().shape[0]
        self.log(f'{log_prefix}dead_experts',  dead_experts,  on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch.batch_size)
        # ---
        # Return total loss
        return loss

    # Expert opinion function
    def exper_opinion(self,
            batch: Data
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Extract node attributes and edges
        x = batch.x
        edge_index = batch.edge_index
        # Encoder is pointwise, only needs seed nodes (see above)
        # The gating GNN still receives the full raw x for its message passing.
        # https://pytorch-geometric.readthedocs.io/en/latest/modules/loader.html#torch_geometric.loader.NeighborLoader
        # Sampled nodes are sorted based on the order in which they were sampled. In particular, the first batch_size nodes represent the set of original mini-batch nodes.
        embeddings   = self.encoder(x[:batch.batch_size])
        opinion_tau  = self._tau() if self.training else self.tau_end_vl
        opinion_hard_group  = self._gumbel_hard_group()  if self.training else self.gating_hard_group
        opinion_hard_expert = self._gumbel_hard_expert() if self.training else self.gating_hard_expert
        gating_weights, group_weights = self.gating_network(
            x, embeddings, edge_index, opinion_tau, opinion_hard_group, opinion_hard_expert,
            batch_size=batch.batch_size)
        top_expert_idx = torch.argmax(gating_weights, dim=1)
        top_group_idx  = torch.argmax(group_weights, dim=1)
        return embeddings, gating_weights, top_expert_idx, top_group_idx

    # Forward pass
    def forward(self,
            batch: Data,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Extract node attributes and edges
        x = batch.x
        edge_index = batch.edge_index
        # Encoder is pointwise, only needs seed nodes (see above)
        # The gating GNN still receives the full raw x for its message passing.
        # https://pytorch-geometric.readthedocs.io/en/latest/modules/loader.html#torch_geometric.loader.NeighborLoader
        # Sampled nodes are sorted based on the order in which they were sampled. In particular, the first batch_size nodes represent the set of original mini-batch nodes.
        embeddings   = self.encoder(x[:batch.batch_size])
        forward_tau  = self._tau() if self.training else self.tau_end_vl
        forward_hard_group  = self._gumbel_hard_group()  if self.training else self.gating_hard_group
        forward_hard_expert = self._gumbel_hard_expert() if self.training else self.gating_hard_expert
        gating_weights, _  = self.gating_network(
            x, embeddings, edge_index, forward_tau, forward_hard_group, forward_hard_expert,
            batch_size=batch.batch_size)
        experts_stacked = self.moe_decoder(embeddings)
        # Identify top expert index and create mask
        top_expert_idx  = torch.argmax(gating_weights, dim=1)
        top_expert_mask = F.one_hot(top_expert_idx, num_classes=self.num_exp_total).unsqueeze(2).float()
        # Apply mask to get only the top expert's output for each case
        reconstruction  = (experts_stacked * top_expert_mask).sum(dim=1)
        return embeddings, reconstruction, top_expert_idx, experts_stacked, gating_weights

    # Skipped epochs leave ReduceLROnPlateau's num_bad_epochs at 0
    # so patience starts fresh on the first stepped epoch.
    def lr_scheduler_step(self, scheduler, metric):
        if self.current_epoch < self.scheduler_start_at_epoch:
            return
        if metric is None:
            scheduler.step()
        else:
            scheduler.step(metric)

    # Optimizers and learning rate schedulers
    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(
        #     self.parameters(), 
        #     lr=self.learning_rate, 
        #     weight_decay=self.weight_decay)
        optimizer = torch.optim.AdamW([
            {
                'params': self.encoder.parameters(), 
                'lr': self.enc_learning_rate
            },
            {
                'params': self.gating_network.parameters(), 
                'lr': self.gtn_learning_rate
            },
            {
                'params': self.moe_decoder.experts.parameters(), 
                'lr': self.dec_learning_rate
            }
            ], weight_decay=self.weight_decay
            )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=self.scheduler_factor, 
            patience=self.scheduler_patience, 
            min_lr=1e-10)
        return {
            'optimizer': optimizer, 
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': self.monitor_lr,
                'interval': 'epoch',
                'frequency': 1}}
