import os
import re
import random
import math
from datetime import datetime
import argparse
import subprocess

import torch
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, RichProgressBar
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger

from torch_geometric.loader import NeighborLoader

from aemose import *

class DelayedModelCheckpoint(ModelCheckpoint):
    def __init__(self, *args, min_save_epoch: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_save_epoch = min_save_epoch
    def _should_skip_saving_checkpoint(self, trainer) -> bool:
        return (
            super()._should_skip_saving_checkpoint(trainer)
            or trainer.current_epoch < self.min_save_epoch
        )


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='AEMoE Training Script.')
    parser.add_argument('-e', '--earlystop', action='store_true')
    parser.add_argument('-r', '--randomseed', action='store_true')
    parser.add_argument('-s', '--set_seed', type=int)

    args = parser.parse_args()

    # --- Setup ---

    if args.randomseed:
        random_seed = random.randint(100, 100000)
    elif args.set_seed is not None:
        random_seed = args.set_seed
    else:
        random_seed = 72770
    max_epochs      = 1500
    min_epochs      = 600
    batch_size      = 512
    earlystop       = args.earlystop
    earlystop_delta = 0.0
    earlystop_patnc = 10
    loader_workers  = 11
    
    encoder_dims    = [192, 128, 96, 64]
    gating_dims     = [[64, 32], [32, 16]]
    num_experts     = [7, 3]

    gating_tau_ann_start_val = 5.0
    gating_tau_ann_end_val   = 1.0
    gating_tau_ann_start_at_epoch = 30
    gating_tau_ann_len_at_epoch   = 300

    # Straight-through (hard) Gumbel-softmax switches (per level)
    # Group routing hardens first so experts see stronger, less diffuse
    # gating signal; intra-group expert routing can stay soft (or harden later).
    gating_hard_group                 = True
    gating_hard_group_start_at_epoch  = 500
    gating_hard_expert                = False
    gating_hard_expert_start_at_epoch = 150
    scheduler_start_at_epoch          = 30 + gating_hard_group_start_at_epoch

    gconv_type      = 'SAGE'
    gconv_dims      = [128]
    gat_heads       = None
    gat_concat      = None
    gat_residual    = None
    # gconv_type      = 'GAT'
    # gconv_dims      = [171]
    # gat_heads       = 8
    # gat_concat      = True
    # gat_residual    = False

    # MLP settings
    # --- shared seetings ---
    all_neg_slope    = 0.01
    all_batchnorm    = True
    all_layernorm    = False
    all_dropout      = 0.0
    # --- specific settings ---
    enc_neg_slope   = all_neg_slope
    enc_batchnorm   = all_batchnorm
    enc_layernorm   = all_layernorm
    enc_dropout     = all_dropout
    enc_init        = 'orthogonal'
    enc_final_bias  = True
    enc_final_act   = False
    enc_learning_rate = 1.0e-3
    gtn_neg_slope   = all_neg_slope
    gtn_batchnorm   = all_batchnorm
    gtn_layernorm   = all_layernorm
    gtn_dropout     = all_dropout
    gtn_init        = 'normal_small'
    gtn_final_bias  = True
    gtn_final_act   = False
    gtn_learning_rate = 1.0e-3
    dec_neg_slope   = all_neg_slope
    dec_batchnorm   = all_batchnorm
    dec_layernorm   = all_layernorm
    dec_dropout     = all_dropout
    dec_init        = 'orthogonal'
    dec_final_bias  = True
    dec_final_act   = False
    dec_learning_rate = 1.0e-3

    # Loss coefficients
    cv2_loss_coef_grp = 0.05
    cv2_loss_coef_exp = 0.1
    cv2_loss_tol_grp  = 0.1
    cv2_loss_tol_exp  = 0.1
    spec_loss_coef_grp = 0.005
    spec_loss_coef_exp = 0.005

    # learning_rate   = 1.0e-3
    weight_decay    = 1.0e-3
    monitor_es      = 'val_loss'
    monitor_lr      = 'val_loss'

    dataset_nickname    = 'ukc2021-proc20251009001rs'
    dataset_graph_path  = f'data/pyg/{dataset_nickname}__pygdata.pth'
    model_version       = 'aemose-' + aemose_version.replace('.', '-')

    # --- Load Data ---

    data_graph       = torch.load(dataset_graph_path, weights_only=False)
    data_tensor_ncol = data_graph.x.shape[1]
    print(f"Graph data splits\n Train: {data_graph.train_mask.sum()}\n Valid: {data_graph.valid_mask.sum()}\n\n")

    # --- File settings ---

    model_timestamp  = datetime.now().strftime('%Y%m%d%H%M%S')
    model_version   += '_e' + ('x'.join([str(e) for e in num_experts]))  + \
                       '_s' + ('-'.join([str(d) for d in encoder_dims])) + \
                       ('_' + gconv_type.lower()) + ('-'.join([str(d) for d in gconv_dims])) +\
                       '_g0-' + ('-'.join([str(d) for d in gating_dims[0]])) + \
                       '-g1-' + ('-'.join([str(d) for d in gating_dims[1]])) + \
                       '_b' + str(batch_size) + \
                       ('_es' if earlystop else '') + \
                       '_' + model_timestamp
    model_name       = dataset_nickname + '__' + model_version
    print(f"\n\nModel name: {model_name}\n")

    dir_project     = os.path.expanduser(f'scratch/{model_name}')
    dir_checkpoints = os.path.join(dir_project, 'checkpoints')
    dir_logs        = os.path.join(dir_project, 'logs')
    model_info_path = os.path.join(dir_project, f'{model_name}__info.txt')
    model_itmp_path = os.path.join(dir_project, f'{model_name}__info-tmp.txt')
    model_resu_path = os.path.join(dir_project, f'{model_name}__results.csv')
    if not os.path.exists(dir_project):
        os.makedirs(dir_project)

    # --- Model ---

    L.seed_everything(random_seed)

    steps_per_epoch = math.ceil(data_graph.train_mask.sum().item() / batch_size)

    gating_tau_ann_start_at = gating_tau_ann_start_at_epoch * steps_per_epoch
    gating_tau_ann_len_at   = gating_tau_ann_len_at_epoch   * steps_per_epoch

    gating_hard_group_start_at  = gating_hard_group_start_at_epoch  * steps_per_epoch
    gating_hard_expert_start_at = gating_hard_expert_start_at_epoch * steps_per_epoch

    aemose_model = AutoencoderMoSE(
        input_dim=     data_tensor_ncol,
        encoder_dims=  encoder_dims,
        gating_dims=   gating_dims,
        gating_tau_ann_start_val=gating_tau_ann_start_val,
        gating_tau_ann_end_val=  gating_tau_ann_end_val,
        gating_tau_ann_start_at= gating_tau_ann_start_at,
        gating_tau_ann_len_at=   gating_tau_ann_len_at,
        gating_hard_group=           gating_hard_group,
        gating_hard_group_start_at=  gating_hard_group_start_at,
        gating_hard_expert=          gating_hard_expert,
        gating_hard_expert_start_at= gating_hard_expert_start_at,
        num_experts=   num_experts,
        enc_neg_slope= enc_neg_slope,
        enc_batchnorm= enc_batchnorm,
        enc_layernorm= enc_layernorm,
        enc_dropout=   enc_dropout,
        enc_init=      enc_init,
        enc_final_bias=enc_final_bias,
        enc_final_act= enc_final_act,
        enc_learning_rate= enc_learning_rate,
        gtn_neg_slope= gtn_neg_slope,
        gtn_batchnorm= gtn_batchnorm,
        gtn_layernorm= gtn_layernorm,
        gtn_dropout=   gtn_dropout,
        gtn_init=      gtn_init,
        gtn_final_bias=gtn_final_bias,
        gtn_final_act= gtn_final_act,
        gtn_learning_rate= gtn_learning_rate,
        dec_neg_slope= dec_neg_slope,
        dec_batchnorm= dec_batchnorm,
        dec_layernorm= dec_layernorm,
        dec_dropout=   dec_dropout,
        dec_init=      dec_init,
        dec_final_bias=dec_final_bias,
        dec_final_act= dec_final_act,
        dec_learning_rate= dec_learning_rate,
        gconv_type=    gconv_type,
        gconv_dims=    gconv_dims,
        gat_heads=     gat_heads,
        gat_concat=    gat_concat,
        gat_residual=  gat_residual,
        cv2_loss_coef_grp=cv2_loss_coef_grp,
        cv2_loss_coef_exp=cv2_loss_coef_exp,
        cv2_loss_tol_grp= cv2_loss_tol_grp,
        cv2_loss_tol_exp= cv2_loss_tol_exp,
        spec_loss_coef_grp=spec_loss_coef_grp,
        spec_loss_coef_exp=spec_loss_coef_exp,
        weight_decay=  weight_decay,
        monitor_es=    monitor_es,
        monitor_lr=    monitor_lr,
        scheduler_start_at_epoch=scheduler_start_at_epoch,
        batch_size=    batch_size,
        model_name=    model_name
        )

    print(aemose_model)

    # --- Training configuration ---

    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'

    callbacks = []
    callback_lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(callback_lr_monitor)
    callback_model_checkpoint = ModelCheckpoint(
        dirpath=dir_checkpoints, 
        save_top_k=10,
        mode="min", 
        monitor=monitor_es, 
        save_last=True,
        filename='{epoch}-{step}-{val_loss:.6f}'
        )
    # ckpt_min_save_epoch = gating_tau_ann_start_at_epoch + gating_tau_ann_len_at_epoch
    # if gating_hard_group or gating_hard_expert:
    #     # Allow a buffer after the latest hard switch so val_loss transient
    #     # does not dominate the saved top-k.
    #     latest_hard_epoch = 0
    #     if gating_hard_group:
    #         latest_hard_epoch = max(latest_hard_epoch, gating_hard_group_start_at_epoch)
    #     if gating_hard_expert:
    #         latest_hard_epoch = max(latest_hard_epoch, gating_hard_expert_start_at_epoch)
    #     ckpt_min_save_epoch = max(
    #         ckpt_min_save_epoch,
    #         latest_hard_epoch + 10
    #     )
    # print(f"Checkpoint minimum save epoch set to {ckpt_min_save_epoch}", flush=True)
    # callback_model_checkpoint = DelayedModelCheckpoint(
    #     dirpath=dir_checkpoints,
    #     save_top_k=10,
    #     mode="min",
    #     monitor=monitor_es,
    #     save_last=True,
    #     filename='{epoch}-{step}-{val_loss:.6f}',
    #     min_save_epoch=ckpt_min_save_epoch,
    #     )
    callbacks.append(callback_model_checkpoint)
    if earlystop:
        callback_early_stop = EarlyStopping(monitor=monitor_es, min_delta=earlystop_delta, patience=earlystop_patnc, verbose=False, mode='min')
        callbacks.append(callback_early_stop)
    # Themed rich progress bar
    theme = RichProgressBarTheme(
        description="green_yellow",
        progress_bar="deep_pink1",
        progress_bar_finished="deep_pink1",
        time="bright_blue",
        metrics="yellow1"
        )
    callback_rich_progress = RichProgressBar(theme=theme)
    callbacks.append(callback_rich_progress)

    logger_folder = dir_logs
    logger_test_name  = f'log_{model_name}'
    logger_tb = TensorBoardLogger(logger_folder, name=logger_test_name)
    logger_csv = CSVLogger(logger_folder, name=logger_test_name)

    print(f"Val | batch size: {batch_size} - num parts: {math.ceil(data_graph.valid_mask.sum() / batch_size)}")
    val_loader = NeighborLoader(
        data=data_graph,
        # include all neighbors for as many hops as gconv_dims
        num_neighbors=[-1]*(len(gconv_dims)),
        input_nodes=data_graph.valid_mask,
        # # include all edges between all sampled nodes
        # directed = False,
        # arguments to data loader
        batch_size=batch_size,
        shuffle=False,
        num_workers=loader_workers,
        pin_memory=True,
        persistent_workers=True
    )

    train_loader = NeighborLoader(
        data=data_graph,
        # include all neighbors for as many hops as gconv_dims
        num_neighbors=[-1]*(len(gconv_dims)),
        input_nodes=data_graph.train_mask,
        # # include all edges between all sampled nodes
        # directed = False,
        # arguments to data loader
        batch_size=batch_size,
        shuffle=True,
        num_workers=loader_workers,
        pin_memory=True,
        persistent_workers=True
        )


    trainer = L.Trainer(
        devices=1,
        max_epochs=max_epochs,
        min_epochs=min_epochs,
        accelerator=accelerator,
        logger=[logger_tb, logger_csv],
        callbacks=callbacks,
        enable_progress_bar=True
        )
    

    # --- Summary before training ---

    model_str  = f"Input dataset:\n {dataset_graph_path}"
    model_str += f"\n\n"
    model_str += "Config:\n"
    model_str += f"\nrandom_seed = {random_seed}"
    model_str += f"\nmax_epochs = {max_epochs}"
    model_str += f"\nmin_epochs = {min_epochs}"
    model_str += f"\nearlystop_delta = {earlystop_delta}"
    model_str += f"\nearlystop_patience = {earlystop_patnc}"
    model_str += f"\n\n"
    
    model_str += f"\nLoader type: {type(train_loader)}\n"
    model_str += f"\n\n"

    model_str += "\n\nAEMoE architecture:\n"
    model_str += str(aemose_model)

    model_str += "\n\nAutoencoderMoE hyperparameters:\n"
    for hparam in aemose_model.hparams:
        model_str += f"\t{hparam} = {aemose_model.hparams[hparam]}\n"
    model_str += "\nEncoder hyperparameters:\n"
    for hparam_name, hparam_val in aemose_model.encoder.__dict__.items():
        if not hparam_name.startswith('_'):
            model_str += f"\t{hparam_name} = {hparam_val}\n"
    model_str += "\nMoE Decoder hyperparameters:\n"
    for hparam_name, hparam_val in aemose_model.moe_decoder.__dict__.items():
        if not hparam_name.startswith('_'):
            model_str += f"\t{hparam_name} = {hparam_val}\n"
    model_str += "\tSingle expert (decoder) hyperparameters:\n"
    for hparam_name, hparam_val in aemose_model.moe_decoder.experts[0].__dict__.items():
        if not hparam_name.startswith('_'):
            model_str += f"\t\t{hparam_name} = {hparam_val}\n"
    model_str += "\nGatingNetwork MLP hyperparameters:\n"
    for hparam_name, hparam_val in aemose_model.gating_network.__dict__.items():
        if not hparam_name.startswith('_'):
            model_str += f"\t{hparam_name} = {hparam_val}\n"
    model_str += "\tPost-GNN MLP:\n"
    for hparam_name, hparam_val in aemose_model.gating_network.group_mlp.__dict__.items():
        if not hparam_name.startswith('_'):
            model_str += f"\t\t{hparam_name} = {hparam_val}\n"

    print(model_str, flush=True)
    with open(model_itmp_path, 'w') as f:
        f.write(model_str)


    # --- Training ---

    trainer.fit(
        aemose_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
        )


    # --- Summary after training ---

    model_str += "\nTotal number of parameters: "
    total_params = sum(p.numel() for p in aemose_model.parameters())
    model_str += str(total_params) + "\n"
    model_str += "Trainable parameters: "
    trainable_params = sum(p.numel() for p in aemose_model.parameters() if p.requires_grad)
    model_str += str(trainable_params) + "\n"
    model_str += "Non-trainable parameters: "
    non_trainable_params = total_params - trainable_params
    model_str += str(non_trainable_params) + "\n"

    aemose_model_best_epoch = int(re.search(r'epoch=(\d+)', callback_model_checkpoint.best_model_path).group(1))
    aemose_model_best_step  = int(re.search(r'step=(\d+)',  callback_model_checkpoint.best_model_path).group(1))

    model_str += f"\n\nTotal epochs trained: {trainer.current_epoch}\n"
    model_str += f"Best model (early stopping) epoch: {aemose_model_best_epoch}\n"
    model_str += f"Best model (early stopping) step: {aemose_model_best_step}\n"

    resu_str   = "model,best_epoch,best_step,checkpoint,split,metric,value\n"
    resu_str_ckpt = os.path.relpath(callback_model_checkpoint.best_model_path)

    # Load best checkpoint and evaluate to get best metrics
    print(f"\nSaving metrics from best checkpoint: {callback_model_checkpoint.best_model_path}")
    best_model_logger_name  = f'log_{model_name}_best_earlystop'
    best_model_logger_csv = CSVLogger(logger_folder, name=best_model_logger_name)
    best_model_logger_tb = TensorBoardLogger(logger_folder, name=best_model_logger_name)
    best_model_trainer = L.Trainer(
        devices=1,
        max_epochs=max_epochs,
        accelerator=accelerator,
        logger=[best_model_logger_tb, best_model_logger_csv],
        enable_progress_bar=True
    )
    best_model = AutoencoderMoSE.load_from_checkpoint(callback_model_checkpoint.best_model_path)
    
    best_model_trainer.validate(best_model, dataloaders=train_loader)
    model_str += f"\n\nMetrics at best early stopping epoch:\n"
    model_str += f"\ttrain_loss = {best_model_trainer.callback_metrics.get('val_loss', float('nan'))}\n"
    model_str += f"\ttrain_recon_loss = {best_model_trainer.callback_metrics.get('val_recon_loss', float('nan'))}\n"
    model_str += f"\ttrain_spec_loss_grp = {best_model_trainer.callback_metrics.get('val_spec_loss_grp', float('nan'))}\n"
    model_str += f"\ttrain_spec_loss_exp = {best_model_trainer.callback_metrics.get('val_spec_loss_exp', float('nan'))}\n"
    model_str += f"\ttrain_dead_experts = {best_model_trainer.callback_metrics.get('val_dead_experts', float('nan'))}\n"
    # model_str += f"\ttrain_recon_mae = {best_model_trainer.callback_metrics.get('val_recon_mae', float('nan'))}\n"
    resu_str += f"""{model_name},{aemose_model_best_epoch},{aemose_model_best_step},\"{resu_str_ckpt}\",train,loss,{best_model_trainer.callback_metrics.get('val_loss', float('nan'))}\n"""
    resu_str += f"""{model_name},{aemose_model_best_epoch},{aemose_model_best_step},\"{resu_str_ckpt}\",train,recon_loss,{best_model_trainer.callback_metrics.get('val_recon_loss', float('nan'))}\n"""
    resu_str += f"""{model_name},{aemose_model_best_epoch},{aemose_model_best_step},\"{resu_str_ckpt}\",train,spec_loss_grp,{best_model_trainer.callback_metrics.get('val_spec_loss_grp', float('nan'))}\n"""
    resu_str += f"""{model_name},{aemose_model_best_epoch},{aemose_model_best_step},\"{resu_str_ckpt}\",train,spec_loss_exp,{best_model_trainer.callback_metrics.get('val_spec_loss_exp', float('nan'))}\n"""
    resu_str += f"""{model_name},{aemose_model_best_epoch},{aemose_model_best_step},\"{resu_str_ckpt}\",train,dead_experts,{best_model_trainer.callback_metrics.get('val_dead_experts', float('nan'))}\n"""
    # resu_str += f"""{model_name},{aemose_model_best_epoch},{aemose_model_best_step},\"{resu_str_ckpt}\",train,recon_mae,{best_model_trainer.callback_metrics.get('val_recon_mae', float('nan'))}\n"""

    best_model_trainer.validate(best_model, dataloaders=val_loader)
    model_str += f"\tval_loss = {best_model_trainer.callback_metrics.get('val_loss', float('nan'))}\n"
    model_str += f"\tval_recon_loss = {best_model_trainer.callback_metrics.get('val_recon_loss', float('nan'))}\n"
    model_str += f"\tval_spec_loss_grp = {best_model_trainer.callback_metrics.get('val_spec_loss_grp', float('nan'))}\n"
    model_str += f"\tval_spec_loss_exp = {best_model_trainer.callback_metrics.get('val_spec_loss_exp', float('nan'))}\n"
    model_str += f"\tval_dead_experts = {best_model_trainer.callback_metrics.get('val_dead_experts', float('nan'))}\n"
    # model_str += f"\tval_recon_mae = {best_model_trainer.callback_metrics.get('val_recon_mae', float('nan'))}\n"
    resu_str += f"""{model_name},{aemose_model_best_epoch},{aemose_model_best_step},\"{resu_str_ckpt}\",val,loss,{best_model_trainer.callback_metrics.get('val_loss', float('nan'))}\n"""
    resu_str += f"""{model_name},{aemose_model_best_epoch},{aemose_model_best_step},\"{resu_str_ckpt}\",val,recon_loss,{best_model_trainer.callback_metrics.get('val_recon_loss', float('nan'))}\n"""
    resu_str += f"""{model_name},{aemose_model_best_epoch},{aemose_model_best_step},\"{resu_str_ckpt}\",val,spec_loss_grp,{best_model_trainer.callback_metrics.get('val_spec_loss_grp', float('nan'))}\n"""
    resu_str += f"""{model_name},{aemose_model_best_epoch},{aemose_model_best_step},\"{resu_str_ckpt}\",val,spec_loss_exp,{best_model_trainer.callback_metrics.get('val_spec_loss_exp', float('nan'))}\n"""
    resu_str += f"""{model_name},{aemose_model_best_epoch},{aemose_model_best_step},\"{resu_str_ckpt}\",val,dead_experts,{best_model_trainer.callback_metrics.get('val_dead_experts', float('nan'))}\n"""
    # resu_str += f"""{model_name},{aemose_model_best_epoch},{aemose_model_best_step},\"{resu_str_ckpt}\",val,recon_mae,{best_model_trainer.callback_metrics.get('val_recon_mae', float('nan'))}\n"""

    model_str += f"\n"
    print(model_str, flush=True)
    with open(model_info_path, 'w') as f:
        f.write(model_str)
    with open(model_resu_path, 'w') as f:
        f.write(resu_str)
    os.remove(model_itmp_path)

print("Running inference on best model checkpoint...")
subprocess.run(['python', 'code/aemose_inference.py', callback_model_checkpoint.best_model_path], check=True)
print("done.")
