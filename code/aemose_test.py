import os
import argparse
import torch
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from torch_geometric.loader import NeighborLoader

from aemose import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='AEMoE Test Script.')
    parser.add_argument('model_path_ckpt', type=str, help='Path to model checkpoint')
    args = parser.parse_args()

    # dataset_path        = 'data/28714514_2021UKCensus_processed_20251009001.parquet'
    dataset_nickname    = 'ukc2021-proc20251009001rs'
    dataset_graph_path  = f'data/pyg/{dataset_nickname}__pygdata.pth'

    model_path_ckpt  = args.model_path_ckpt
    
    dir_run          = os.path.dirname(model_path_ckpt)
    dir_project      = os.path.dirname(dir_run)
    dir_logs         = os.path.join(dir_project, 'logs_test')

    checkpoint  = torch.load(model_path_ckpt, weights_only=False)
    aemose_model_best_epoch = checkpoint['epoch']
    aemose_model_best_step  = checkpoint['global_step']
    aemose_model = AutoencoderMoSE.load_from_checkpoint(checkpoint_path = model_path_ckpt)
    model_name  = aemose_model.hparams.model_name
    batch_size  = aemose_model.hparams.batch_size
    gconv_dims  = aemose_model.hparams.gconv_dims

    max_epochs           = 0
    model_loader_workers = 4

    model_resu_path  = os.path.join(dir_project, f'{model_name}__results.csv')


    # --- Load Data ---

    data_graph       = torch.load(dataset_graph_path, weights_only=False)
    data_tensor_ncol = data_graph.x.shape[1]
    print(f"Graph data splits\n Test: {data_graph.test_mask.sum()}\n\n")

    # --- Testing ---

    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'

    with torch.no_grad():
        aemose_model.eval()

        logger_folder = dir_logs
        logger_test_name  = f'log_{model_name}'
        logger_tb = TensorBoardLogger(logger_folder, name=logger_test_name)
        logger_csv = CSVLogger(logger_folder, name=logger_test_name)

        test_loader = NeighborLoader(
            data=data_graph,
            # include all neighbors for as many hops as gconv_dims
            num_neighbors=[-1]*len(gconv_dims),
            input_nodes=data_graph.test_mask,
            # # include all edges between all sampled nodes
            # directed = False,
            # arguments to data loader
            batch_size=batch_size,
            shuffle=False,
            num_workers=model_loader_workers,
            pin_memory=True,
            persistent_workers=True
            )

        trainer = L.Trainer(
            devices=1,
            max_epochs=max_epochs,
            accelerator=accelerator,
            logger=[logger_tb, logger_csv],
            enable_progress_bar=True
            )

        trainer.test(aemose_model, dataloaders=test_loader)

        with open(model_resu_path, 'a') as f:
            resu_str  = ""
            resu_str += f"""{model_name},{aemose_model_best_epoch},{aemose_model_best_step},\"{model_path_ckpt}\",test,loss,{trainer.callback_metrics.get('test_loss', float('nan'))}\n"""
            resu_str += f"""{model_name},{aemose_model_best_epoch},{aemose_model_best_step},\"{model_path_ckpt}\",test,recon_loss,{trainer.callback_metrics.get('test_recon_loss', float('nan'))}\n"""
            resu_str += f"""{model_name},{aemose_model_best_epoch},{aemose_model_best_step},\"{model_path_ckpt}\",test,spec_loss_grp,{trainer.callback_metrics.get('test_spec_loss_grp', float('nan'))}\n"""
            resu_str += f"""{model_name},{aemose_model_best_epoch},{aemose_model_best_step},\"{model_path_ckpt}\",test,spec_loss_exp,{trainer.callback_metrics.get('test_spec_loss_exp', float('nan'))}\n"""
            resu_str += f"""{model_name},{aemose_model_best_epoch},{aemose_model_best_step},\"{model_path_ckpt}\",test,dead_experts,{trainer.callback_metrics.get('test_dead_experts', float('nan'))}\n"""
            # resu_str += f"""{model_name},{aemose_model_best_epoch},{aemose_model_best_step},\"{model_path_ckpt}\",test,recon_mae,{trainer.callback_metrics.get('test_recon_mae', float('nan'))}\n"""
            f.write(resu_str)
