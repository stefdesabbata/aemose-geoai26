import os
import sys
import math
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import torch
from torch_geometric.loader import NeighborLoader

from aemose import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='AEMoE Inference Script.')
    parser.add_argument('model_path_ckpt', type=str, help='Path to model checkpoint')
    parser.add_argument('--hard-group',  dest='hard_group',  action=argparse.BooleanOptionalAction, default=True,
                        help='Straight-through (hard) Gumbel-softmax for group routing at inference (default: True). Use --no-hard-group to disable.')
    parser.add_argument('--hard-expert', dest='hard_expert', action=argparse.BooleanOptionalAction, default=False,
                        help='Straight-through (hard) Gumbel-softmax for expert routing at inference (default: False). Use --hard-expert to enable.')
    args = parser.parse_args()

    dataset_nickname   = 'ukc2021-proc20251009001rs'
    dataset_graph_path  = f'data/pyg/{dataset_nickname}__pygdata.pth'
    dataset_oa_idx_path = f'data/pyg/{dataset_nickname}__oa_index.parquet'

    model_path_ckpt  = args.model_path_ckpt

    dir_run          = os.path.dirname(model_path_ckpt)
    dir_project      = os.path.dirname(dir_run)

    checkpoint  = torch.load(model_path_ckpt, weights_only=False)
    aemose_model = AutoencoderMoSE.load_from_checkpoint(checkpoint_path = model_path_ckpt)
    # Override checkpoint hparams to control inference gating
    aemose_model.gating_hard_group  = args.hard_group
    aemose_model.gating_hard_expert = args.hard_expert
    print(f"Inference gating: hard_group={aemose_model.gating_hard_group}, hard_expert={aemose_model.gating_hard_expert}")
    model_name  = aemose_model.hparams.model_name + '__ep' + str(checkpoint['epoch']) + '-st' + str(checkpoint['global_step'])
    batch_size  = aemose_model.hparams.batch_size
    gconv_dims   = aemose_model.hparams.gconv_dims

    data_graph       = torch.load(dataset_graph_path, weights_only=False)
    data_tensor_ncol = data_graph.x.shape[1]

    dataset_oa_idx = pd.read_parquet(dataset_oa_idx_path)

    # Save results
    # Generate latent in batches and combine results
    with torch.no_grad():
        aemose_model.eval()
        # aemose_model.model.eval()

        # Load the full dataset for inference
        latent_csv_path   = os.path.join(dir_project, f'{model_name}__latent.csv')
        latent_prq_path   = os.path.join(dir_project, f'{model_name}__latent.parquet')
        classif_gpkg_path = os.path.join(dir_project, f'{model_name}__classif.gpkg')

        latent_batches = []
        gating_weights_batches = []
        top_expert_batches = []
        top_group_batches = []

        data_loader_for_encoder = NeighborLoader(
            data=data_graph,
            # include all neighbors for as many hops as gconv_dims
            num_neighbors=[-1]*len(gconv_dims),
            # # include all edges between all sampled nodes
            # directed = False,
            # arguments to data loader
            batch_size=batch_size,
            shuffle=False # IMPORTANT: must be False to match the ids
            )
        for batch in data_loader_for_encoder:
                batch = batch.to(aemose_model.device)
                latent_batch, gating_weights, top_expert, top_group = aemose_model.exper_opinion(batch)
                latent_batch = latent_batch.cpu().detach().numpy()
                latent_batches.append(latent_batch)
                gating_weights = gating_weights.cpu().detach().numpy()
                gating_weights_batches.append(gating_weights)
                top_expert = top_expert.cpu().detach().numpy().reshape(-1, 1)
                top_expert_batches.append(top_expert)
                top_group = top_group.cpu().detach().numpy().reshape(-1, 1)
                top_group_batches.append(top_group)
        latent = np.vstack(latent_batches)
        gating_weights = np.vstack(gating_weights_batches)
        top_expert = np.vstack(top_expert_batches)
        top_group = np.vstack(top_group_batches)
        
        output_df = pd.concat([
            dataset_oa_idx,
            pd.DataFrame(latent, columns=[f'EMB_{i:03d}' for i in range(latent.shape[1])]),
            pd.DataFrame(gating_weights, columns=[f'EXR_{i:03d}' for i in range(gating_weights.shape[1])]),
            pd.DataFrame(top_expert, columns=['top_expert_index']),
            pd.DataFrame(top_group, columns=['top_group_index'])
        ], axis=1)
        output_df = output_df.set_index('OA')
        print(output_df.info())

        output_df['EXR_entropy'] = -np.sum(gating_weights * np.log(gating_weights + 1e-12), axis=1)
        output_df['top_expert_name'] = output_df['top_group_index'].astype(str) + "_" + output_df['top_expert_index'].astype(str).str.zfill(2)

        print(output_df.describe())

        print(output_df['top_expert_name'].value_counts().sort_index())

        output_df.to_parquet(latent_prq_path)
        # output_df.to_csv(latent_csv_path)

        # --- Generate Maps ---
        print("\nGenerating maps...")

        map_png_path = os.path.join(dir_project, f'{model_name}__map.png')

        spatial = gpd.read_file('data/spatial/OA/combined/uk_oa.gpkg')
        spatial = spatial.set_index('OA')
        spatial = spatial.join(output_df, how='inner')

        spatial.to_file(classif_gpkg_path, driver='GPKG')

        ldas    = gpd.read_file('data/spatial/LAD/Local_Authority_Districts_December_2021_GB_BGC_2022_7766355490887282253/LAD_DEC_2021_GB_BGC.shp')

        # Define consistent color mapping for all maps
        unique_experts = sorted(spatial['top_expert_name'].unique())
        cmap1 = plt.get_cmap('tab20b')
        cmap2 = plt.get_cmap('tab20c')
        combined = np.vstack([cmap1.colors, cmap2.colors])
        tab20bc  = ListedColormap(combined, name='tab20bc')
        results_cmap = 'tab20' if len(unique_experts) <= 20 else tab20bc
        cmap = plt.get_cmap(results_cmap, len(unique_experts))
        color_dict = {expert: cmap(i) for i, expert in enumerate(unique_experts)}

        def plot_map(spatial_data, ldas_data, ax, title=None):
            """Helper function to plot with consistent colors and legend."""
            spatial_data.plot(
                column='top_expert_name',
                ax=ax,
                categorical=True,
                legend=True,
                legend_kwds={'title': 'Top Expert', 'loc': 'upper left', 'bbox_to_anchor': (1, 1)},
                cmap=cmap,
                categories=unique_experts  # ensures consistent ordering
            )
            ldas_data.boundary.plot(color='black', linewidth=0.5, ax=ax)
            if title:
                ax.set_title(title)

        # UK
        fig, ax = plt.subplots(figsize=(16, 40))
        plot_map(spatial, ldas, ax, title='UK - Top Expert Index')
        ax.set_axis_off()
        plt.savefig(map_png_path, dpi=100, bbox_inches='tight')
        plt.close(fig)

        # Entropy map
        num_experts   = aemose_model.hparams.num_experts
        num_exp_total = num_experts[0] * num_experts[1]
        fig, ax = plt.subplots(figsize=(16, 40))
        spatial.plot(column='EXR_entropy', ax=ax, cmap='viridis', legend=True, vmin=0, vmax=math.log(num_exp_total), legend_kwds={'label': "Gating Entropy", 'shrink': 0.5})
        ldas.boundary.plot(color='black', linewidth=0.5, ax=ax)
        ax.set_axis_off()
        plt.savefig(map_png_path.replace('.png', '_entropy.png'), dpi=100, bbox_inches='tight')
        plt.close(fig)

        # London
        minx, miny, maxx, maxy = 500000, 155000, 565000, 205000
        spatial_filtered = spatial.cx[minx:maxx, miny:maxy]
        ldas_filtered = ldas.cx[minx:maxx, miny:maxy]
        fig, ax = plt.subplots(figsize=(12, 10))
        plot_map(spatial_filtered, ldas_filtered, ax, title='London - Top Expert Index')
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.set_axis_off()
        plt.savefig(map_png_path.replace('.png', '_london.png'), dpi=100, bbox_inches='tight')
        plt.close(fig)

        # Leicester
        minx, miny, maxx, maxy = 452000, 298000, 466000, 311000
        spatial_filtered = spatial.cx[minx:maxx, miny:maxy]
        ldas_filtered = ldas.cx[minx:maxx, miny:maxy]
        fig, ax = plt.subplots(figsize=(12, 10))
        plot_map(spatial_filtered, ldas_filtered, ax, title='Leicester - Top Expert Index')
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.set_axis_off()
        plt.savefig(map_png_path.replace('.png', '_leicester.png'), dpi=100, bbox_inches='tight')
        plt.close(fig)

        # Liverpool
        minx, miny, maxx, maxy = 327000, 380000, 347000, 399000
        spatial_filtered = spatial.cx[minx:maxx, miny:maxy]
        ldas_filtered = ldas.cx[minx:maxx, miny:maxy]
        fig, ax = plt.subplots(figsize=(12, 10))
        plot_map(spatial_filtered, ldas_filtered, ax, title='Liverpool - Top Expert Index')
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.set_axis_off()
        plt.savefig(map_png_path.replace('.png', '_liverpool.png'), dpi=100, bbox_inches='tight')
        plt.close(fig)

        # Glasgow
        minx, miny, maxx, maxy = 249000, 655000, 271000, 675000
        spatial_filtered = spatial.cx[minx:maxx, miny:maxy]
        ldas_filtered = ldas.cx[minx:maxx, miny:maxy]
        fig, ax = plt.subplots(figsize=(12, 10))
        plot_map(spatial_filtered, ldas_filtered, ax, title='Glasgow - Top Expert Index')
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.set_axis_off()
        plt.savefig(map_png_path.replace('.png', '_glasgow.png'), dpi=100, bbox_inches='tight')
        plt.close(fig)

        print("done.")
