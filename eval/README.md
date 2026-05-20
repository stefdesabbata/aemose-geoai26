# Evaluation

This folder contains the code for the evaluation of the model.

- `metrics_plot.ipynb` plots the traning and validation losses.
- `metrics_compare.ipynb` compares the generated expert assignments to the 2021 Output Area Classification.
- `police_knn_dataprep.ipynb` prepares the evaluation targets for a downstream task using the [crimes reported in 2025 in England and Wales](https://data.police.uk/data/archive/).
- `police_knn_estimate.ipynb` evaluates the generated expert assignments and embeddings on the downstream task of estimating the proximity to different categories of [crimes reported in 2025 in England and Wales](https://data.police.uk/data/archive/).
