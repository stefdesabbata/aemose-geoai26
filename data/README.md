# Data

The data used for this project are the [Unified UK Census Data (2021/2)](https://data.geods.ac.uk/dataset/unified-uk-census-data), available via the [Geographic Data Service](https://data.geods.ac.uk/). In running order:

1. `data_prep_20251009001.ipynb` loads the signle tables, combines them into a single dataframe and applies a few simple pre-processing steps using the functions defined in `sodaprep.py`.
2. `data_prep_shuffle.ipynb` shuffles the order of the rows to ensure any subsequent training or test is conducted on a random order, and ` data_prep_shuffle_check.ipynb` checks that the resulting shuffled dataset contains all and the same data as the original one.
3. `data_prep_pygdata.ipynb` creates the PyTorch Geometric dataset used for modelling.

The notebooks are designed to run using the Pixi environment defined in `pixi.toml` from the root directory.

### External resources

The notebooks and scripts in this repository use the following datasets:

- [Unified UK Census Data (2021/2)](https://data.geods.ac.uk/dataset/unified-uk-census-data) available via the [Geographic Data Service](https://data.geods.ac.uk/).
- [2021 Output Area Classification](https://data.geods.ac.uk/dataset/output-area-classification-2021) available via the [Geographic Data Service](https://data.geods.ac.uk/).
- Combined output area and data zone boundaries from the [Office for National Statistics](https://geoportal.statistics.gov.uk/datasets/ons::output-areas-december-2021-boundaries-ew-bgc-v2/about), [National Records of Scotland](https://www.nrscotland.gov.uk/publications/2022-census-geography-products/) and [Northern Ireland Statistics and Research Agency](https://www.nisra.gov.uk/support/geography/data-zones-census-2021).
- [Local Authority Districts](https://geoportal.statistics.gov.uk/datasets/ons::local-authority-districts-december-2021-boundaries-gb-bgc-1/about) from the Office for National Statistics and [Government Districts](https://www.nisra.gov.uk/support/geography/data-zones-census-2021) from the Northern Ireland Statistics and Research Agency.
- [Crimes reported in 2025 in England and Wales](https://data.police.uk/data/archive/) available from the [Police UK data portal](https://data.police.uk/data) (download the December 2025 archive).
