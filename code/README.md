# Code

This folder contains the code necessary to prepare the datasets and train the model.

- `aemose.py` defines the AE-MoSE architecture.
- `aemose_train.py` defines the training procedure.
- `aemose_test.py` test a model checkpoint on the test split.
- `aemose_inference.py` uses a model checkpoint to generate the embeddings and expert assignments for the whole dataset.

The scripts are designed to run using the Pixi environment defined in `pixi.toml` from the root directory.
