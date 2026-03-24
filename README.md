# Installation

The dependency management is done using [poetry](https://python-poetry.org/). Once poetry is on your system, you can install the dependencies using
```bash
poetry install
```

Depending on how poetry was configured and what operating system you are on, you might need to activate your virtural environment first with
```bash
source .venv/bin/activate
```

# Usage

Once all depencencies are installed, you can train a model with the baseline configuration using
```bash
python src/train.py
```
The datasets should all automatically be downloaded and the model weights will be saved to `src/datasets/.data/model.pt`.

You can then export the predictions of the model to `src/datasets/.data/predictions` as images using
```bash
python src/save_predictions.py
```

Alternatively, you can run
```bash
python src/compare.py
```
to compare a pre-defined set of model configurations. A .json file with the individual performance metrics will be stored to `src/datasets/.data/compare.json`.


# Contributing

Please install [pre-commit](https://pre-commit.com/) before contributing to this repo to keep the code quality high.
