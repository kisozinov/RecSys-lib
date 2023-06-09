recsys-lib
==============================

This repository implements ALS and LightFM baselines, and also LightGCN model architectures for recommender systems.

### Usage
You need to go to the `recsys-gcn-gan/` directory:
```
cd recsys-gcn-gan/
```
Also you need `amazmemllib` package to get ALS/LightFM metrics. The command line to start using project is the following:
```
python main.py [COMMAND] [OPTIONS]
```

### Commands
* `train` - Launch model training on the selected dataset and saving binaries/checkpoints to future usage (`models/`). If LightGCN, additionally plot loss & metrics curves (saving to `reports/figures/`)
* `evaluate` - Print evaluation metrics of the selected model on the selected dataset. 

### Options
The set of options is the same for both commands. 
* `-m`: Model name ('als', 'lfm', 'lgcn')
* `-d`: Dataset name ('ml-1m', 'gowalla', 'yelp2018', 'amazon-books')

### Configurate
Baseline models (ALS, LightFM) can be configured directly in initialization in `recsys-gcn-gan/main.py`.
If you want to change LightFM configuration, edit `recsys-gcn-gan/models/lightgcn_cfg.py`

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    │
    ├── recsys-gcn-gan     <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │    
    │   ├── main.py        <- Executable script to run train/test    
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── dataloader.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── lightgcn.py
    │   │   └── lightgcn_cfg.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
