oct-diagn-semi-supervised
==============================

Semi-supervised classification of Retinal OCT images

<img src="reports/figures/dataset_sample.png" alt="drawing" width="300"/>

Project is based on:
- [Cortex](https://github.com/rdevon/cortex) - wrapper around Pytorch
- [MlFlow](https://mlflow.org/) - experiments tracking

ArXiv link: TODO

Project Organization
------------

    ├── LICENSE
    ├── README.md 
    ├── data
    │   └── OCT2017                     <- Dataset, loaded from https://www.kaggle.com/paultimothymooney/kermany2018, see src/data/download.py
    |       ├── test      
    |       ├── train     
    │       └── val       
    │
    ├── notebooks                       <- Jupyter notebooks
    |   ├── 00_dataset_statistics.ipynb <- OCT2017 exploratory data analysis
    |   └── 01_experiments.ipynb        <- Plots/graphs for experimental results  
    │
    ├── reports                         <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures                     <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt                <- The requirements file for reproducing the analysis environment
    │
    ├── setup.py                        <- makes project pip installable (pip install -e .) so src can be imported
    │
    └── src                             <- Source code for use in this project
        ├── data                        <- Scripts to download and manipulate data
        │   ├── dataset_plugins.py      <- Modification of default cortex DataLoaders for SSL
        │   ├── download.py             <- Script for data download
        │   ├── rand_augment.py         <- RandAugment functions
        │   └── transforms.py           <- Addiotional augmentations
        │
        └── models                      <- Scripts to train models and then use trained models to make predictions
            ├── fix_match               <- FixMatch
            |   ├── controller.py               <- Cortex controller 
            |   ├── main.py                     <- Train script 
            |   ├── utils.py            
            |   └── varying_number_of_labels.py <- Hyperparameter search / varying number of labels runs
            |   
            ├── full_supervised         <- Fully Supervised / Transfer Learning models
            |   ├── ...                          <- Same structure, as in src/models/fix_match
            |
            ├── mix_match               <- MixMatch
            |   ├── ...                         <- Same structure, as in src/models/fix_match
            |
            ├── utils.py
            └── wideresnet.py           <- Wide-Res-Net backbone
--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
