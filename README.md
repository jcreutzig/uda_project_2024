# dynamic_stability_datasets_gnn_paper-companion

This repository contains the code attached to the project of the "Unstructured Data Analysis" Module.  

It mainly focuses on the reproduction of some results of the paper "Predicting Instability in Complex Oscillator Networks: Limitations and Potentials of Network Measures and Machine Learning" (https://arxiv.org/abs/2402.17500).  Please also see the pdf file "uda_project_creutzig.pdf".  

## Datasets
We used mainly two sets of synthetically generated data.  The generation of this data is described in "Toward dynamic stability assessment of power grid topologies using graph neural networks" with the DOI: https://doi.org/10.1063/5.0160915, and is downloaded here from the Zenodo: https://zenodo.org/records/8204334.  

Download and setup of the datasets is handled in the file load_data.py.  The datasets are stored in the folders "data_download_zip"/"data".  

## Training of the ML models
We generate a larger TAG model in several variations, and demonstrate the training impact of hyperparameter choices.  

To this end we provide the files 

tag_model.py - main files for the TAG model
train_tag.py - main file used for generating for training the TAG model

Trained models are stored in the subfolders trained_models/, while the training results are stored in the subfolder training_hist/

## Setup 

We provide a conda env file 'conda_environment.yml' which is tailored to use python 3.12.  

We also provide a jupyter notebook that can run the loader and training of data, and also contains several plots and analytics of the training results.  

## License 

The content of this github repo is licensed under the MIT License.  For details read LICENSE.txt 

## Contact

jc8423@ic.ac.uk  

[]: # (END)
