# glacier_pinn
Repository containing the code for the mass conserving physics-informed neural network (PINN) to estimate glacier ice thickness and the code to produce the dataset to train and test it. 
The preprocessing of the data is done using a conda environment with oggm (environment_oggm.yaml). First the ice thickness data from GlaThiDa (Version 3.1.0) is linked to the corresponding RGI IDs. Then the dataset is created (create_dataset.ipynb) using OGGM. 
The data used for the preprocessing is 
1) GlaThiDa data version 3.1.0, TTT.csv, downloaded from https://www.gtn-g.ch/data_catalogue_glathida/ (16.1.23) as zipfile. 
2) dh/dt data from Hugonnet et al. (2021): "Accelerated global glacier mass loss in the early twenty-first century", licensed under the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/). Downloaded Feb 22, 2024
All the other data sources are provided through OGGM. 

Cross validation procedure including the training of the model happens in LOGO_CV.py 
Results are saved to a folder, including the train and test datasets as csv files for every training fold. The results are discussed in ....

The processed training data and final results are stored in ....