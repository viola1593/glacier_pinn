# Ice thickness estimates from mass conserving PINN
Mass conserving physics-informed neural network (PINN) to estimate glacier ice thickness and the code to produce the dataset to train and test it as described in Steidl et al.: 'Physics-aware Machine Learning for Glacier Ice Thickness Estimation: A Case Study for Svalbard'
## Preprocessing
The preprocessing of the data is done using a conda environment with oggm (environment_oggm.yaml). 
The data used for the preprocessing is:
- Ice thickness data from GlaThiDa version 3.1.0, TTT.csv, downloaded from https://www.gtn-g.ch/data_catalogue_glathida/ (16.1.23) as zipfile. 
- The elevation change map (dh/dt) for Svalbard for the time period of Jan 2015- Dec 2019 from https://doi.org/10.6096/13 (related publication: Hugonnet et al. (2021): "Accelerated global glacier mass loss in the early twenty-first century"), licensed under the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/). Downloaded Feb 22, 2024
All the other data sources are provided through OGGM. 

1) linking_glathida_RGI.ipynb: Ice thickness data from GlaThiDa (Version 3.1.0) are linked to the corresponding RGI IDs. 
2) create_dataset.ipynb: Then the dataset including any auxiliary data to train and test the PINN is created using OGGM (Version 1.6). 

## Model training and cross validation

Cross validation procedure including the training of the model happens in LOGO_CV.py. To execute the CV procedure you need to provied a config.yaml file that provides information on how to set up the experiment, the model architecture and hyperparameters, loss weights, data loading and where to find the data. model_config.yaml is the configuration that created the results reported in the manuscript. 

The results of the cross validation, including the train and test datasets as .csv files for every training fold, are saved to the folder specified in the config file. 
The environment to run the PINN model can be created from the environment.yml. 

To try the PINN LOGO_CV just run LOGO_CV.py --config_path path/to/config_file

The processed training data and CV results can be found at 10.5281/zenodo.11474955.

## Creating Figures
The figures in the manuscript were created by running
1) plot_dataset.ipynb: plots the study area with acquisition lines of ice thickness measurements and the LOGO test glaciers. The coastline of Svalbard in the figures showing the whole study area is taken from Moholdt, G., Maton, J., Majerska, M., Kohler, J. (2021). Annual coastlines for Svalbard [Data set]. Norwegian Polar Institute. https://doi.org/10.21334/npolar.2021.21565514
(Last accessed: 24.04.2024), licensed under the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/). 

2) make_CV_predictions.py: loads every model from the LOGO_CV, makes predictions of ice thicknesses for the whole study area, and stores the result in a new .csv file.

3) make_PINN_pred_figures.py: loads the predictions we just made and generates a mean over the predictions of the 7 LOGO model's predictions. The mean is saved to two separate .csv files, one for all the unlabbeld grid points and one for the labelled grid points. The scatter plots comparing the mean predicted ice thickness and other ice thickness estimates, and the plots of mean and coefficient of variation for every grid point in the study area are created.

4) make_LOGO_figures.py: loads every model from the LOGO_CV, makes predictions of ice thicknesses *only* for the LOGO test glaciers, and stores the result in a new .csv file. Then the ice thickness estimates for each of the glacier is plotted in a single figure. Depending on the command line instructions, either only the residuals to the measured ice thicknesses are plotted or only the predicted ice thickness, or both are are plotted together.


## 1d synthetic experiment
We tested the model on a synthetic dataset (see Appendix). The code to generate the synthetic data and run the experiments, including the config file is in [1d_synthetic_experiment](1d_synthetic_experiment)

## Citation 
Steidl, V., Bamber, J. L., and Zhu, X. X.: Physics-aware Machine Learning for Glacier Ice Thickness Estimation: A Case Study for Svalbard, EGUsphere [preprint], https://doi.org/10.5194/egusphere-2024-1732, 2024. 

## License
The code is licensed under the MIT license. T