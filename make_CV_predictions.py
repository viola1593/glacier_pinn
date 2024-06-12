"""Make predictions for all CV models and save them in a csv file. Also create some plots to evaluate the predictions."""
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import os

import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer

from dataset_new import PL_GlacierDataset
from model import PL_PINN
from utils import read_config, predictions_on_data_with_groundtruth, predictions_on_data_without_groundtruth



def make_predicitons(gpu):
    # set path to the datasets that should be used for the predictions
    path_measurements = "data/spitsbergen_measurements_aggregated_nosurges_dhdt2014smoothed_complete.csv"
    path_grid ="data/spitsbergen_allunmapped_griddeddata_nosurges_dhdt2014smoothed_complete.csv"

    # set path to experiment directory where all the CV models are stored
    directory = "CV/allunmappedglaciers_notsurging/reproduce_tests/test30_spitsbergen_completedhdt_correct_depth_avg"
    # create dataframes to store the predictions of every CV fold model
    grid_results_df = pd.DataFrame()
    measurement_results_df = pd.DataFrame()

    # goimg through all the directories in the experiment directory and check if there is a checkpoint file to load the model and make predictions
    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        if os.path.isdir(subdir_path):
            # set the seed for reproducibility
            pl.seed_everything(42, workers=True)
            try: 
                checkpoint_path = os.path.join(subdir_path, "checkpoints", "last.ckpt")
                loaded_model = PL_PINN.load_from_checkpoint(checkpoint_path)
                config = read_config(subdir_path+'/config.yaml') # needs to be the same config file as the one used for training to get the correct scaling parameters
            except:
                print("\n No checkpoint file found in", subdir_path, "\n")
                continue
            rgi_id = subdir_path.split("_")[-3]
            # make pl trainer that will take care of the predictions
            trainer = Trainer(accelerator='gpu', devices=[gpu])



            # prepare the config file for to initialize the datasets
            if 'num_points' in config["ds"]:
                del config["ds"]["num_points"] # we want to make predicitons for all points of the dataset
            config["ds"]["glacier_ids"] = [] # we want to make predicitons for all glaciers of the dataset

            # prediction on grid --> only unlabeled data
            config["ds"]["labeled_sample_size"]=0.0
            config["ds"]["unlabeled_sample_size"]=1.0
            config["ds"]["data_dir_unlabeled"]=[path_grid]

            ## create dataset 
            gridded_dataset = PL_GlacierDataset(config)
            
            print('---------------------------------')
            print('Predicting for model trained without ', rgi_id)
            print('---------------------------------')
            lonlat, pred_thicknesses_on_grid, velocities_on_grid = predictions_on_data_without_groundtruth(trainer, loaded_model, config, grid_dataset=gridded_dataset)
            #lonlat is in epsg4326

            # put all the predictions in a dataframe together with other the estimates of millan and the consensus estimate
            df_unlabeled_pred = pd.DataFrame({'x':lonlat.iloc[:,0], 
                                        'y':lonlat.iloc[:,1], 
                                        'pinn_thickness':pred_thicknesses_on_grid, 
                                        'consensus_thickness': gridded_dataset.dataset.unlabeled_dataset.consensus_ice_thickness.values,
                                        'millan_ice_thickness':gridded_dataset.dataset.unlabeled_dataset.millan_ice_thickness.values,
                                        'depth_avg_vel_x':velocities_on_grid[:,0],
                                        'depth_avg_vel_y':velocities_on_grid[:,1],
                                        'rgi_id':rgi_id})   
            # append the predictions to the dataframe                               
            grid_results_df =pd.concat([grid_results_df, df_unlabeled_pred])

        #--------------------------------------------------------------------------------------------------------

            # predictions on points where we have measurements(=ground truth data)--> only labelled data
            config["ds"]["data_dir_labeled"]=[path_measurements]
            config["ds"]["labeled_sample_size"]=1
            config["ds"]["unlabeled_sample_size"]=.0
            measurements_dataset = PL_GlacierDataset(config)
            coordinates, pred_thicknesses_on_measurements, velocities_on_measurements, _ = predictions_on_data_with_groundtruth(trainer, loaded_model, config, gt_dataset=measurements_dataset) #coordinates are in epsg4326(lon lat)

            # put all the predictions in a dataframe together with other the estimates of millan and the consensus estimate
            df_labeled_pred = pd.DataFrame({'x':coordinates.iloc[:,0], 
                                        'y':coordinates.iloc[:,1], 
                                        'pinn_thickness':pred_thicknesses_on_measurements, 
                                        'true_thickness':measurements_dataset.dataset.target.iloc[:,0].values, 
                                        'consensus_thickness': measurements_dataset.dataset.labeled_dataset.consensus_ice_thickness.values,
                                        'millan_ice_thickness':measurements_dataset.dataset.labeled_dataset.millan_ice_thickness.values,
                                        'depth_avg_vel_x':velocities_on_measurements[:,0],
                                        'depth_avg_vel_y':velocities_on_measurements[:,1],
                                        'rgi_id':rgi_id})
            # append the predictions to the dataframe                               
            measurement_results_df =pd.concat([measurement_results_df, df_labeled_pred])

    # save the predictions as csv files
    grid_results_df.rename(columns={'rgi_id': 'LOGO_rgi'}).to_csv(directory+"/predictions_on_grid.csv", index=False)
    measurement_results_df.rename(columns={'rgi_id': 'LOGO_rgi'}).to_csv(directory+"/predictions_on_measurements.csv", index=False)

    #--------------------------------------------------------------------------------------------------------
    # creating some plots to look at the predicitons

    # group all predictions by each point, so that we get the prediction of all 7 models at this point
    results_group = grid_results_df.groupby(['x', 'y'])
    # calculate the mean thickness
    mean_thickness = results_group['pinn_thickness'].mean()
    print(len(results_group))

    lon, lat = mean_thickness.index.get_level_values(0), mean_thickness.index.get_level_values(1)
    plt.figure(figsize=(30,20))
    plt.scatter(lon, lat, c=mean_thickness, cmap='viridis', s=1)
    plt.tick_params(labelsize=20)
    cbar = plt.colorbar(label='Thickness [m]')
    cbar.ax.tick_params(labelsize=20)
    plt.title('Mean thickness over all 7 models', fontsize=20)

    plt.savefig(directory+'/mean_thickness_on_grid.png')

    # calculate std and coefficient of variance for each point
    std_df = results_group['pinn_thickness'].std()
    coeff_var = std_df/mean_thickness

    print(std_df)
    plt.figure(figsize=(30,20))
    plt.scatter(lon,lat, c=coeff_var,  s=1,  cmap='Reds' )
    plt.tick_params(labelsize=20)
    cbar = plt.colorbar(label='Coefficient of Variance')
    cbar.ax.tick_params(labelsize=20)
    plt.title('Coefficient of Variation for all 7 models', fontsize=20)
    plt.savefig(directory+'/coeff_var_thickness_on_grid.png')




def start():
    parser = argparse.ArgumentParser(
    prog="make_CV_predictions.py", description="Make predictions for all CV models."
    )
    parser.add_argument("--gpu", 
                        help="Set the id of the gpu to use for predictions. Default is 0.", 
                        required=False, 
                        default=0,
                        type=int)
    args = parser.parse_args()
    make_predicitons(gpu=args.gpu)
   

if __name__ == "__main__":

    start()
