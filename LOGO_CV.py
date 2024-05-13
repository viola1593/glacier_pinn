""" Leave-one-glacier-out training script for the PINN model."""

import argparse
from datetime import datetime
import json
import numpy as np
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from typing import Any, Dict

from dataset_new import PL_GlacierDataset
from model import PL_PINN
from utils import read_config, save_config, predictions_on_data_with_groundtruth, predictions_on_data_without_groundtruth, plot_thickness_pred, plot_difference_to_measurement, plot_velocity_predictions, plot_glacier





# Define LOGO Test Glaciers that will be left out alternatingly
GLACIER_IDS = [
'RGI60-07.00240',
'RGI60-07.00344',
'RGI60-07.00496',
"RGI60-07.00497",
'RGI60-07.01100',
'RGI60-07.01481',
'RGI60-07.01482'
]        
               



def generate_trainer(config: Dict[str, Any]) -> pl.Trainer:
    """Generate a pytorch lightning trainer."""
    loggers = [
        
        pl.loggers.WandbLogger(
            name=config["experiment"]["save_dir"].split("/")[-1],
            save_dir=config["experiment"]["save_dir"],
            project=config["wandb"]["project"],
            resume="allow",
            config=config,
            mode=config["wandb"].get("mode", "online"),
        ),
        pl.loggers.CSVLogger(config["experiment"]["save_dir"], name="csv_logs"),
    ]

    track_metric = "val Thickness RMSE"
    mode = "min"

    callback_list =[]
    checkpoint_callback = ModelCheckpoint(
        dirpath=config["experiment"]["save_dir"]+"/checkpoints",
        save_last=True,
        save_top_k=3,
        monitor=track_metric,
        mode=mode,
        every_n_epochs=1,
    )
    callback_list.append(checkpoint_callback)

    if "swa_lrs" in config["optimizer"]:
        swa_callback = StochasticWeightAveraging(swa_lrs=config["optimizer"]["swa_lrs"], 
                                                 swa_epoch_start=config["optimizer"]["swa_epoch_start"], 
                                                 annealing_epochs=config["optimizer"]["swa_annealing_epochs"]) 
        callback_list.append(swa_callback)
    return pl.Trainer(
        **config["pl"],
        default_root_dir=config["experiment"]["save_dir"],
        logger=loggers,      
        callbacks=callback_list,
    )


def create_experiment_dir(config: Dict[str, Any]) -> str:
    """Create experiment directory.

    Args:
        config: config file

    Returns:
        config with updated save_dir
    """
    os.makedirs(config["experiment"]["exp_dir"], exist_ok=True)
    exp_dir_name = (
        f"{config['experiment']['experiment_name']}"
        f"_{datetime.now().strftime('%m-%d-%Y_%H:%M:%S')}"
    )
    exp_dir_path = os.path.join(config["experiment"]["exp_dir"], exp_dir_name)
    os.makedirs(exp_dir_path)
    config["experiment"]["save_dir"] = exp_dir_path

    return config


def LOGO_CV(config_path):
    """
    Training Script with LOGO CV.

    Args:
        config_path (str): The path to the configuration file.

    Returns:
        None
    """

    insample_rmses = []
    ood_rmses = []
    
    for glacier_id in GLACIER_IDS:
        pl.seed_everything(42, workers=True)
        config = read_config(config_path)
        print('Training on glaciers except: ', glacier_id)
        config["ds"]["glacier_ids"] = [glacier_id]
        config["experiment"]["experiment_name"] = config["experiment"]["experiment_name"]+f"LOGO_{glacier_id}"


        config = create_experiment_dir(config)

        # generate trainer
        trainer = generate_trainer(config)

        # generate model
        model = PL_PINN(config)
        # get data
        glacier_data = PL_GlacierDataset(config)

        # configure logging
        trainer.loggers[0].watch(model, log='all')
        trainer.log_every_n_steps = config["pl"]["log_every_n_steps"] 
        
        # train model
        trainer.fit(model, datamodule=glacier_data)


        # save scaling parameters 
        config["ds"]["transformation_features_mean"] = glacier_data.scaler.mean_.tolist()
        config["ds"]["transformation_features_var"] = glacier_data.scaler.scale_.tolist()
        config["ds"]["transformation_target_mean"] = glacier_data.target_scaler.mean_.tolist()
        config["ds"]["transformation_target_var"] = glacier_data.target_scaler.scale_.tolist()
        config["ds"]["len_traindataset"]=glacier_data.train_dataset.__len__()
        
        # save config in experiment directory to be able to reproduce results
        save_config(config, os.path.join(config["experiment"]["save_dir"], "config.yaml"))
       

        # evaluation on in-smaple validation set

        eval = trainer.validate(model, datamodule=glacier_data)
        print(eval)
        insample_rmses.append(eval[0]["val Thickness RMSE"])
        with open(os.path.join(config["experiment"]["save_dir"],'evaluation_on_test'), "w") as fp:
                    json.dump(eval,fp)

   
    
        # plot results
        gt_coordinates, _, _, residuals = predictions_on_data_with_groundtruth(trainer, model, config, path=config["ds"]["data_dir_labeled"])
        grid_coordinates, grid_thicknesses, velocities = predictions_on_data_without_groundtruth(trainer, model, config, path=config["ds"]["data_dir_unlabeled"])

        plot_thickness_pred(grid_coordinates, grid_thicknesses, eval, config)
        plot_difference_to_measurement(gt_coordinates, grid_coordinates, residuals, grid_thicknesses,  config, eval, title='thickness_predVStruth_nobounds.png', vmax=None,s=1)
        plot_velocity_predictions(grid_coordinates, velocities, config)
        ood_glacier_eval = plot_glacier(glacier_id, trainer, model, config)
        ood_rmses.append(ood_glacier_eval[0]["val Thickness RMSE"])
        
        trainer.loggers[0].experiment.finish()

    print("Successfully completed LOGO CV.")
    print("In-sample RMSE scores were: ", insample_rmses)
    print("Mean in-sample RMSE: ", np.mean(insample_rmses))
    print("RMSE scores on ood glaciers were: ", ood_rmses)
    print("Mean ood RMSE: ", np.mean(ood_rmses))

    # Create a dictionary to save rmse scores
    data = {
        "mean_rmse": np.mean(insample_rmses),
        "rmses": insample_rmses,
        "ood_glacier_rmses": ood_rmses,
        "ood_mean_rmse": np.mean(ood_rmses),
    }

    # Write the dictionary to a file
    with open(os.path.join(config["experiment"]["exp_dir"], 'evaluation.json'), "w") as fp:
        json.dump(data, fp)



def start():
    """Start training."""
    # Command line arguments
    parser = argparse.ArgumentParser(
        prog="LOGO_CV.py", description="Automated leave-one-glacier-out cross-validation."
    )

    parser.add_argument("--config_path", help="Path to the config file", required=True)

    args = parser.parse_args()

    LOGO_CV(args.config_path)


if __name__ == "__main__":
    start()


