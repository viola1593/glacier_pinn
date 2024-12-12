""" Leave-one-glacier-out training script for the PINN model."""

import argparse
from datetime import datetime
import json
import numpy as np
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from typing import Any, Dict
import matplotlib.pyplot as plt
from dataset_1d import PL_GlacierDataset
from model_1d import PL_PINN
import yaml

# Function to load yaml configuration file
def read_config(config_path: str):
    with open(config_path) as file:
        config = yaml.safe_load(file)
    return config

# Function to save yaml configuration file
def save_config(config: Dict[str, Any], path: str):
    with open(path, 'w') as file:
        yaml.dump(config, file)
           



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


def train(config_path):
    """
    Training Script.

    Args:
        config_path (str): The path to the configuration file.

    Returns:
        None
    """

    insample_rmses = []
    insample_mapes = []
    ood_rmses = []
    ood_mapes = []
    

    pl.seed_everything(42, workers=True)
    config = read_config(config_path)

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
    
    # glacier_data = PL_GlacierDataset(config)
    predictions = trainer.predict(model, datamodule=glacier_data)
    thicknesses = np.concatenate([pred[0] for pred in predictions]).ravel()
    velocities = np.concatenate([pred[1] for pred in predictions]).ravel()

    trainer.loggers[0].experiment.finish()

    return glacier_data, thicknesses, velocities, config["experiment"]["save_dir"]

def plot_predictions(save_dir, glacier_data, thicknesses_physics, velocities_physics):
        # plot predictions
        
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)
    plt.scatter(glacier_data.dataset.inputs['x'], thicknesses_physics, label="Predicted Ice Thickness", marker='.')
    plt.scatter(glacier_data.dataset.inputs['x'], glacier_data.dataset.target['ice_thickness'], label="True Ice Thickness",marker='.')
    plt.scatter(glacier_data.train_dataset.inputs['x'], glacier_data.train_dataset.target['ice_thickness'], label="Ice Thickness Training Points",marker='x')
    #plt.figtext(0.1, 0.05, s='PINN estimate: Val RMSE', eval[0]['val Thickness RMSE'],  ' Val MAPE ' , eval[0]['val MAPE'])
    plt.tick_params(axis='both', which='major', labelsize=12)  # Set tick font size
    plt.ylabel("Ice thickness [m]",fontsize=14)
    plt.title("Predicted and True ice thickness", fontsize=18)
    plt.legend(fontsize=14)

    plt.subplot(2, 1, 2)
    plt.scatter(glacier_data.dataset.inputs['x'], velocities_physics, label="Depth-averaged velocity",marker='.')
    plt.scatter(glacier_data.dataset.inputs['x'],glacier_data.dataset.inputs['surface_velocity'], label="Surface velocity",marker='.')
    plt.tick_params(axis='both', which='major', labelsize=12)  # Set tick font size

    plt.ylabel("Velocity [m/a]", fontsize=14)
    plt.title("Depth-avg vs Surface velocity",fontsize=18)
    plt.legend(fontsize=14)
    plt.savefig(save_dir, dpi=300)

   
def plot_physics_synthetic(save_dir, glacier_data, thicknesses_physics, velocities_physics, thicknesses_synthetic, velocities_synthetic):
        # plot predictions
        
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)
    plt.scatter(glacier_data.dataset.inputs['x'], thicknesses_physics, label="Predicted Ice Thickness", marker='.')
    plt.scatter(glacier_data.dataset.inputs['x'], glacier_data.dataset.target['ice_thickness'], label="True Ice Thickness",marker='.')
    plt.scatter(glacier_data.train_dataset.inputs['x'], glacier_data.train_dataset.target['ice_thickness'], label="Ice Thickness Training Points",marker='x')
    plt.figtext(-0.1, 1.05, '(a)', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=12)  # Set tick font size
    plt.ylabel("Ice thickness [m]",fontsize=14)
    plt.xlabel("Distance [m]",fontsize=14)
    plt.title("With physics-aware losses", fontsize=18)
    plt.legend(fontsize=14)

    plt.subplot(2, 1, 2)
    plt.scatter(glacier_data.dataset.inputs['x'], thicknesses_synthetic, label="Predicted Ice Thickness", marker='.')
    plt.scatter(glacier_data.dataset.inputs['x'], glacier_data.dataset.target['ice_thickness'], label="True Ice Thickness",marker='.')
    plt.scatter(glacier_data.train_dataset.inputs['x'], glacier_data.train_dataset.target['ice_thickness'], label="Ice Thickness Training Points",marker='x')
    plt.figtext(-0.1, 1.05, '(b)', fontsize=18)    
    plt.tick_params(axis='both', which='major', labelsize=12)  # Set tick font size
    plt.ylabel("Ice thickness [m]",fontsize=14)
    plt.xlabel("Distance [m]",fontsize=14)
    plt.title("Without physics-aware losses", fontsize=18)
    plt.legend(fontsize=14)
    plt.tight_layout()

    # plt.subplot(2, 1, 2)
    # plt.scatter(glacier_data.dataset.inputs['x'], velocities_physics, label="Depth-averaged velocity",marker='.')
    # plt.scatter(glacier_data.dataset.inputs['x'],glacier_data.dataset.inputs['surface_velocity'], label="Surface velocity",marker='.')
    # plt.tick_params(axis='both', which='major', labelsize=12)  # Set tick font size

    # plt.ylabel("Velocity [m/a]", fontsize=14)
    # plt.title("Depth-avg vs Surface velocity",fontsize=18)
    # plt.legend(fontsize=14)
    plt.savefig(save_dir, dpi=300) 




def start():
    """Start training."""
    # Command line arguments
    parser = argparse.ArgumentParser(
        prog="Trainer_synthetic.py", description="Training on a synthetic 1D dataset of a glacier."
    )

    parser.add_argument("--config_path", help="Path to the config file", required=True)

    args = parser.parse_args()

    glacier_data, thicknesses_p, velocities_p, save_dir = train(args.config_path)
    glacier_data, thicknesses_s, velocities_s, _ = train('config_1d_nophysics.yaml')
    plot_predictions(save_dir, glacier_data, thicknesses_p, velocities_p)
    plot_physics_synthetic(save_dir, glacier_data, thicknesses_p, velocities_p, thicknesses_s, velocities_s)

if __name__ == "__main__":
    start()


