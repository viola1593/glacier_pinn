"""Has utility functions for reading and saving configuration files, transforming coordinates, and plotting predictions."""
import geopandas as gpd
from shapely.geometry import LineString, box

import numpy as np
import pandas as pd
import pyproj
import yaml
import os
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from model import PL_PINN
from dataset_new import PL_GlacierDataset
from typing import Any, Dict, Union, Tuple
from matplotlib.ticker import FormatStrFormatter, MaxNLocator


# Function to load yaml configuration file
def read_config(config_path: str):
    with open(config_path) as file:
        config = yaml.safe_load(file)
    return config

# Function to save yaml configuration file
def save_config(config: Dict[str, Any], path: str):
    with open(path, 'w') as file:
        yaml.dump(config, file)


def transform_projection(coordinates: Union[Tuple[Union[pd.Series, np.ndarray], Union[pd.Series, np.ndarray]], pd.DataFrame], to_epsg: int=4326, from_epsg: int=3049):
    """
    Transforms the coordinates in the given DataFrame from one coordinate reference system (CRS) to another.

    Args:
        coordinates_df (pandas.DataFrame): A DataFrame containing the coordinates to be transformed.
        to_crs (int): The EPSG code of the target CRS. Default is 4326, which represents the lonlat projection with underlying WGS84 lon lat.
        from_crs (int): The EPSG code of the source CRS. Default is 3049, which represents the geodetic CRS ETRS89.

    Returns:
        pandas.DataFrame: A DataFrame containing the transformed coordinates.

    """
    crs_to = pyproj.crs.CRS.from_epsg(to_epsg)
    crs_from = pyproj.crs.CRS.from_epsg(from_epsg)
    
    transformer_latlon_to_3049 = pyproj.Transformer.from_crs(crs_from, crs_to, always_xy=True)
    if isinstance(coordinates, pd.DataFrame):
        x, y = transformer_latlon_to_3049.transform(coordinates.iloc[:,0].values, coordinates.iloc[:,1].values) 
        position_transf = pd.DataFrame()
        position_transf['x'] = x
        position_transf['y'] = y
        return position_transf
    elif isinstance(coordinates, tuple):
        x, y = transformer_latlon_to_3049.transform(coordinates[0], coordinates[1])
        return x, y
        
# --------------------------------------------------------
# Functions for prediction and plotting
# --------------------------------------------------------


def predictions_on_data_with_groundtruth(trainer: pl.Trainer, model: PL_PINN, config: Dict[str, Any], path: str=None, gt_dataset: PL_GlacierDataset=None):
    """Predict for the data at path, where ground truth data is known. 
    The config file determines how the dataset is loaded as PL_ItsLiveDataset. 
    Config should also provide scaling mean and variance, that was used for the scaling of the training data.
    Alternatively the dataset can be provided directly.
    The coordinates are transformed to lonlat coordinates.

    Args:
        trainer (Trainer): The PyTorch Lightning trainer object.
        model (nn.Module): The PyTorch Lightning model object.
        config (dict): The configuration dictionary.
        path (str): The path to the data.
        gt_dataset (PL_GlacierDataset, optional): The ground truth dataset. Defaults to None.

    Returns:
        tuple: A tuple containing the transformed coordinates, thicknesses, velocities, and residuals.
    """

    if gt_dataset is None:
        config["ds"]["data_dir_labeled"]=path
        config["ds"]["unlabeled_sample_size"]=0.
        config["ds"]["labeled_sample_size"]=1.
        gt_dataset = PL_GlacierDataset(config)
    predictions = trainer.predict(model, datamodule = gt_dataset)

    thicknesses = np.concatenate([thick[0] for thick in predictions]).ravel()
    velocities = np.concatenate([pred[1] for pred in predictions])
    residuals = gt_dataset.dataset.target[config["ds"]["target"][0]]-thicknesses

    coordinates = gt_dataset.dataset.inputs[config["ds"]["input_features"][:2]]
    # transform coordinates to epsg4326 aka longitude and latitude
    if 'epsg_crs' in config['ds']:
        coordinates = transform_projection(coordinates, from_epsg=config['ds']['epsg_crs'])

    return coordinates, thicknesses, velocities, residuals

def predictions_on_data_without_groundtruth(trainer: pl.Trainer, model: PL_PINN, config: Dict[str, Any], path: str=None, grid_dataset: PL_GlacierDataset=None):
    """
    Generate predictions on unlabeled data without ground truth. The config file determines how the dataset is loaded as PL_GlacierDataset and 
    the scaling of the inputs and targets. Alternatively the dataset can be provided directly.
    The coordinates are transformed to lonlat coordinates.

    Args:
        trainer (pl.Trainer): The PyTorch Lightning trainer object.
        model (PL_PINN): The PyTorch Lightning model object.
        config (dict): Configuration parameters for the dataset.
        path (str): The path to the unlabeled data.
        grid_dataset (PL_GlacierDataset, optional): The PyTorch Lightning dataset object. Defaults to None.

    Returns:
        tuple: A tuple containing the transformed coordinates, thicknesses, and velocities.
    """
    
    if grid_dataset is None:
        config["ds"]["data_dir_unlabeled"]=path 
        #config["ds"]["data_dir_labeled"]=path # does not matter, just pick any, as the sample size is set to 0 here
        config["ds"]["unlabeled_sample_size"]=1.
        config["ds"]["labeled_sample_size"]=0.
        
        grid_dataset = PL_GlacierDataset(config)
    
    predictions = trainer.predict(model, datamodule = grid_dataset)

    thicknesses = np.concatenate([thick[0] for thick in predictions]).ravel()
    velocities = np.concatenate([pred[1] for pred in predictions])

    coordinates = grid_dataset.dataset.inputs[config["ds"]["input_features"][:2]]
    # transform coordinates to epsg4326 aka longitude and latitude
    if 'epsg_crs' in config['ds']:
        coordinates = transform_projection(coordinates, from_epsg=config['ds']['epsg_crs'])
    return coordinates, thicknesses, velocities


class SpitsbergenPlotFormatter():
    """A class that helps formatting all the plots of Spitsbergen, Edgeoya, and Barentsoya. """
    def __init__(self, fontsize_dict: Dict[str, int]=None):
        self.fontsize_small = fontsize_dict['fontsize_small']
        self.fontsize_big = fontsize_dict['fontsize_big']
        self.fontsize_medium = fontsize_dict['fontsize_medium']
        self.coastline = None


    def format_spitsbergen_plot(self, ax, title: str=None, xlabel: str='Longitude', ylabel: str='Latitude'):
        ax.set_xlabel(xlabel, fontsize=self.fontsize_small)
        if ylabel is not None:
            ax.set_ylabel(ylabel, fontsize=self.fontsize_small)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d° E'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%d° N'))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.set_title(title, fontsize=self.fontsize_big)
        ax.tick_params(axis='both', which='both', labelsize=self.fontsize_small)

    def format_spitsbergen_cbar(self, ax, plot, label, extend='neither'):
        cbar = plt.colorbar(plot, ax=ax, extend=extend,fraction=0.07, pad=0.04)
        cbar.set_label(label=label,size=self.fontsize_small)
        cbar.ax.tick_params(labelsize=self.fontsize_small)
    
    def add_coastline(self, ax):
        if self.coastline is None:
            self._get_coastline(ax, self.coastline)
        self.coastline.plot(ax=ax, edgecolor='black', facecolor='none', lw=1, label='Coastline')

    def _get_coastline(self, ax, coastline: pd.DataFrame):
        # Read the coastline data
        coastlines = gpd.read_file('data/svalbardcoastline/Coast2015.shp')
        # Transform the coordinate system to EPSG:4326
        coastlines = coastlines.to_crs(epsg=4326)

        # Define the bounding box
        minx, miny, maxx, maxy = 11, 76.5, 24, 80  # Adjust these values as needed
        bounding_box = box(minx, miny, maxx, maxy)

        # Clip the coastline data to the bounding box
        coastlines_clipped = gpd.clip(coastlines, bounding_box)
        # Create a line that connects the two points [20°E, 80°N] and [24°E, 78°N]
        line = LineString([(20, 80), (24, 78)])
        # Exclude all the coastlines that are east of this line
        coastlines_clipped = coastlines_clipped[coastlines_clipped.geometry.apply(lambda x: not x.intersects(line))]

        self.coastline = coastlines_clipped


def plot_thickness_pred(coordinate_grid: pd.DataFrame, thicknesses_grid: np.ndarray, eval: list, config: Dict[str, Any], title: str='thickness_prediction.png', vmax: float=None):
    """
    Plots the predicted ice thickness and saves it as png file with title.

    Args:
        position_grid (pandas.DataFrame): DataFrame containing the position grid.
        thicknesses_grid (numpy.ndarray): Array containing the predicted ice thicknesses.
        eval (list): List of evaluation metrics.
        config (dict): Configuration settings.
        title (str, optional): Title of the plot. Defaults to 'thickness_prediction.png'.
        vmax (float, optional): Maximum value for the colorbar. Defaults to None.

    Returns:
        None
    """
    ccmap = plt.cm.viridis
    ccmap.set_under('magenta')
    f, ax1 = plt.subplots(1,1, figsize=(10,10))
    p1=ax1.scatter(coordinate_grid.iloc[:,0], coordinate_grid.iloc[:,1], c=thicknesses_grid, marker='.',  cmap=ccmap,vmax=vmax, vmin=0,s=1)
    plt.colorbar(p1, ax=ax1,label='Ice thickness [m]', extend='min')
    ax1.set_title('Predicted ice thickness')
    
    plt.figtext(0.1, 0.05, s=('PINN estimate: Val RMSE', eval[0]['val Thickness RMSE'],  ' Val MAPE ' , eval[0]['val MAPE']))
    plt.savefig(os.path.join(config["experiment"]["save_dir"], title))
    plt.close()


def plot_difference_to_measurement(residuals_coords: pd.DataFrame, grid_coords: pd.DataFrame, residuals: pd.Series, thicknesses_grid: np.ndarray,  config: Dict[str, Any], eval=None, title: str='thickness_predictionvstruth.png', vmax: float=700, s: int=None):
    """
    Plots the predicted ice thickness and the difference to measurements.

    Args:
        residuals_coords (pd.DataFrame): DataFrame containing the coordinates of the residuals.
        grid_coords (pd.DataFrame): DataFrame containing the coordinates of the grid.
        residuals (pd.Series): Series containing the difference between the true and predicted values.
        thicknesses_grid (np.ndarray): Array containing the predicted ice thickness values.
        config (dict): Dictionary containing the configuration settings.
        eval (optional): Evaluation metrics for the predictions. Defaults to None.
        title (str, optional): Title of the plot. Defaults to 'thickness_predictionvstruth.png'.
        vmax (float, optional): Maximum value for the colorbar. Defaults to 700.
        s (int, optional): Size of the markers. Defaults to None.
    """
    ccmap = plt.cm.viridis
    ccmap.set_under('magenta')

    f, ax1 = plt.subplots(1,1, figsize=(10,10))
    p1=ax1.scatter(grid_coords.iloc[:,0], grid_coords.iloc[:,1], c=thicknesses_grid, marker='.',  cmap=ccmap,vmax=vmax, vmin=0,s=s)
    plt.colorbar(p1, ax=ax1,label='Ice thickness [m]', extend='min')
    ax1.set_title('Predicted ice thickness')
    p2 = ax1.scatter(residuals_coords.iloc[:,0], residuals_coords.iloc[:,1], c=residuals, marker='.',cmap='RdBu',vmax=50, vmin=-50 , s=s)
    plt.colorbar(p2, ax=ax1, label='Difference to measurements (True-Pred)')
    
    if eval is not None:
        plt.figtext(0.1, 0.05, s=('PINN estimate: Val RMSE', eval[0]['val Thickness RMSE'],  ' Val MAPE ' , eval[0]['val MAPE']))
    plt.savefig(os.path.join(config["experiment"]["save_dir"], title))
    plt.close()



def plot_velocity_predictions(coordinate_grid: pd.DataFrame, velocities: np.ndarray, config: Dict[str, Any]):
    """
    Plots the velocity predictions on a coordinate grid. Maximum and minimum values are given by the mean and the 3 sigma standard deviation of the velocities.

    Args:
        coordinate_grid (pd.DataFrame): The coordinate grid data.
        velocities (np.ndarray): The velocity data.
        config (dict): The configuration settings.

    Returns:
        None
    """
    max_value_x = abs(np.mean(velocities[:,0]))+3*np.std(velocities[:,0])
    max_value_y = abs(np.mean(velocities[:,1])) +3*np.std(velocities[:,1])
    
    f, (ax1,ax2) = plt.subplots(1,2, figsize=(20,10))
    p1=ax1.scatter(coordinate_grid.iloc[:,0], coordinate_grid.iloc[:,1], c=velocities[:,0], marker='.',  cmap='coolwarm',s=1,vmax=max_value_x, vmin=-max_value_x,)
    plt.colorbar(p1, ax=ax1,label='Velocity [m/a]')
    ax1.set_title('Depth-averaged Vel_x [m/a]')
    p2 = ax2.scatter(coordinate_grid.iloc[:,0], coordinate_grid.iloc[:,1], c=velocities[:,1], marker='.',cmap='coolwarm', s=1,vmax=max_value_y, vmin=-max_value_y)
    ax2.set_title('Depth-averaged Vel_y [m/a]')
    plt.colorbar(p2, ax=ax2, label='Velocity [m/a]')
    plt.savefig(os.path.join(config["experiment"]["save_dir"],'velocity_prediction.png'))
    plt.close()



def plot_glacier(rgi_id: str, trainer: pl.Trainer, model: PL_PINN, config: Dict[str, Any], title: str='thickness_predVStruth_leftoutglacier.png'):
    """
    Plots the glacier thickness predictions and the residual to the measurements for the glacier with rgi_id.
    Calculates evaluation metrics for the given glacier. This function is used to evaluate the model on a left out glacier in the LOGO_CV.py script.

    Args:
        rgi_id (str): The ID of the glacier.
        trainer (pl.Trainer): The PyTorch Lightning trainer object.
        model (PL_PINN): The PyTorch Lightning model object.
        config (dict): The configuration dictionary.
        title (str, optional): The title of the plot. Defaults to 'thickness_predVStruth_leftoutglacier.png'.

    Returns:
        dict: The evaluation metrics for the left out glacier.
    """
    if "num_points" in config["ds"]:
        del config["ds"]["num_points"]
    config["ds"]["glacier_ids"] = []
    config["ds"]["only_glacier"] = rgi_id

    # Grid dataset
    grid_config = config.copy()
    grid_config["ds"]["unlabeled_sample_size"] = 1.0
    grid_config["ds"]["labeled_sample_size"] = 0.0
    print(grid_config["ds"]["unlabeled_sample_size"], grid_config["ds"]["labeled_sample_size"])
    grid_dataset = PL_GlacierDataset(grid_config)

    grid_coords, grid_thick, _ = predictions_on_data_without_groundtruth(trainer, model, grid_config, path=grid_config["ds"]["data_dir_unlabeled"], grid_dataset=grid_dataset)

    # Groundtruth dataset
    gt_config = config.copy()
    gt_config["ds"]["unlabeled_sample_size"] = 0.0
    gt_config["ds"]["labeled_sample_size"] = 1.0
    print(gt_config["ds"]["unlabeled_sample_size"], gt_config["ds"]["labeled_sample_size"])

    gt_dataset = PL_GlacierDataset(gt_config)
    eval1 = trainer.validate(model, datamodule=gt_dataset)
    print('Evaluation on left out glacier: ', eval1)

    gt_coords, _, _, residuals = predictions_on_data_with_groundtruth(trainer, model, gt_config, path=gt_config["ds"]["data_dir_labeled"], gt_dataset=gt_dataset)

    plot_difference_to_measurement(gt_coords, grid_coords, residuals, grid_thick, config, eval=eval1, title=title, vmax=None)
    return eval1



def comparison_to_millan_full_grid(config: Dict[str, Any], path: str, trainer: pl.Trainer, model: PL_PINN):
    """
    Compare the predicted ice thickness and velocities to the target values from the Millan dataset.

    Args:
        config (dict): Configuration parameters for the experiment.
        path (str): Path to the data directory.
        trainer: Trainer object for training and evaluation.
        model: Model object for prediction.

    Returns:
        None
    """
    config["ds"]["target"][0] = 'millan_ice_thickness'
    config["ds"]["data_dir_labeled"] = path  
    config["ds"]["unlabeled_sample_size"] = 0.
    config["ds"]["labeled_sample_size"] = 1.
    test_dataset = PL_GlacierDataset(config)
    evaluation = trainer.validate(model, datamodule=test_dataset)
    predictions = trainer.predict(model, datamodule=test_dataset)

    coords = test_dataset.dataset.inputs[config["ds"]["input_features"][:2]]
    thicknesses = np.concatenate([thick[0] for thick in predictions]).ravel()
    residual = test_dataset.dataset.target['millan_ice_thickness'] - thicknesses

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
    p1 = ax1.scatter(coords.iloc[:, 0], coords.iloc[:, 1], c=thicknesses, marker='.', cmap='viridis', vmax=700, vmin=0, s=1)
    plt.colorbar(p1, ax=ax1)
    ax1.set_title('Predicted ice thickness [m]')
    p2 = ax2.scatter(coords.iloc[:, 0], coords.iloc[:, 1], c=test_dataset.dataset.target['millan_ice_thickness'], marker='.', vmax=700, s=1)
    plt.colorbar(p2, ax=ax2)
    ax2.set_title('Target ice thickness [m]')
    p3 = ax3.scatter(coords.iloc[:, 0], coords.iloc[:, 1], c=residual, cmap='RdBu', marker='.', vmin=-50, vmax=50, s=1)
    plt.colorbar(p3, ax=ax3)
    ax3.set_title('Residual: Millan-Predicted')

    plt.figtext(0.6, 0.05, s=evaluation)
    plt.savefig(os.path.join(config["experiment"]["save_dir"], "comparison_thickness_prediction_to_millan.png"))
    plt.close()

    velocities = np.concatenate([pred[1] for pred in predictions])
    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(30, 10))
    p1 = ax1.scatter(coords.iloc[:, 0], coords.iloc[:, 1], c=velocities[:, 0], vmin=-50, vmax=50, s=0.5)
    plt.colorbar(p1, ax=ax1)
    p2 = ax2.scatter(coords.iloc[:, 0], coords.iloc[:, 1], c=test_dataset.dataset.inputs['millan_vx_smoothed'], vmin=-50, vmax=50, s=0.5)
    plt.colorbar(p2, ax=ax2)
    p3 = ax3.scatter(coords.iloc[:, 0], coords.iloc[:, 1], c=velocities[:, 1], vmin=-100, vmax=100, s=0.5)
    plt.colorbar(p3, ax=ax3)
    p4 = ax4.scatter(coords.iloc[:, 0], coords.iloc[:, 1], c=test_dataset.dataset.inputs['millan_vy_smoothed'], vmin=-100, vmax=100, s=0.5)
    plt.colorbar(p4, ax=ax4)

    ax1.title.set_text('depth_avg x')
    ax2.title.set_text('surface vel x')
    ax3.title.set_text('depth_avg y')
    ax4.title.set_text('surface_vel y')
    plt.savefig(os.path.join(config["experiment"]["save_dir"], 'velocity_prediction.png'))
    plt.close()
