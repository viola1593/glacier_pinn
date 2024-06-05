"""Make predicitons of the LOGO test glaciers with the respective model trained without the glacier and save the predictions in a csv file.
Also we plot the predictions and the residual to the measured ice thicknesses. """

import argparse
import numpy as np
import os
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from matplotlib import ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from dataset_new import PL_GlacierDataset
from model import PL_PINN
from utils import read_config

SUBLABEL= ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)']
FONTSIZE_SMALL=14
FONTSIZE_BIG=20
FONTSIZE_MEDIUM=16

GLACIER_IDS = {
        'RGI60-07.00240': {'num_plot': 1, 'xticks': [15.7], 'yticks': [77.0]},
        'RGI60-07.00344': {'num_plot': 2, 'xticks': [16.3,], 'yticks': [78.0]},
        'RGI60-07.00496': {'num_plot': 3, 'xticks': [10.5, 11.5], 'yticks': [78.5, 79.5]},
        "RGI60-07.00497": {'num_plot': 4, 'xticks': [10.5, 11.5], 'yticks': [78.5, 79.5]},
        'RGI60-07.01100': {'num_plot': 5, 'xticks': [10.5, 11.5], 'yticks': [78.5, 79.5]},
        'RGI60-07.01481': {'num_plot': 6, 'xticks': [10.5, 11.5], 'yticks': [78.5, 79.5]},
        'RGI60-07.01482': {'num_plot': 7, 'xticks': [10.5, 11.5], 'yticks': [78.5, 79.5]}
        }


       
def plot_velocity_predictions(position_grid, velocities, save_dir, rgi):
    
    max_value_x = abs(np.mean(velocities[:,0]))+3*np.std(velocities[:,0])
    max_value_y = abs(np.mean(velocities[:,1])) +3*np.std(velocities[:,1])
    #print(max_value_x, max_value_y)
    f, (ax1,ax2) = plt.subplots(1,2, figsize=(12,7), sharey=True, layout='constrained')
    scatter1=ax1.scatter(position_grid.iloc[:,0], position_grid.iloc[:,1], c=velocities[:,0],  cmap='coolwarm',vmax=max_value_x, vmin=-max_value_x,marker='.', s=10)
    ax1.set_xlabel('Longitude', fontsize=FONTSIZE_SMALL)
    ax1.set_ylabel('Latitude', fontsize=FONTSIZE_SMALL)
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%.2f° E'))
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f° N'))
    ax1.set_title('Depth-averaged velocity $ \overline{v}_x$ ', fontsize=FONTSIZE_BIG)
    ax1.tick_params(axis='both', which='both', labelsize=FONTSIZE_SMALL)

    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('Velocity [m a$^{-1}$]', fontsize=FONTSIZE_SMALL, labelpad=0)
    cbar1.ax.tick_params(labelsize=FONTSIZE_SMALL)
    ax1.text(-0.1, 1.05, '(a)', fontsize=FONTSIZE_BIG, transform=ax1.transAxes)

    scatter2 = ax2.scatter(position_grid.iloc[:,0], position_grid.iloc[:,1], c=velocities[:,1], cmap='coolwarm', vmax=max_value_y, vmin=-max_value_y,marker='.', s=10)
    ax2.set_xlabel('Longitude', fontsize=FONTSIZE_SMALL)
    # ax2.set_ylabel('Latitude')
    ax2.xaxis.set_major_formatter(FormatStrFormatter('%.2f° E'))
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f° N'))
    ax2.tick_params(axis='both', which='both', labelsize=FONTSIZE_SMALL)

    ax2.set_title('Depth-averaged velocity $ \overline{v}_y$', fontsize=FONTSIZE_BIG)
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('Velocity [m a$^{-1}$]', fontsize=FONTSIZE_SMALL, labelpad=0)
    cbar2.ax.tick_params(labelsize=FONTSIZE_SMALL)
    ax2.text(-0.1, 1.05, '(b)', fontsize=FONTSIZE_BIG, transform=ax2.transAxes)
    f.savefig(os.path.join(save_dir,'LOGOvelocity_prediction'+rgi+'.png'))



def plot_residuals(grid_coords, measurement_coords, grid_thickness, residuals, fig, num_plot, rgi):
    if num_plot <5:
        ax = fig.add_subplot(2,4,num_plot)
       
    else:
        ax = fig.add_subplot(2,3,num_plot-1)

    ax.set_title(rgi, fontsize=FONTSIZE_BIG)
    
    # transform lat lon to meters
    x_grid, y_grid = grid_coords.iloc[:,0], grid_coords.iloc[:,1]
    x_measurement, y_measurement = measurement_coords.iloc[:,0], measurement_coords.iloc[:,1]
    # plot ice thickness and residuals
    p1 = ax.scatter(x_grid, y_grid, c=grid_thickness, cmap='viridis',marker='.', s=20, alpha=0.1)
    p2 = ax.scatter(x_measurement, y_measurement, c=residuals, cmap='RdBu', vmin=-50, vmax=50,marker='.', s=20)

    
    # -------------------------------------
    # colorbars
    # cbar1 is for the ice thickness
    cbar1 = plt.colorbar(p1, ax=ax)
    cbar1.ax.tick_params(labelsize=FONTSIZE_MEDIUM)
     # Set the number of ticks you want
    locator = ticker.MaxNLocator(nbins=3) 
    cbar1.locator = locator
    cbar1.update_ticks()
    # Set the label of the colorbar only in the last plot
    if num_plot == 7:
        cbar1.ax.set_ylabel('Thickness $H_\mathrm{PINN}$ [m]', fontsize=FONTSIZE_BIG)

    # cbar2 is for the residuals, only show it in the first plot as an inset
    if num_plot ==1:
        cbaxes = inset_axes(ax, width="30%", height="3%", loc='upper right', borderpad=2) 
        cbar2 = plt.colorbar(p2, cax=cbaxes,  orientation='horizontal')
        cbar2.ax.set_xlabel('$H_\mathrm{GPR}-H_\mathrm{PINN}$', fontsize=FONTSIZE_MEDIUM)
        cbar2.ax.tick_params(labelsize=FONTSIZE_MEDIUM)
        locator = ticker.MaxNLocator(nbins=1)
        cbar2.locator = locator
        cbar2.update_ticks()

    # -------------------------------------   
    # tick params
    # set number of x and y ticks, defines where and how many grid lines there are
    ax.xaxis.set_major_locator(MaxNLocator(nbins=1))  # Set number of x-ticks
    ax.yaxis.set_major_locator(MaxNLocator(nbins=1))  # Set number of y-ticks
    

    # if we show the tick labels we want them inside the plot, x axis should be rotated to fit along the gridlines
    ax.tick_params(axis='x', which='both', direction='in', length=0, labelsize=FONTSIZE_SMALL, pad=-100)
    ax.tick_params(axis='y', which='both', direction='in', length=0, labelsize=FONTSIZE_SMALL, pad=-50)

    # don't show the tick labels
    ax.tick_params(labelbottom=False, labeltop=False, labelleft=False, labelright=False)

    # -------------------------------------
    #Add scalebar
    # Define the length of the scale bar and the label
    scalebar_length = 1000  # 1000 meters = 1 km
    scalebar_label = "1 km"
    # Define the vertical extent of the scale bar and make it have the same extent in every plot although the extents of the plots vary
    extent = np.max(y_grid)-np.min(y_grid)
    scalebar_vertical = extent/60
    # Create a font properties object
    fontprops = fm.FontProperties(size=FONTSIZE_MEDIUM)
    # Create the scale bar
    scalebar = AnchoredSizeBar(ax.transData, scalebar_length, scalebar_label, 'lower left', pad=0.1, 
                               color='black', frameon=False, size_vertical=scalebar_vertical, fontproperties=fontprops,
                               )
    # Add the scale bar to the axes
    ax.add_artist(scalebar)
    # -------------------------------------
    # Add the labels to each plot
    if SUBLABEL is not None:
        if num_plot<5:
            ax.text(-0.1, 1.1, SUBLABEL[num_plot-1], fontsize=FONTSIZE_BIG, transform=ax.transAxes)
        else:
            ax.text(-0.1, 1.05, SUBLABEL[num_plot-1], fontsize=FONTSIZE_BIG, transform=ax.transAxes)

    return fig

def plot_all_residuals_from_df(grid_df, residuals_df, fig, savedir ):
    grouped_grid = grid_df.groupby('RGI_ID')
    grouped_residuals = residuals_df.groupby('RGI_ID')

    for rgi, grid in grouped_grid:
        num_plot = GLACIER_IDS[rgi]['num_plot']
        residuals = grouped_residuals.get_group(rgi)
        fig = plot_residuals(grid[['POINT_LON', 'POINT_LAT']], residuals[['POINT_LON', 'POINT_LAT']], grid['pinn_thickness'], residuals['residual'], fig, num_plot=num_plot, rgi=rgi)
        fig.subplots_adjust(hspace=0.2)
    return fig

def plot_single_figures_from_df(grid_df, residuals_df, rgi, savedir):

    fig, ax = plt.subplots(layout='constrained', figsize=(7,8))
    ax.set_title(rgi, fontsize=FONTSIZE_BIG)
    p1 = ax.scatter(grid_df['POINT_LON'], grid_df['POINT_LAT'], c=grid_df['pinn_thickness'], cmap='viridis',marker='.', s=20)
    p2 = ax.scatter(residuals_df['POINT_LON'], residuals_df['POINT_LAT'], c=residuals_df['residual'], cmap='RdBu', vmin=-50, vmax=50,marker='.', s=20)
    plt.xticks(rotation=90)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    if rgi == 'RGI60-07.00240':

        cbaxes = inset_axes(ax, width="30%", height="3%", loc='upper right', borderpad=3)
        cbaxes.patch.set_facecolor('white')
        cbar2 = plt.colorbar(p2, cax=cbaxes,  orientation='horizontal')
        cbar2.ax.set_xlabel('$H_\mathrm{GPR}-H_\mathrm{PINN}$ [m]', fontsize=FONTSIZE_SMALL,bbox=dict(facecolor='white', edgecolor='none', pad=0))
        for label in cbar2.ax.xaxis.get_ticklabels():
            label.set_bbox(dict(facecolor='white', edgecolor='none', pad=0, boxstyle='square,pad=0'))

    cbar1 = plt.colorbar(p1, ax=ax)
    cbar1.ax.set_ylabel('Thickness $H_\mathrm{PINN}$ [m]', fontsize=FONTSIZE_SMALL)
    cbar1.ax.tick_params(labelsize=FONTSIZE_SMALL)
    ax.set_xlabel('Longitude', fontsize=FONTSIZE_SMALL)
    ax.set_ylabel('Latitude', fontsize=FONTSIZE_SMALL, )
        # set tick params
    ax.xaxis.set_major_locator(MaxNLocator(nbins=1))  # Set number of x-ticks
    ax.yaxis.set_major_locator(MaxNLocator(nbins=1))  # Set number of y-ticks
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f° E'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f° N'))
    y_labels =  ax.get_yticklabels()
    print(y_labels)
    if rgi == 'RGI60-07.00344' or rgi == 'RGI60-07.00496':
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f° N'))

    ax.tick_params(axis='x', which='both', direction='in', length=0, labelsize=FONTSIZE_SMALL, )
    ax.tick_params(axis='y', which='both', direction='in', length=0, labelsize=FONTSIZE_SMALL, )
    
    ax.text(0, 1.1, SUBLABEL[GLACIER_IDS[rgi]['num_plot']-1], fontsize=FONTSIZE_BIG, transform=ax.transAxes)
    fig.savefig(os.path.join(savedir,'LOGOresiduals'+rgi+'.png'))
    return fig
    

def evaluate_model_on_singleglacier(gpu: int, datadir:str, gridpath: str, measurementpath:str, figure, num_plot: int, rgi_id: str=None, saving_grid_df: pd.DataFrame=None, saving_residuals_df: pd.DataFrame=None):
    pl.seed_everything(42, workers=True)
    
    try:
        config = read_config(datadir+'/config.yaml')
        loaded_model = PL_PINN.load_from_checkpoint(datadir+"/checkpoints/last.ckpt")
    except:
        print("Could not load model from ", datadir)
        return figure, saving_grid_df, saving_residuals_df
    trainer = Trainer(accelerator='gpu', devices=[gpu])


    ## prepare prediction on grid
    config["ds"]["labeled_sample_size"]=0
    config["ds"]["unlabeled_sample_size"]=1.0
    config["ds"]["data_dir_unlabeled"]=[gridpath]

    
    if 'num_points' in config["ds"]:
        del config["ds"]["num_points"]
    config["ds"]["glacier_ids"] = []
    if rgi_id is None:
        rgi_id = datadir.split("_")[-3]
        print(rgi_id)
    
    ## Prediction on grid
    gridded_dataset = PL_GlacierDataset(config)
    gridded_dataset.unlabeled_data = gridded_dataset.unlabeled_data[(gridded_dataset.unlabeled_data.RGI_ID==rgi_id)]


    predictions_on_grid = trainer.predict(loaded_model, datamodule = gridded_dataset)
    pred_thicknesses_on_grid = np.concatenate([thick[0] for thick in predictions_on_grid]).ravel()
    velocities_on_grid = np.concatenate([vel[1] for vel in predictions_on_grid])
    grid_coords = gridded_dataset.dataset.inputs[['POINT_LON', 'POINT_LAT']] # are in epsg25833

    # prepare predictions on measurements
    config["ds"]["data_dir_labeled"]=[measurementpath]
    config["ds"]["labeled_sample_size"]=1.0
    config["ds"]["unlabeled_sample_size"]=.0
    config["ds"]["min_years"]=2000
    measurement_dataset = PL_GlacierDataset(config)

    measurement_dataset.labeled_data = measurement_dataset.labeled_data[(measurement_dataset.labeled_data.RGI_ID==rgi_id)]

    ## Prediction on measurements
    predictions_on_measurements = trainer.predict(loaded_model, datamodule = measurement_dataset)
    eval = trainer.validate(loaded_model, datamodule = measurement_dataset)
    print(rgi_id, eval)
    pred_thicknesses_on_measurements = np.concatenate([thick[0] for thick in predictions_on_measurements]).ravel()

    residual_prediction_to_measurement = measurement_dataset.dataset.labeled_dataset.THICKNESS-pred_thicknesses_on_measurements
    measurement_coords = measurement_dataset.dataset.inputs[['POINT_LON', 'POINT_LAT']] # are in epsg25833

    # save predictions in dataframe
    if saving_grid_df is not None:
        df = pd.DataFrame({
                           'POINT_LON':grid_coords.iloc[:,0], 
                           'POINT_LAT':grid_coords.iloc[:,1], 
                           'pinn_thickness':pred_thicknesses_on_grid,
                           'depth-averaged_velocity_x':velocities_on_grid[:,0], 
                           'depth-averaged_velocity_y':velocities_on_grid[:,1],
                           })
        df['RGI_ID'] = rgi_id
        saving_grid_df = pd.concat([saving_grid_df, df])
    

    if saving_residuals_df is not None:
        df = pd.DataFrame({ 
                           'POINT_LON':measurement_coords.iloc[:,0], 
                           'POINT_LAT':measurement_coords.iloc[:,1], 
                           'residual':residual_prediction_to_measurement})
        df['RGI_ID'] = rgi_id
        saving_residuals_df = pd.concat([saving_residuals_df, df])
    
    ## Plotting
    figure = plot_residuals(grid_coords, measurement_coords, pred_thicknesses_on_grid, residual_prediction_to_measurement, figure, num_plot, rgi_id)
    print("lowest predicted thickness: ", np.min(pred_thicknesses_on_grid)) 
    print('Root mean squared residual to measurement: ',np.sqrt(np.mean(residual_prediction_to_measurement**2)))

    plot_velocity_predictions(grid_coords, velocities_on_grid, datadir, rgi_id)
    
    return figure, saving_grid_df, saving_residuals_df

def make_LOGO_figure(gpu: int, directory:str, gridpath: str, measurementpath:str):
    plt.style.use('default')
    figure = plt.figure(figsize=(16,12)) # (16,12) for paper

    if not os.path.exists(os.path.join(directory, 'LOGOglacierspredictions_on_grid.csv')): 
        saving_grid_df = pd.DataFrame()
        saving_residuals_df = pd.DataFrame()
        for subdir in os.listdir(directory):
            subdir_path = os.path.join(directory, subdir)
            if os.path.isdir(subdir_path):
                rgi_id = subdir_path.split("_")[-3]

                figure, saving_grid_df, saving_residuals_df = evaluate_model_on_singleglacier(gpu, subdir_path, gridpath, measurementpath, figure,
                                                        num_plot=GLACIER_IDS[rgi_id]['num_plot'], 
                                                        saving_grid_df=saving_grid_df, 
                                                        saving_residuals_df=saving_residuals_df)

        saving_grid_df.to_csv(os.path.join(directory,'LOGOglacierspredictions_on_grid.csv'))
        saving_residuals_df.to_csv(os.path.join(directory,'LOGOglaciersresiduals.csv'))
        figure.savefig(os.path.join(directory,'LOGOresiduals.png'))
    else:
        saving_grid_df = pd.read_csv(os.path.join(directory, 'LOGOglacierspredictions_on_grid.csv'), low_memory=False)
        saving_residuals_df = pd.read_csv(os.path.join(directory, 'LOGOglaciersresiduals.csv'), low_memory=False)
        figure = plot_all_residuals_from_df(saving_grid_df, saving_residuals_df, figure, savedir=directory)
        figure.savefig(os.path.join(directory,'LOGOresiduals.png'), dpi=300)



def start():
    """Evaluation and prediction."""
    # Command line arguments
    parser = argparse.ArgumentParser(
        prog="predictLOGOglaciers.py", description="Makes predicitons for all the LOGO test glaciers and plots the ice thickness with the residuals to measurements."
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU to use.")
    args = parser.parse_args()
        
    directory = "CVresults\LOGO_results"  
    # get datasets
    measurementpath = "data/spitsbergen_measurements_aggregated_nosurges_dhdt2014smoothed_complete.csv"
    gridpath ="data/spitsbergen_allunmapped_griddeddata_nosurges_dhdt2014smoothed_complete.csv"
    
    make_LOGO_figure(args.gpu, directory, gridpath, measurementpath)
    

if __name__ == "__main__":
    start()