"""Plotting figures for the manuscript: Here we plot the mean of the cross validatiom predictions. 
Also we create scatter plots comparing the mean predicted ice thickness to those of Millan et al (2022), Farinotti et al.'s consensus estimate (2020). """
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import os
from utils import SpitsbergenPlotFormatter


cm = 1/2.54  # centimeters in inches
fontsize_medium=18
fontsize_small = 14
fontsize_big = 20
figsize=(12,8.3)

fontsize_dict = {'fontsize_medium':18,
                'fontsize_small': 14,
                'fontsize_big': 20}

axis_dict = {'consensus_thickness': '$H_\mathrm{consensus}$',
                'millan_ice_thickness': '$H_\mathrm{Millan}$',
                'interpolated_pelt_data': '$H_\mathrm{Pelt}$',
                'pinn_thickness': '$H_\mathrm{PINN}$',
                'true_thickness': '$H_\mathrm{GPR}$',
                }

formatter_spitsbergen = SpitsbergenPlotFormatter(fontsize_dict=fontsize_dict)

def plot_depth_avg_velocities(model_avg, cv_dir):
    """Plots the mean of the depth-averaged velocities of the PINN predictions and saves the figure."""
    vmax_x = model_avg['depth_avg_vel_x'].mean()+model_avg['depth_avg_vel_x'].std()*3
    vmax_y = model_avg['depth_avg_vel_y'].mean()+model_avg['depth_avg_vel_y'].std()*3

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7), sharey=True, layout='constrained')


    scatter1 = ax1.scatter(model_avg['lon'], model_avg['lat'], c=model_avg.depth_avg_vel_x, cmap='coolwarm', s=10, marker='.', vmax=vmax_x, vmin=-vmax_x)
    formatter_spitsbergen.format_spitsbergen_plot(ax1, title='Depth-averaged velocity $ \overline{v}_x$')
    formatter_spitsbergen.format_spitsbergen_cbar(ax1, scatter1, label='Velocity [m a$^{-1}$]')
    ax1.text(-0.1, 1.05, '(a)', fontsize=fontsize_big, transform=ax1.transAxes)
    formatter_spitsbergen.add_coastline(ax1)

    scatter2 = ax2.scatter(model_avg['lon'], model_avg['lat'], c=model_avg.depth_avg_vel_y, cmap='coolwarm', s=10, marker='.', vmax=vmax_y, vmin=-vmax_y)
    formatter_spitsbergen.format_spitsbergen_plot(ax2, title='Depth-averaged velocity $ \overline{v}_y$')
    formatter_spitsbergen.format_spitsbergen_cbar(ax2, scatter2, label='Velocity [m a$^{-1}$]')
    ax2.text(-0.1, 1.05, '(b)', fontsize=fontsize_big, transform=ax2.transAxes)
    ax2.set_ylabel('')
    formatter_spitsbergen.add_coastline(ax2)

    fig.savefig(cv_dir+'/mean_depth-avg_vel.png', dpi=300)


def scatter_preds_with_linearfit(model_avg, x_data: str, title: str, save_as: str=None, y_data: str = 'pinn_thickness',cv_dir: str=None):
    """Creates scatterplots with x_data at the x axis and y_data on the y axis. 
    Fits a linear curve to the values."""
    pinn_predictions_nonans = model_avg.dropna(subset=[x_data, y_data])

    # Extract the x and y values for the linear fit from the DataFrame
    x = pinn_predictions_nonans[x_data].values
    y = pinn_predictions_nonans[y_data].values

    # Fit a linear function to the data
    coefficients = np.polyfit(x, y, 1)
    slope = coefficients[0]
    intercept = coefficients[1]

    # Print the slope and intercept of the linear function
    print("Slope:", slope)
    print("Intercept:", intercept)
    #label='Slope: {:.2f} \nIntercept: {:.2f}'.format(slope, intercept)

    # calculate mean absolute difference 
    mad = np.mean(np.abs(y - x))
    f, ax = plt.subplots(figsize=(4,4 ))
    ccmap = plt.cm.viridis
    ccmap.set_under('white')
    model_avg.plot.hexbin(x=x_data, y=y_data, gridsize=500,cmap=ccmap , ax=ax, vmin=1, colorbar =False)
    plt.xlim(0, 800)
    plt.ylim(0, 800)
    ax.set_facecolor('white')
    ax.set_title(title)
    ax.text(0.6, 0.05, 'MAD:  {:.0f} m'.format(mad), fontsize=fontsize_small, transform=ax.transAxes)
    plt.plot([0, 800], [0, 800], '--', color='black', alpha=0.5)
    plt.xticks([0, 700], visible=True)
    plt.yticks([0, 700], visible=True)
    plt.ylabel(axis_dict[y_data]+' [m]', fontsize=fontsize_medium, labelpad=-12)
    plt.xlabel(axis_dict[x_data]+' [m]', fontsize=fontsize_medium, labelpad=-10) 
    ax.xaxis.get_label().set_visible(True)
    ax.yaxis.get_label().set_visible(True)

    plt.plot(np.linspace(0, 800, 100), np.polyval(coefficients, np.linspace(0, 800, 100)), color='r', alpha=0.5, label='Slope: {:.2f} \nIntercept: {:.2f}'.format(slope, intercept))
    plt.legend()
    
    plt.savefig(cv_dir+save_as)
    


def scatter_preds_with_linearfit_to_fig(model_avg, x_data: str, title: str, fig, num_plot, y_data: str = 'pinn_thickness'):
    """Adds a scatterplot at position num_plot to the given figure with x_data at the x axis and y_data on the y axis. 
    Fits a linear curve to the values."""
    pinn_predictions_nonans = model_avg.dropna(subset=[x_data, y_data])

    # Extract the x and y values for the linear fit from the DataFrame
    x = pinn_predictions_nonans[x_data].values
    y = pinn_predictions_nonans[y_data].values

    # Fit a linear function to the data
    coefficients = np.polyfit(x, y, 1)
    slope = coefficients[0]
    intercept = coefficients[1]

    # Print the slope and intercept of the linear function
    print("Slope:", slope)
    print("Intercept:", intercept)
    

    # calculate mean absolute difference 
    mad = np.mean(np.abs(y - x))

    ccmap = plt.cm.viridis
    ccmap.set_under('white')
    
    ax = fig.add_subplot(2,3,num_plot, aspect='equal')

    model_avg.plot.hexbin(x=x_data, y=y_data, gridsize=500,cmap=ccmap , ax=ax, vmin=1, colorbar =False)
    ax.set_facecolor('white')
    ax.plot([0, 800], [0, 800], '--', color='black', alpha=0.5)
    ax.plot(np.linspace(0, 800, 100), np.polyval(coefficients, np.linspace(0, 800, 100)), color='r', alpha=0.5, label='Slope: {:.2f} \nIntercept: {:.2f}'.format(slope, intercept))

    sublabel= ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    ax.text(-0.1, 1.05, sublabel[num_plot-1], fontsize=fontsize_big, transform=ax.transAxes)

    ax.set_title(axis_dict[x_data]+' vs. '+axis_dict[y_data], fontsize=fontsize_big)
    ax.text(0.6, 0.05, 'MAD:  {:.0f} m'.format(mad), fontsize=fontsize_small, transform=ax.transAxes)

    # Increase the size of the ticks and labels
    ax.tick_params(axis='both', which='both', labelsize=fontsize_small)
    plt.xlim(0, 800)
    plt.ylim(0, 800)
    plt.xticks([0, 700], visible=True)
    plt.yticks([0, 700], visible=True)

    plt.ylabel(axis_dict[y_data]+' [m]', fontsize=fontsize_medium, labelpad=-12)
    plt.xlabel(axis_dict[x_data]+' [m]', fontsize=fontsize_medium, labelpad=-10) 

    ax.xaxis.get_label().set_visible(True)
    ax.yaxis.get_label().set_visible(True)
    
    ax.legend(fontsize=fontsize_small) 
    
    return fig

def scatter_subplots(model_avg, columns, compare_to, save_as_prefix):
    """Creates a figure that can be filled with the scatter subplots."""

    fig = plt.figure(figsize=(12,9), layout='constrained')
    #fig.suptitle(title_prefix, fontsize=16)
    
    
    for  i, (column, to) in enumerate(zip(columns, compare_to)):

        title_prefix = column + ' vs '+ to
        fig = scatter_preds_with_linearfit_to_fig(model_avg, column, title_prefix, fig, i+1, y_data=to)
        
    fig.savefig(save_as_prefix + '.png', dpi=300) # scatter plots are too big to be saved as eps


def interpolate_other_to_PINNthickness(points, values, to_grid_x, to_grid_y):
    """
    Interpolates the given points and values to the grid defined by to_grid_x and to_grid_y, using linear interpolation.
    Be careful to use the correct coordinate system for the points and the grid.
    points: np.array of shape (n, 2), grid of the values to be interpolated
    values: np.array of shape (n,), values to be interpolated

    to_grid_x: np.array of shape (m, 1), x-coordinates of the grid to interpolate to
    to_grid_y: np.array of shape (m, 1), y-coordinates of the grid to interpolate to
    """
    interpolated_data = griddata(points, values, (to_grid_x, to_grid_y), method='linear')
    return interpolated_data



def generate_figures(grid_or_measurements='grid', cv_dir=None):
    if grid_or_measurements == 'grid':
        pinn_predictions = pd.read_csv(cv_dir +'/predictions_on_grid.csv', low_memory=False)
    if grid_or_measurements == 'measurement':
        pinn_predictions = pd.read_csv(cv_dir +'/predictions_on_measurements.csv', low_memory=False)

    results_group = pinn_predictions.groupby(['x', 'y']) # group all predictions by each point, so that we get the prediction of all 7 models at this point
    
    # create the average model predictions for each grid point
    if 'model_avg_grid.csv' not in os.listdir(cv_dir):
        # calculate the means for every grid point
        consensus_thick = results_group.consensus_thickness.mean()
        millan_thick = results_group.millan_ice_thickness.mean()
        pinn_thick = results_group.pinn_thickness.mean()
        depth_vel_x =results_group.depth_avg_vel_x.mean()
        depth_vel_y =results_group.depth_avg_vel_y.mean()

        if grid_or_measurements == 'measurements':
            # add the ground truth data
            true_thick = results_group.true_thickness.mean()
            #concatenate the doubly indexed dataframes along their indices
            model_avg = pd.concat([consensus_thick, 
                                true_thick,
                            millan_thick, pinn_thick, depth_vel_x,depth_vel_y], axis=1).reset_index()
        elif grid_or_measurements == 'grid':
            #concatenate the doubly indexed dataframes along their indices
            model_avg = pd.concat([consensus_thick, 
                            millan_thick, pinn_thick, depth_vel_x,depth_vel_y], axis=1).reset_index()
            
        # rename the coordinate columns to make clear that this is in the lonlat coordinate system
        model_avg['lon'] = model_avg.x
        model_avg['lat'] =  model_avg.y

        # # get pelt dataset for comparison and interpolate it to the grid points
        pelt_thickness = pd.read_csv('data/PELT_thickness_svalbard_update.csv', low_memory=False)

        # # Interpolate pelt data to the grid points of the PINN predicitons
        print('interpolating pelt data to grid points...')
        model_avg['interpolated_pelt_data'] = interpolate_other_to_PINNthickness(pelt_thickness[['lon', 'lat']].values, pelt_thickness['data'].values, model_avg.x,  model_avg.y) 

        model_avg.to_csv(cv_dir+'/model_avg_grid.csv', index=False)
    
    else:
        model_avg = pd.read_csv(cv_dir+'/model_avg_grid.csv', low_memory=False)
    # ---------------------------------------------------------------------------------------------------------

    # compare to the true thickness if we loaded the dataset with measurements
    if grid_or_measurements == 'measurement':
        scatter_preds_with_linearfit(model_avg, 'true_thickness', 'Van Pelt\'s ice thickness vs GPR ice thickness', '/pelt_vs_true.png', compare='interpolated_pelt_data',cv_dir=cv_dir)
        scatter_preds_with_linearfit(model_avg, 'true_thickness', 'Millan\'s ice thickness vs GPR ice thickness', '/millan_vs_true.png', compare='millan_ice_thickness',cv_dir=cv_dir)
        scatter_preds_with_linearfit(model_avg, 'true_thickness', 'Consensus ice thickness vs GPR ice thickness', '/consensus_vs_true.png', compare='consensus_thickness', cv_dir=cv_dir)
        scatter_preds_with_linearfit(model_avg, 'true_thickness', 'PINN ice thickness vs GPR ice thickness', '/pinn_vs_true.png', compare='pinn_thickness', cv_dir=cv_dir)

        print("RMSE Millan vs True: ", np.sqrt(np.mean((model_avg['millan_ice_thickness'] - model_avg['true_thickness'])**2)))
        print("RMSE Consensus vs True: ", np.sqrt(np.mean((model_avg['consensus_thickness'] - model_avg['true_thickness'])**2)))
        print("RMSE PINN vs True: ", np.sqrt(np.mean((model_avg['pinn_thickness'] - model_avg['true_thickness'])**2)))
        print("RMSE Pelt vs True: ", np.sqrt(np.mean((model_avg['interpolated_pelt_data'] - model_avg['true_thickness'])**2)))
        print("MAPE Millan vs True: ", np.mean(np.abs(model_avg['millan_ice_thickness'] - model_avg['true_thickness'])/model_avg['true_thickness']))
        return None

    scatter_subplots(model_avg,  ['consensus_thickness', 'millan_ice_thickness', 'interpolated_pelt_data', 'consensus_thickness', 'millan_ice_thickness', 'interpolated_pelt_data'], ['pinn_thickness', 'pinn_thickness', 'pinn_thickness', 'millan_ice_thickness', 'interpolated_pelt_data','consensus_thickness',], cv_dir+'/scatter_subplots')


    # ---------------------------------------------------------------------------------------------------------

    # Plot mean depth-averaged velocity
    print('Plot mean depth-averaged velocity...')
    plot_depth_avg_velocities(model_avg, cv_dir)

    
    #---------------------------------------------------------------------------------------------------------

    # # Plot mean thickness and coeff of variation of PINN
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7), sharey=True, layout='constrained')

    scatter1 = ax1.scatter(model_avg['lon'], model_avg['lat'], c=model_avg.pinn_thickness, cmap='viridis', s=10, marker='.')
    formatter_spitsbergen.format_spitsbergen_plot(ax1, title='Mean PINN Prediction')
    formatter_spitsbergen.format_spitsbergen_cbar(ax1, scatter1, label='Ice Thickness [m]')
    formatter_spitsbergen.add_coastline(ax1)
    ax1.text(-0.1, 1.05, '(a)', fontsize=fontsize_big, transform=ax1.transAxes)

    scatter2 = ax2.scatter(model_avg['lon'], model_avg['lat'], c=coeff_var_pinn, cmap='Reds', s=10, marker='.')
    formatter_spitsbergen.format_spitsbergen_plot(ax2, title='Variation of PINN Predictions', ylabel=None)
    formatter_spitsbergen.format_spitsbergen_cbar(ax2, scatter2, label='Coefficient of Variation')
    formatter_spitsbergen.add_coastline(ax2)
    ax2.set_ylabel('')
    ax2.text(-0.1, 1.05, '(b)', fontsize=fontsize_big, transform=ax2.transAxes)

    fig.savefig(cv_dir+'/mean_and_coeff_var_pinn_thickness_grid.png', dpi=300)



    # ---------------------------------------------------------------------------------------------------------

    # statistics about variability of the predictions
    print("Statistics about variability of the ice thickness estimates")
    mean_thick_physics_based = model_avg[['consensus_thickness', 'millan_ice_thickness', 'interpolated_pelt_data']].mean(axis=1)
    coeff_var_thick_physics_based = model_avg[['consensus_thickness', 'millan_ice_thickness', 'interpolated_pelt_data']].std(axis=1)/abs(mean_thick_physics_based)
    pinn_thick = results_group.pinn_thickness.mean()
    pinn_thick_std = results_group.pinn_thickness.std()
    coeff_var_pinn = pinn_thick_std/abs(pinn_thick)
    print("PINN thickness standard deviation:")
    print(pinn_thick_std.describe())
    print("PINN thickness coefficient of variation:")
    print(coeff_var_pinn.describe())
    print("90th percentile of pinn thickness std:", pinn_thick_std.quantile(0.9))
    print("90th percentile of coeff_var_pinn:", coeff_var_pinn.quantile(0.9))
    print("Physics-based coff_var:")
    print(coeff_var_thick_physics_based.describe())
    print("physics-based std:", model_avg[['consensus_thickness', 'millan_ice_thickness', 'interpolated_pelt_data']].std(axis=1).describe())
    print("90th percentile of coeff_var_physics-based:", coeff_var_thick_physics_based.quantile(0.9))


def start():
    parser = argparse.ArgumentParser(
    prog="compare_PINN_to_other.py", description="Generate figures to evaluate PINN prediciton and compare to physics-based models."
    )
    parser.add_argument("--gom", 
                        help="grid or measurement parameter; set grid to get figures for the prediciton on the whole grid; set measurement to get figures that compare PINN prediciton to the measurements", 
                        required=False, 
                        default='grid')
    args = parser.parse_args()
    
    cv_dir = "CV/allunmappedglaciers_notsurging/reproduce_tests/test30_spitsbergen_completedhdt_correct_depth_avg_moredata"
    generate_figures(grid_or_measurements=args.gom, cv_dir=cv_dir)
   

if __name__ == "__main__":

    start()
