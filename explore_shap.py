# Description: This script is used to explore the SHAP values of a trained model.
import shap
import torch
import numpy as np
from model import PL_PINN
from dataset_new import PL_GlacierDataset
from utils import read_config
import pytorch_lightning as pl
import pandas as pd
import matplotlib.pyplot as plt
import os

from captum.attr import ShapleyValueSampling, DeepLiftShap


# Helper method to print importances and visualize distribution
def visualize_importances(feature_names, importances, save_as='summary_plot', title="Average Feature Importances", plot=True, axis_title="Features"):
    print(title)
    
    x_pos = (np.arange(len(feature_names)))
    if plot:

        plt.figure(figsize=(12,6))
        plt.bar(x_pos, importances, align='center')
        plt.xticks(x_pos, feature_names, wrap=True)
        plt.xlabel(axis_title)
        plt.title(title)
        for i, v in enumerate(importances):
            plt.text(i, v, str(round(v, 2)), ha='center', va='bottom')
        plt.savefig(model_dir+'/'+save_as+".png")
        plt.close()

pl.seed_everything(42, workers=True)
directory = "CV/allunmappedglaciers_notsurging/reproduce_tests/test30_newvelboudaries"

mean_shap_values_list = []
# Define the feature names
feature_names =  ['x', 'y', 'slope', 'vx','vy','beta_mag','beta_vx', 'beta_vy', 'elevation', 'area', 'dis to border']
feature_labels =  ['x', 'y', 'slope', '$ v_x $','$ v_y $','$ \\beta _\mathrm{mag}$','$ \\beta_{x}$', '$\\beta_y$', 'elevation', 'area', 'dist to \n border']
for subdir in os.listdir(directory):
        # if True:
        #      break
    
        model_dir = os.path.join(directory, subdir)
        if os.path.isdir(model_dir):
            # set the seed for reproducibility
            pl.seed_everything(42, workers=True)
            print(model_dir)

# model_dir = "CV/allunmappedglaciers_notsurging/reproduce_tests/test30_newvelboudaries/new_velboundariesLOGO_RGI60-07.00496_08-01-2024_14:19:51"

            # # Load your trained model
            model = PL_PINN.load_from_checkpoint(model_dir +"/checkpoints/last.ckpt")
            model.eval()  # Set the model to evaluation mode


            # # Move the model to the specified GPU if available
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            device = torch.device('cuda:1')

            print(f"Using device: {device}")
            model.to(device)

            # # Load the configuration file
            config = read_config(model_dir+'/config.yaml')
            rgi_id = config["ds"]["glacier_ids"][0]
            config["ds"]["labeled_sample_size"]=1.0
            config["ds"]["unlabeled_sample_size"]=0.0
            config["ds"]["data_dir_labeled"]=[model_dir+'/labelled_val_data.csv']
            config["ds"]["data_dir_unlabeled"]=[model_dir+'/labelled_val_data.csv']
            config["dataloader"]["batch_size"] = 128
            config["ds"]["glacier_ids"] = []
            # # Prepare your data
            dataset = PL_GlacierDataset(config)
            dataset.setup('validate')
            print(dataset.val_dataset.__len__())
            data_loader = dataset.val_dataloader()

            # Get a batch of data to create baseline
            data_iter = iter(data_loader)
            inputs, _, _ = next(data_iter)
            inputs = inputs.to(device)

            # Create a baseline (e.g., zero tensor of the same shape as input)
            # 0 baseline is fine as we normalized the data to have mean 0 and std 1
            baseline = torch.zeros_like(inputs)

            # Initialize DeepLiftShap
            dl_shap = DeepLiftShap(model)

            all_shap_values = []
            all_inputs = []

            for inputs, _, _ in data_loader:
                inputs = inputs.to(device)
                
                # Compute SHAP values using the mean baseline
                shap_values = dl_shap.attribute(inputs, target=0, baselines=baseline)
                
                # Convert SHAP values to numpy for accumulation
                shap_values_np = shap_values.cpu().detach().numpy()
                inputs_np = inputs.cpu().detach().numpy()
                
                all_shap_values.append(shap_values_np)
                all_inputs.append(inputs_np)

            # Concatenate all SHAP values and inputs
            all_shap_values = np.concatenate(all_shap_values, axis=0)
            all_inputs = np.concatenate(all_inputs, axis=0)

            print(all_shap_values.shape)

            # Visualize the SHAP values

            

            # make barplot of the absolute values of the shap values
            visualize_importances(feature_labels, np.absolute(all_shap_values).mean(axis=0),save_as="barplot_absolute_feature_importance_all")

            # make summary plot of the shap values but without x and y as they are way bigger than the other features
            
            f = shap.summary_plot(all_shap_values[:,2:], all_inputs[:,2:], feature_names=feature_labels[2:], plot_size=[7,6])
            #plt.tight_layout()
            # Adjust the font size of the labels
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.xlabel('SHAP value', fontsize=20)  # Adjust the label and fontsize as needed
            # Access the colorbar and set its label size
            cbar = plt.gcf().axes[-1]  # Get the colorbar axis
            cbar.tick_params(labelsize=20)  # Set the colorbar tick label size
            # Set the colorbar label size
            cbar.set_ylabel(cbar.get_ylabel(), fontsize=20, labelpad=1.0)  # Adjust the fontsize as needed
                 

            # shap.summary_plot(all_shap_values, all_inputs, feature_names=feature_names)
            # Save the SHAP summary plot
            plt.savefig(model_dir +'/shap_summaryplot.png', dpi=300, bbox_inches='tight')
            plt.close()
            
   
            # # Save the SHAP values

            # # Optionally, save SHAP values as a pandas DataFrame
            # import pandas as pd
            #add shap values to the dataframe
            
            mean_shap_values_list.append(np.absolute(all_shap_values).mean(axis=0))

# df = pd.DataFrame(mean_shap_values_list, columns=feature_names)
df = pd.read_csv(directory+'/captum_shap_values.csv')
# Calculate the mean and standard deviation of the SHAP values
mean_shap_values = df.mean(axis=0)
std_shap_values = df.std(axis=0)

# Plot the mean SHAP values with error bars indicating the standard deviation
# plt.figure(figsize=(12, 10))

# plt.bar(feature_names, mean_shap_values, align='center', yerr=std_shap_values, capsize=5)
# plt.xlabel("Features", fontsize=16)
# plt.ylabel("Mean SHAP Values", fontsize=16)
# plt.xticks(fontsize=14, rotation=45) 
# plt.yticks(fontsize=14)
# plt.title("Mean SHAP Values with Standard Deviation", fontsize=16)
# # plt.xticks()

# plt.savefig(directory+'/mean_shap_values.png', dpi=300)
# plt.show()


print('Mean: ', df.mean(axis=0))
print('Describe: ', df.describe())
# import pdb; pdb.set_trace()
#df.to_csv(directory+'/captum_shap_values.csv', index=False)


import matplotlib.image as mpimg
from PIL import Image

# Load the saved figure
image_path = directory+"/new_velboundariesLOGO_RGI60-07.00240_08-01-2024_12:20:54/shap_summaryplot.png"
loaded_image = Image.open(image_path)
# import pdb; pdb.set_trace()
resized_image = loaded_image.resize((loaded_image.size[0]*1, loaded_image.size[1]*1))  # Adjust the size as needed


# Convert the PIL image to a NumPy array
resized_image_np = np.array(resized_image)

# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 8), gridspec_kw={'width_ratios': [0.7, 1]})


axs[0].bar(feature_names, mean_shap_values, align='center', yerr=std_shap_values, capsize=5)
#axs[0].xlabel("Features", fontsize=16)
axs[0].set_ylabel("Mean absolute SHAP Values",fontsize=14)
axs[0].set_xticklabels(feature_labels,  rotation=90, fontsize=12,) 
axs[0].tick_params(axis='y', labelsize=12)
axs[0].text(-0.1, 1.2, '(a)', fontsize=14, transform=axs[0].transAxes)
axs[0].set_aspect(aspect=1.7) 

# Display the loaded image in the first subplot
axs[1].imshow(resized_image_np)
axs[1].axis('off')  # Hide the axis
axs[1].text(0, 1.0, '(b)', fontsize=14, transform=axs[1].transAxes)
# axs[1].set_title('Loaded Figure', fontsize=16)



# Adjust layout and show the plot

plt.savefig(directory+'/SHAP_analysis.png', dpi=300, bbox_inches='tight')