""" Conduct LOGO CV with leaving out loss components (setting loss component weight to 0) one after another."""
from LOGO_CV import LOGO_CV
from utils import read_config, save_config



loss_components= ['w_pinnloss', 'w_depthAvg', 'w_VelMag', 'w_negative_thickness', 'w_smoothness', 'w_thicknessloss']

for i, loss in enumerate(loss_components):
    print(f"Training with 0 as weight for loss component"+ loss)
    config = read_config("model_config.yaml")
    config["experiment"]["exp_dir"]="CV/allunmappedglaciers_notsurging/loss_weights_test/"+loss
    config["experiment"]["experiment_name"]= "loss"
    config["loss_fn"][loss] = 0
    save_config(config, "CV/allunmappedglaciers_notsurging/loss_weights_test/model_config_"+loss+".yaml")
    
    LOGO_CV("CV/allunmappedglaciers_notsurging/loss_weights_test/model_config_"+loss+".yaml")
