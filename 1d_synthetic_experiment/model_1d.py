import matplotlib.pyplot as plt
import os
import copy
import pytorch_lightning as pl
import torch
import warnings
from torch import nn
from torchmetrics import MeanAbsolutePercentageError
from typing import Any, Dict, Tuple


from losses_1d import  Mass_conservation_loss,  DepthAvgVel_loss, DepthAvgVelMag_loss, NegThick_loss, Thickness_smoothing_loss





class FourierFeaturesLayer(nn.Module):
    def __init__(self,  gaussian_mapping_dim: int=None, coordinate_dim: int=2, gaussian_scale: float=10.) -> None:
        """takes tensor with dimension [batch_size, ..., coord_dim+feature_dim] and performs gaussian feature mapping 
        returns tensor of dim [batch_size,..., gaussian_mapping_dim*2+feature_dim]"""
        super().__init__()
        self.coord_dim = coordinate_dim
        self.gaussian_mapping_dim = gaussian_mapping_dim
        self.gaussian_scale = gaussian_scale
        self.set_gaussian_matrix()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        assert x.shape[1] >= self.coord_dim, "Input tensor does not have enough dimensions"
        if self.gaussian_mapping_dim is not None:
            coords = x[:,:self.coord_dim]
            coords = self.fourier_mapping(coords, self.B)
            x = torch.cat([coords, x[:,self.coord_dim:]], dim=-1)
        return x

     # Fourier feature mapping from https://doi.org/10.48550/arXiv.2006.10739
    def fourier_mapping(self, x: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        if B is None:
            return x
        else: 
           
            B = B.to(x.device)
            x_proj = (2.*torch.pi*x) @ B
            return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], axis=-1)
        
    
    def set_gaussian_matrix(self):
        if self.gaussian_mapping_dim is None:
            # Standard network - no mapping
            self.B = None
        else:
            # generate random gaussian mapping matrix
            seed = torch.Generator().manual_seed(42)
            self.B = torch.randn(self.coord_dim, self.gaussian_mapping_dim, generator=seed) * self.gaussian_scale




# https://www.geeksforgeeks.org/training-neural-networks-using-pytorch-lightning/


class PL_PINN(pl.LightningModule):
    """
    PyTorch Lightning module for Physics-Informed Neural Networks (PINN).

    Args:
        config (Dict[str, Any]): Configuration dictionary containing various parameters for the model.

    Attributes:
        config (Dict[str, Any]): Configuration dictionary containing various parameters for the model.
        output_path (str): Path to save the training outputs.
        coordinate_dim (int): Dimension of the coordinate input.
        feature_dim (int): Dimension of the feature input.
        model (torch.nn.Sequential): Sequential model architecture.
        gauss_mapping_dim (int): Dimension of the Gaussian mapping layer.
        layer_dims (List[int]): List of layer dimensions for the linear blocks.
        learning_rate (float): Learning rate for the optimizer.
        physics_loss (PhysicsAwareLoss_unscaled): Physics-aware loss function.
        w_physics_loss (float): Weight for the physics loss.
        reconstruction_loss (torch.nn.MSELoss): Reconstruction loss function.
        w_thickness_loss (float): Weight for the thickness loss.
        depth_averaged_velocity_loss (DepthAvgVelLoss_improved): Depth-averaged velocity loss function.
        vel_mag_loss (VelocityMagnitudeLoss_improved): Velocity magnitude loss function.
        negative_thickness_loss (NegThickLoss): Negative thickness loss function.
        smooth_thickness_loss (Thickness_smoothing_Loss): Thickness smoothing loss function.
        validation_loss (torch.nn.MSELoss): Validation loss function.
        mean_abs_percentage_error (MeanAbsolutePercentageError): Mean absolute percentage error metric.
        target_scale (Optional[torch.Tensor]): Scale values for the target variable.
        target_mean (Optional[torch.Tensor]): Mean values for the target variable.
        input_scale (Optional[torch.Tensor]): Scale values for the input features.
        input_mean (Optional[torch.Tensor]): Mean values for the input features.
        burn_in (int): Number of burn-in epochs.
        burn_out (int): Number of burn-out epochs.

    Methods:
        layer_block: Creates a sequential layer block.
        training_step: Performs a single training step.
        validation_step: Performs a single validation step.
        on_validation_start: Called at the start of the validation loop.
        predict_step: Performs a single prediction step.
        select_problematic_sample_indices_from_batch_errors: Selects indices of problematic samples from a batch of errors.
        plot_physics_aware_error: Plots the physics-aware error.
        configure_optimizers: Configures the optimizer for training.

    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = copy.deepcopy(config)

        # create model architecture
        self.model = torch.nn.Sequential()

        # Add gaussian mapping layer if specified in config 
        self.coordinate_dim = 1 # two dimensions for the coordinates in x and y direction
        self.feature_dim = len(config["ds"]["input_features"])-self.coordinate_dim  
        
        try:
            if config["model"]["gaussian_mapping_dim"] is not None:           
                self.gauss_mapping_dim = config["model"]["gaussian_mapping_dim"] 
                self.model.add_module("FourierLayer", FourierFeaturesLayer(self.gauss_mapping_dim, coordinate_dim=self.coordinate_dim, gaussian_scale=config["model"]["gaussian_scale"]))
                self.coordinate_dim = self.gauss_mapping_dim*2
        except KeyError:
             warnings.warn("Gaussian mapping dimension not found in config. Fourier layer will not be added.")

        # Add blocks of dense layers and softplus activation functions as specified in config
        self.layer_dims = [self.coordinate_dim+self.feature_dim]+ [config["model"]["hidden_dim"]]*config["model"]["num_layers"] 
        linear_blocks = [self.layer_block(in_f, out_f) for in_f, out_f in zip(self.layer_dims, self.layer_dims[1:])]
        for i, block in enumerate(linear_blocks): 
            self.model.add_module("softplusblock"+str(i), block)
        self.model.add_module("finallayer", nn.Linear(config["model"]["hidden_dim"], config["model"]["output_dim"]))
       
        # set learning rate
        self.learning_rate = config["optimizer"]["lr"]

        # set training losses
        self.masscons_loss = Mass_conservation_loss()    
        self.w_physics_loss=config["loss_fn"]["w_pinnloss"]
        self.reconstruction_loss = nn.MSELoss()
        self.w_thickness_loss=config["loss_fn"]["w_thicknessloss"]
        self.depth_averaged_velocity_loss = DepthAvgVel_loss(vel_lowerbound=config["loss_fn"]["vel_lowerbound"], 
                                                                     weight=config["loss_fn"]["w_depthAvg"])
        self.vel_mag_loss = DepthAvgVelMag_loss(vel_lowerbound=config["loss_fn"]["vel_lowerbound"], 
                                                                   weight=config["loss_fn"]["w_VelMag"])
        self.negative_thickness_loss = NegThick_loss(weight=config["loss_fn"]["w_negative_thickness"])
        self.smooth_thickness_loss = Thickness_smoothing_loss(weight=config["loss_fn"]["w_smoothness"])
       # self.smooth_vel_loss = Velocity_smoothing_loss(weight=config["loss_fn"]["w_smoothness_vel"])
        
        # validation losses
        self.validation_loss = nn.MSELoss()
        self.mean_abs_percentage_error = MeanAbsolutePercentageError()

        # burn-in epochs for the training process: fit the model to the data at first. That makes sense to narrow down the space of possible solutions to the PDE.
        self.burn_in = config["loss_fn"]["burn_in_epochs"]
        # burn-out: fit the model only with physics-aware losses after a certain number of epochs. Was not used in the final model as it strongly degraded the performance.
        if "burn_out_after_epochs" in config["loss_fn"]:
            self.burn_out = config["loss_fn"]["burn_out_after_epochs"] 
        else: self.burn_out = config["pl"]["max_epochs"]

        # scaler values needed for loss calculations in the physical domain. They will be set from the datamodule at the start of the training, validation or prediction loop.  
        self.target_scale = None
        self.input_scale = None
        self.target_mean = None
        self.input_mean = None

        # save hyperparameters to ensure reproducibility
        self.save_hyperparameters()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

 
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)       
      
    def layer_block(self, in_f: int, out_f: int, *args, **kwargs) -> nn.Sequential:
        '''
        Args:
            in_f (int): Input dimension of the linear layer.
            out_f (int): Output dimension of the linear layer.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            Returns:
            nn.Sequential: Sequential layer block with a linear layer and a softplus activation function.'''
        return nn.Sequential(
            nn.Linear(in_f, out_f, *args, **kwargs),
            nn.Softplus() # do not use ReLU because it is only differentiable once and not smooth at 0
            )
    
    
    def on_train_start(self) -> None:
        '''Called at the start of the training loop to set the scaler values as we need them for the calculation of physics aware losses.'''
        if self.target_scale is None:
            self.target_scale = self.trainer.datamodule.target_scaler.scale_
            self.input_scale = self.trainer.datamodule.scaler.scale_
            self.target_mean = self.trainer.datamodule.target_scaler.mean_
            self.input_mean = self.trainer.datamodule.scaler.mean_
    
    def training_step(self, train_batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        '''Performs a single training step.'''
        x, y, idx = train_batch
        x.requires_grad = True
        pred = self.model(x)

        # Loss caluclation

        # thickness loss, only for labeled data -> mask out nan values
        label_mask = ~torch.isnan(y[:,0].squeeze()) 
        thickness_loss = self.reconstruction_loss(pred[:,0][label_mask], y[:,0][label_mask])   

        # physics-informed losses
        physics_loss = self.masscons_loss(pred, x, y, self.input_scale, self.input_mean, self.target_scale, self.target_mean)
        physics_loss_avg = torch.mean(physics_loss)
    
        negative_thickness = self.negative_thickness_loss(pred, self.target_scale, self.target_mean)         
        smoothness = self.smooth_thickness_loss(pred, x,  self.input_scale, self.target_scale)
        
        # velocity losses   
        depth_avg_veloc_loss = self.depth_averaged_velocity_loss(pred, x, self.input_scale, self.input_mean) 
        vel_mag_loss = self.vel_mag_loss(pred, x, self.input_scale, self.input_mean)
        #smooth_vel = self.smooth_vel_loss(pred, x, self.input_scale)

        # add losses together
        overall_loss = depth_avg_veloc_loss + vel_mag_loss + smoothness + negative_thickness #+ smooth_vel
        if not torch.isnan(thickness_loss):    # thickness loss only added if it is not nan for the entire batch
            overall_loss += self.w_thickness_loss * thickness_loss
        # add mass conservation loss only after the burn in phase
        if self.current_epoch >= self.burn_in: 
            overall_loss += self.w_physics_loss * physics_loss_avg 
        # Logging
        self.log('Thickness_loss', self.w_thickness_loss*thickness_loss)
        self.log('Depth_averaged_velocity_loss', depth_avg_veloc_loss)
        self.log('Velocity_magnitude_loss', vel_mag_loss)
        self.log('PINN_loss', self.w_physics_loss*physics_loss_avg)
        self.log('negative thickness prediction loss', negative_thickness)
        self.log('Thickness smoothing loss', smoothness)
        #self.log('Velocity smoothing loss', smooth_vel)
        self.log('overall train loss', overall_loss)

        return {"loss": overall_loss, "physics_loss": physics_loss, "idx": idx, "sample": x}
    
    
    def on_validation_start(self) -> None:
        '''Called at the start of the validation loop to set the scaler values for the loss calculations.'''
        if self.target_scale is None:
            self.target_scale = self.trainer.datamodule.target_scaler.scale_
            self.input_scale = self.trainer.datamodule.scaler.scale_
            self.target_mean = self.trainer.datamodule.target_scaler.mean_
            self.input_mean = self.trainer.datamodule.scaler.mean_

    def validation_step(self, valid_batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        '''Performs a single validation step. Returns the thickness RMSE and the mean absolute percentage error.'''
        input, target, _ = valid_batch
        pred = self.model(input)
        
        # validation only for labelled data -> mask out nan values
        label_mask = ~torch.isnan(target[:,0].squeeze())
        valid_thick_pred = pred[:,0][label_mask]*self.target_scale[0]+self.target_mean[0]
        valid_thick_target = target[:,0][label_mask]*self.target_scale[0]+self.target_mean[0]

        thick_error = self.validation_loss(valid_thick_pred,valid_thick_target)
        thick_error = torch.sqrt(thick_error)
        mape_error = self.mean_abs_percentage_error(valid_thick_pred, valid_thick_target)

        self.log('val Thickness RMSE', thick_error)
        self.log('val MAPE', mape_error)
        return  thick_error, mape_error
    

        
    def on_predict_start(self) -> None:
        '''Called at the start of the predict loop to set the scaler values for the transformation back to the physical domain.'''
        
        if self.target_scale is None:
            self.target_scale = self.trainer.datamodule.target_scaler.scale_
            self.input_scale = self.trainer.datamodule.scaler.scale_
            self.target_mean = self.trainer.datamodule.target_scaler.mean_
            self.input_mean = self.trainer.datamodule.scaler.mean_

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        '''Performs a single prediction step. Returns the thickness and depth-averaged velocity predictions scaled back to the physical domain.'''
        x, _, _ = batch
        pred = self.model(x)
        pred = pred.cpu().numpy()
        thickness = pred[:,0]*self.target_scale[0]+self.target_mean[0]
        velocities = pred[:,1]*self.input_scale[1]+self.input_mean[1]

        return thickness, velocities
    

    

    def select_problematic_sample_indices_from_batch_errors(self, errors):
        """
        Selects the indices of problematic samples from a batch of errors. Useful for Adaptive Sampling.

        Args:
            errors (list): A list of error values for each sample in the batch.

        Returns:
            list: A list of indices corresponding to the problematic samples.

        """
        problematic_indices = [i for i, error in enumerate(errors) if error > self.config["ds"]["problematic_sample_threshold"]]
        return problematic_indices
    
    def plot_physics_aware_error(self, x, y, error, epoch):
        """Useful for debugging, """
        self.output_path = os.path.join(self.config["experiment"]["save_dir"], "training_ouputs")
        x_unscaled = x*self.input_scale[0]+self.input_mean[0]
        y_unscaled = y*self.input_scale[1]+self.input_mean[1]
        plt.scatter(x_unscaled, y_unscaled, c=error, cmap='viridis', marker='.')
        plt.colorbar()
        plt.savefig(self.output_path+ "physics_loss_epoch"+str(epoch)+".png")
        plt.close()


    


