"""Implementing custom loss functions for the physics-aware neural network."""

import torch


# Losses are inspired from Teisberg et al. (2021): 'A Machine Learning Approach to Mass-Conserving Ice Thickness Interpolation' 
# 2021 IEEE International Geoscience and Remote Sensing Symposium IGARSS


        
class NegThick_loss(torch.nn.Module):
    """Punishes negative predictions for ice thickness Scales the predictions back to the physical domain.

    Args:
        weight (float): The weight to apply to the loss.

    Returns:
        torch.Tensor: The calculated loss value.

    """
    def __init__(self, weight: float=1.) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, pred, target_std, target_mean):
        return torch.mean(torch.square(torch.clamp(pred[:,0]*target_std[0]+target_mean[0], max=0))) * self.weight
    

 

class Mass_conservation_loss(torch.nn.Module):
    """Calculates the mass conservation loss for a given model output and target.
    
    The mass conservation loss is calculated as the squared difference between the divergence of the ice flux and the apparent mass balance.
    The divergence is calculated as div(hv) = h_x*vx + h*vx_x + h_y*vy + h*vy_y, where h, vx, vy are components of the model output,
    and h_x, h_y, vx_x, vy_y are the gradients of these components with respect to the model input.
    
    Args:
        weight (float, optional): Weight to apply to the loss. Defaults to 1.0.
    """
    def __init__(self, weight: float=1.0, x_idx=0, y_idx=1, vel_x_idx=3, vel_y_idx=4, h_idx=0, depth_vel_x_idx=1, depth_vel_y_idx=2, h_true_idx=0, mb_idx=1) -> None:
        super().__init__()
        self.weight = weight
        # set the indices of the model input, output, and target components, can be changed if the order of the inputs or outputs changes.
        
        self.x_idx = x_idx
        self.y_idx = y_idx
        self.vel_x_idx = vel_x_idx
        self.vel_y_idx = vel_y_idx
        self.h_idx = h_idx
        self.dvel_x_idx = depth_vel_x_idx
        self.dvel_y_idx = depth_vel_y_idx
        self.h_true_idx = h_true_idx
        self.mb_idx = mb_idx
        
        
    def forward(self, model_output, model_input, target, input_std, input_mean, target_std, target_mean):
        """
        Calculates the mass conservation loss for a given model output and target.
        
        Args:
            model_output (torch.Tensor): Output of the model. (h, vx, vy)
            model_input (torch.Tensor): Input to the model. (x, y)
            target (torch.Tensor): Target values. (Apparent mass balance)
            input_std (torch.Tensor): Standard deviation of the input.
            input_mean (torch.Tensor): Mean of the input.
            target_std (torch.Tensor): Standard deviation of the target.
            target_mean (torch.Tensor): Mean of the target.
        
        Returns:
            torch.Tensor: The mass conservation loss.
        """

        h_x = torch.autograd.grad(model_output[:,self.h_idx], model_input, 
                                          grad_outputs=torch.ones_like(model_output[:,self.h_idx]).type_as(model_input), 
                                          create_graph=True, retain_graph=True)[0][:,self.x_idx]*target_std[self.h_true_idx]/input_std[self.x_idx]
        h_y = torch.autograd.grad(model_output[:,self.h_idx], model_input, 
                                          grad_outputs=torch.ones_like(model_output[:,self.h_idx]).type_as(model_input), 
                                          create_graph=True, retain_graph=True)[0][:,self.y_idx]*target_std[self.h_true_idx]/input_std[self.y_idx]
        vx_x = torch.autograd.grad(model_output[:,self.dvel_x_idx], model_input, 
                                          grad_outputs=torch.ones_like(model_output[:,self.dvel_x_idx]).type_as(model_input), 
                                          create_graph=True, retain_graph=True)[0][:,self.x_idx]*input_std[self.vel_x_idx]/input_std[self.x_idx]
        vy_y = torch.autograd.grad(model_output[:,self.dvel_y_idx], model_input, 
                                          grad_outputs=torch.ones_like(model_output[:,self.dvel_y_idx]).type_as(model_input), 
                                          create_graph=True, retain_graph=True)[0][:,self.y_idx]*input_std[self.vel_y_idx]/input_std[self.y_idx]

        h = model_output[:,self.h_idx]*target_std[self.h_true_idx]+target_mean[self.h_true_idx]
        vx = model_output[:,self.dvel_x_idx]*input_std[self.vel_x_idx]+input_mean[self.vel_x_idx]
        vy = model_output[:,self.dvel_y_idx]*input_std[self.vel_y_idx]+input_mean[self.vel_y_idx]
        mb = target[:,self.mb_idx]*target_std[self.mb_idx]+target_mean[self.mb_idx]

        div = h_x*vx +h*vx_x+h_y*vy+h*vy_y
        
        return torch.square(div-mb)


class Thickness_smoothing_loss(torch.nn.Module):
    """From Teisberg et al.: thickness distribution should not have sharp edges. """
    def __init__(self, weight: float=1.0, x_idx=0, y_idx=1,  h_idx=0,  h_true_idx=0) -> None:
        super().__init__()
        self.weight = weight
        # set the indices of the model input, output, and target components, can be changed if the order of the inputs or outputs changes.
        
        self.x_idx = x_idx
        self.y_idx = y_idx
        self.h_idx = h_idx
        self.h_true_idx = h_true_idx


    def forward(self, model_output, model_input, input_std, target_std):
        """Punishes large gradients in the ice thickness prediction of the model: div(H)=0 will be enforced. 
        Args:
            model_output (torch.Tensor): Output of the model.
            model_input (torch.Tensor): Input to the model, x and y coordinates needed for the gradient calculation.
            input_std (torch.Tensor): Standard deviation of the input to scale back to the physical domain.
            target_std (torch.Tensor): Standard deviation of the target to scale back to the physical domain.
        Returns:   
            torch.Tensor: The calculated loss value averaged over x and y gradients and multiplied by the weight."""
    
        h_x = torch.autograd.grad(model_output[:,self.h_idx], model_input, 
                                          grad_outputs=torch.ones_like(model_output[:,self.h_idx]).type_as(model_input), 
                                          create_graph=True, retain_graph=True)[0][:,self.x_idx]*target_std[self.h_true_idx]/input_std[self.x_idx]
        h_y = torch.autograd.grad(model_output[:,self.h_idx], model_input, 
                                          grad_outputs=torch.ones_like(model_output[:,self.h_idx]).type_as(model_input), 
                                          create_graph=True, retain_graph=True)[0][:,self.y_idx]*target_std[self.h_true_idx]/input_std[self.y_idx]
        return (torch.mean(torch.square(h_x)+torch.square(h_y)))*self.weight
    

class Velocity_smoothing_loss(torch.nn.Module):
    """Jonathan Bamber: velocities should be smooth. 
    In Practice the loss is really small and does not have any effect unless we artificially set its weight super high."""
    def __init__(self, weight: float=1.0, x_idx=0, y_idx=1, vel_x_idx=3, vel_y_idx=4, depth_vel_x_idx=1, depth_vel_y_idx=2) -> None:
        super().__init__()
        self.weight = weight
        # set the indices of the model input, output, and target components, can be changed if the order of the inputs or outputs changes.       
        self.x_idx = x_idx
        self.y_idx = y_idx
        self.vel_x_idx = vel_x_idx
        self.vel_y_idx = vel_y_idx
        self.dvel_x_idx = depth_vel_x_idx
        self.dvel_y_idx = depth_vel_y_idx


    def forward(self, model_output, model_input, input_std):

       
        vx_x = torch.autograd.grad(model_output[:,self.dvel_x_idx], model_input, 
                                          grad_outputs=torch.ones_like(model_output[:,self.dvel_x_idx]).type_as(model_input), 
                                          create_graph=True, retain_graph=True)[0][:,self.x_idx]*input_std[self.vel_x_idx]/input_std[self.x_idx]
        vx_y = torch.autograd.grad(model_output[:,self.dvel_x_idx], model_input, 
                                          grad_outputs=torch.ones_like(model_output[:,self.dvel_x_idx]).type_as(model_input), 
                                          create_graph=True, retain_graph=True)[0][:,self.y_idx]*input_std[self.vel_x_idx]/input_std[self.y_idx]
        vy_x = torch.autograd.grad(model_output[:,self.dvel_y_idx], model_input, 
                                          grad_outputs=torch.ones_like(model_output[:,self.dvel_y_idx]).type_as(model_input), 
                                          create_graph=True, retain_graph=True)[0][:,self.x_idx]*input_std[self.vel_y_idx]/input_std[self.x_idx]
        vy_y = torch.autograd.grad(model_output[:,self.dvel_y_idx], model_input, 
                                          grad_outputs=torch.ones_like(model_output[:,self.dvel_y_idx]).type_as(model_input), 
                                          create_graph=True, retain_graph=True)[0][:,self.y_idx]*input_std[self.vel_y_idx]/input_std[self.y_idx]
        
        return (torch.mean(torch.square(vx_x)+torch.square(vx_y))+
                           torch.mean(torch.square(vy_x)+torch.square(vy_y)))*self.weight



        


class DepthAvgVel_loss(torch.nn.Module):
    """
    Calculates the depth-averaged velocity loss for a given model prediction and target.

    Args:
        vel_lowerbound (float): The lower bound threshold for the depth-averaged velocity.
        weight (float): The weight to apply to the loss.
        vel_x_idx (int): The index of the x-component of the surface velocity in the model input.
        vel_y_idx (int): The index of the y-component of the surface velocity in the model input.
        depth_vel_x_idx (int): The index of the x-component of the depth-averaged velocity in the model prediction.
        depth_vel_y_idx (int): The index of the y-component of the depth-averaged velocity in the model prediction.
        h_true_idx (int): The index of the true depth in the model input.
        beta_x_idx (int): The index of the x-component of the basal sliding correction factor in the model input.
        beta_y_idx (int): The index of the y-component of the basal sliding correction factor in the model input.
    """
    def __init__(self, vel_lowerbound: float=0.7, weight: float=1.0, vel_x_idx=3, vel_y_idx=4, depth_vel_x_idx=1, depth_vel_y_idx=2, h_true_idx=0, beta_x_idx=6, beta_y_idx=7) -> None:
        super().__init__()
        self.weight = weight
        self.lower = vel_lowerbound
        # set the indices of the model input, output, and target components, can be changed if the order of the inputs or outputs changes.
        self.vel_x_idx = vel_x_idx
        self.vel_y_idx = vel_y_idx
        self.dvel_x_idx = depth_vel_x_idx
        self.dvel_y_idx = depth_vel_y_idx
        self.h_true_idx = h_true_idx
        self.beta_x_idx = beta_x_idx
        self.beta_y_idx = beta_y_idx



    def forward(self, pred, model_input, input_std, input_mean):
        """
        Calculates the forward pass of the depth-averaged velocity loss.

        Args:
            pred (torch.Tensor): The model prediction.
            model_input (torch.Tensor): The model input.
            input_std (torch.Tensor): The standard deviation of the input scaler.
            input_mean (torch.Tensor): The mean of the input scaler.

        Returns:
            torch.Tensor: The depth-averaged velocity loss multiplied by the weight.
        """
        vx_avg = pred[:,self.dvel_x_idx]*input_std[self.vel_x_idx]+input_mean[self.vel_x_idx] # scale back to orignial velocity is needed as we are taking the signum of the surface vevlocity and the right signum is only when scaled back?
        vy_avg = pred[:,self.dvel_y_idx]*input_std[self.vel_y_idx]+input_mean[self.vel_y_idx]
        vx_surface = model_input[:,self.vel_x_idx]*input_std[self.vel_x_idx]+input_mean[self.vel_x_idx]
        vy_surface = model_input[:,self.vel_y_idx]*input_std[self.vel_y_idx]+input_mean[self.vel_y_idx]
        beta_x = model_input[:,self.beta_x_idx]*input_std[self.beta_x_idx]+input_mean[self.beta_x_idx] # beta to correct basal sliding, coming from ratio between slope and velocity magnitude
        beta_y = model_input[:,self.beta_y_idx]*input_std[self.beta_y_idx]+input_mean[self.beta_y_idx] # beta to correct basal sliding, coming from ratio between slope and velocity magnitude

        vel_x_loss = velocity_loss(vx_surface, vx_avg, self.lower,  beta_x)
        vel_y_loss = velocity_loss(vy_surface, vy_avg, self.lower,  beta_y)
        
        velocity_data_loss = (torch.mean(vel_x_loss) + torch.mean(vel_y_loss)) / 2

        return velocity_data_loss*self.weight


    

class DepthAvgVelMag_loss(torch.nn.Module):
    """
    Calculates the depth-averaged velocity magnitude loss.

    Args:
        vel_lowerbound (float, optional): The lower bound for the depth-averaged velocity. Defaults to 0.7.
        weight (float, optional): The weight for the loss. Defaults to 1.0.
        vel_x_idx (int, optional): The index of the x-component of the velocity in the model input. Defaults to 3.
        vel_y_idx (int, optional): The index of the y-component of the velocity in the model input. Defaults to 4.
        depth_vel_x_idx (int, optional): The index of the x-component of the depth-averaged velocity in the model input. Defaults to 1.
        depth_vel_y_idx (int, optional): The index of the y-component of the depth-averaged velocity in the model input. Defaults to 2.
        beta_mag_idx (int, optional): The index of the beta magnitude in the model input. Defaults to 5.
    """

    def __init__(self, vel_lowerbound: float=0.7, weight: float=1.0, vel_x_idx=3, vel_y_idx=4, depth_vel_x_idx=1, depth_vel_y_idx=2, beta_mag_idx=5) -> None:
        super().__init__()
        self.lower = vel_lowerbound
        self.weight = weight
        self.vel_x_idx = vel_x_idx
        self.vel_y_idx = vel_y_idx
        self.dvel_x_idx = depth_vel_x_idx
        self.dvel_y_idx = depth_vel_y_idx
        self.beta_mag_idx = beta_mag_idx

    def forward(self, pred, model_input, input_std, input_mean):
        """
        Calculates the forward pass of the depth-averaged velocity magnitude loss.

        Args:
            pred (torch.Tensor): The predicted values.
            model_input (torch.Tensor): The input values to the model.
            input_std (torch.Tensor): The scaler standard deviation of the input values.
            input_mean (torch.Tensor): The scaler mean of the input values.

        Returns:
            torch.Tensor: The calculated loss.
        """
        vx_avg = pred[:,self.dvel_x_idx]*input_std[self.vel_x_idx]+input_mean[self.vel_x_idx]
        vy_avg = pred[:,self.dvel_y_idx]*input_std[self.vel_y_idx]+input_mean[self.vel_y_idx]
        vx_surface = model_input[:,self.vel_x_idx]*input_std[self.vel_x_idx]+input_mean[self.vel_x_idx]
        vy_surface = model_input[:,self.vel_y_idx]*input_std[self.vel_y_idx]+input_mean[self.vel_y_idx]
        beta = model_input[:,self.beta_mag_idx]*input_std[self.beta_mag_idx]+input_mean[self.beta_mag_idx]

        pred_mag = torch.sqrt(torch.square(vx_avg)+torch.square(vy_avg))
        obs_mag = torch.sqrt(torch.square(vx_surface)+torch.square(vy_surface))
        
        zero_loss = torch.tensor(0).to(pred.device)

        too_high = torch.where(condition=(obs_mag)<pred_mag, input=(obs_mag-pred_mag)**2, other=zero_loss)
        too_low = torch.where(condition=pred_mag<=((self.lower+(1-self.lower)*beta)*obs_mag), input=((self.lower+(1-self.lower)*beta)*obs_mag-pred_mag)**2, other=zero_loss)

        return torch.mean(too_high+too_low)*self.weight





def velocity_loss(v_surf, v_d, lower_bound, beta):
    """
    Calculates the velocity loss based on the surface velocity, desired velocity, and lower bound.

    Args:
        v_surf (torch.Tensor): Surface velocity.
        v_d (torch.Tensor): Desired depth-averaged velocity.
        lower_bound (float): Lower bound value.

    Returns:
        torch.Tensor: Velocity loss.

    Raises:
        ValueError: If the dimensions of the loss and v_d do not match.
    """
    v_surf_sign = torch.sign(v_surf)
    v_d_sign = torch.sign(v_d)
    # v_surface = v_basal_sliding+ v_deformation --> beta = 90%: 90% basal sliding (from Millan et al. 2022) 
    # if v_surf = v_basal_sliding --> v_d = v_surf
    # if v_basal_sliding = 0 (beta=0) --> v_d <= lower bound * v_surf
    
    abs_too_high_same_sign = torch.where(abs(v_surf) < abs(v_d), input= v_d - v_surf, other=torch.tensor(0))
    too_low_same_sign = torch.where(((lower_bound+(1-lower_bound)*beta) * v_surf - v_d) * v_surf_sign > 0, input=((lower_bound+(1-lower_bound)*beta) * v_surf - v_d), other=torch.tensor(0)) # only count error if v_d is lower than the lower bound would allow
    diff_sign = (lower_bound+(1-lower_bound)*beta) * v_surf - v_d

    # if the sign is different we only count the error diff_sign error, 
    # if the sign is the same we count the error of the too high error (if absolute value is too high) 
    # or the too low value (if the difference between the lower bound and the desired value is the same sign as the surface velocity)
    loss = torch.where((v_surf_sign * v_d_sign) < 0, input=diff_sign, other=abs_too_high_same_sign+too_low_same_sign)

    if loss.shape != v_d.shape:
        raise ValueError("The dimensions of the loss and v_d do not match.")
    return torch.square(loss)


class velocity_sign_error(torch.nn.Module):
    """Should punish whenever the sign of the predicted velocity is different from the sign of the surface velocity.
    This is basically already included in the velocity loss but might be useful to have it as a separate loss."""
    def __init__(self, weight) -> None:
        super().__init__()
        self.weight = weight

    def forward(self,v_avg, v_surface, input_std, input_mean):
        v_avg_unscaled = v_avg*input_std+input_mean
        v_surface_unscaled = v_surface*input_std+input_mean
        return self.weight*torch.mean(torch.where(v_avg_unscaled*v_surface_unscaled<0, torch.abs(v_avg_unscaled)+torch.abs(v_surface_unscaled), torch.tensor(0)))
                