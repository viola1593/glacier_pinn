import os
import warnings
import copy

import pandas as pd
import pyproj
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from typing import Callable

### Datasets

class GriddedDataset(Dataset):
    """
    A custom dataset class for gridded data.

    Args:
        config (dict): Configuration parameters for the dataset.
        labelled_data (pd.DataFrame): Labeled data for the dataset.
        unlabelled_data (pd.DataFrame, optional): Unlabeled data for the dataset. Defaults to None.
        transform (callable, optional): A function/transform to be applied to the inputs. Defaults to None.
        target_transform (callable, optional): A function/transform to be applied to the targets. Defaults to None.
        dataset_type (str, optional): Type of the dataset (training or validation). Defaults to None.

    Attributes:
        dataset_type (str): Type of the dataset, either 'training' or 'validation'.
        config (dict): Configuration parameters for the dataset.
        transform (callable): A function/transform to be applied to the inputs.
        target_transform (callable): A function/transform to be applied to the targets.
        unlabeled_dataset (pd.DataFrame): Unlabeled part of the dataset.
        labeled_dataset (pd.DataFrame): Labeled part of the dataset.
        unused_points (pd.DataFrame): Unused points in the dataset.
        unlabeled_target (pd.DataFrame): Target values for the unlabeled dataset.
        unlabeled_inputs (pd.DataFrame): Input features for the unlabeled dataset.
        labeled_target (pd.DataFrame): Target values for the labeled dataset.
        target (pd.DataFrame): Concatenated target values for both labeled and unlabeled datasets.
        inputs (pd.DataFrame): Concatenated input features for both labeled and unlabeled datasets.

    Methods:
        prepare_labelled: Prepare the labeled part of the dataset.
        prepare_inputs_and_targets: Prepare the inputs and targets for the dataset.
        leave_glacier_out: Remove all datapoints of a specific glacier from the dataset.
        remove_all_but_one_glacier: Remove all datapoints of all but one glacier from the dataset.
        _transform_projection: Transform the lon-lat values in the dataset to a specific projection.
        _select_points: Select a specific number of points from a group.
        _apply_scaler: Apply standardization to the inputs and targets.
        __len__: Get the length of the dataset.
        __getitem__: Get a specific item from the dataset.
    """

    def __init__(self, config: dict, labelled_data: pd.DataFrame, unlabelled_data: pd.DataFrame=None, transform: Callable[[any], any]=None, target_transform: Callable[[any], any]=None, dataset_type: str=None) -> None:
        super().__init__()
        self.dataset_type = dataset_type
        self.ds_config = copy.deepcopy(config)
        # scalers for the dataset
        self.transform = transform
        self.target_transform = target_transform
        
        ## prepare unlabeled part of dataset if there is any
        if unlabelled_data is not None:
            self.unlabeled_dataset= unlabelled_data.copy(deep=True) #get unlabelled dataset
            self.unlabeled_dataset = self.unlabeled_dataset.sample(frac=self.ds_config["unlabeled_sample_size"], random_state=42, replace=False) #take a sample of the unlabelled dataset with fraction defined in config file
            if 'beta_to_zero' in self.ds_config and self.ds_config["beta_to_zero"]:
                self.unlabeled_dataset['beta'] = 0
        else:
            self.unlabeled_dataset = pd.DataFrame()


        #prepare labelled part of dataset
        self.labeled_dataset = self.prepare_labelled_dataset(labelled_data)

        # finally create inputs and targets of the dataset
        self.inputs, self.target = self.prepare_inputs_and_targets() 
        

    def prepare_labelled_dataset(self, labelled_data: pd.DataFrame)-> pd.DataFrame:
        '''Prepare the labelled part of the dataset.
        Args:
            labelled_data (pd.DataFrame): The labelled part of the dataset.'''
        if labelled_data is not None:        
            self.labeled_dataset = labelled_data.copy(deep=True)
    
            if self.dataset_type == 'training' and "num_points" in self.ds_config:
                print('Number of points: ', self.ds_config["num_points"])
                #import pdb; pdb.set_trace()
                # chosen_samples = self.labeled_dataset[self.labeled_dataset.index < 2500].sample(n=int(self.ds_config["num_points"]/2), random_state=42, replace=False)
                # chosen_samples = pd.concat([chosen_samples, self.labeled_dataset[self.labeled_dataset.index > 4000].sample(n=int(self.ds_config["num_points"]/2), random_state=42, replace=False)])
                # self.labeled_dataset = chosen_samples
                self.labeled_dataset = self.labeled_dataset.sample(n=self.ds_config["num_points"], random_state=42, replace=False)
            
            else: 
                self.unused_points = pd.DataFrame()    
            
            self.labeled_dataset = self.labeled_dataset.sample(frac=self.ds_config["labeled_sample_size"], random_state=42, replace=False)
            # if 'beta_to_zero' in self.ds_config and self.ds_config["beta_to_zero"]:
            #     self.labeled_dataset['beta'] = 0
            return self.labeled_dataset
        else: return pd.DataFrame()
    
    def prepare_inputs_and_targets(self):
        '''The thickness target values for the unlabelled part of the dataset are set to nan, as we don't have any measurements for them. 
        Then unlabelled and labelled inputs/targets are concatenated.
        If the dataset spatial coordinates are in lonlat crs, they are transformed to the epsg projection for the arctic (EPSG25833).'''

        if not self.unlabeled_dataset.empty:
            self.unlabeled_target = self.unlabeled_dataset[[self.ds_config["target"][1]]].copy(deep=True) # get the target of unlabelled dataset, here is is only mass balance (index=1 in config.yaml) as thickness is not known 
            self.unlabeled_target[self.ds_config["target"][0]] = torch.nan # a new column for the thickness is filled up with nans in unlablelled dataset
            self.unlabeled_target = self.unlabeled_target[self.ds_config["target"]] # get the right order of the target which should be [thickness, mass balance] (same as for labelled dataset)
            self.unlabeled_inputs = self.unlabeled_dataset[self.ds_config["input_features"]].copy(deep=True)
        else:
            self.unlabeled_target = pd.DataFrame()
            self.unlabeled_inputs = pd.DataFrame()
        self.labeled_target = self.labeled_dataset[self.ds_config["target"]].copy(deep=True) #get labelled target
        self.labeled_inputs = self.labeled_dataset[self.ds_config["input_features"]].copy(deep=True) #get labelled inputs

        ## put labeled and unlabeled inputs/targets together
        self.target = pd.concat([self.unlabeled_target, self.labeled_target]) # concat labelled and unlablled target
        self.inputs = pd.concat([self.unlabeled_inputs, self.labeled_inputs]) #concatenate labelled and unlabelled inputs

        return self.inputs, self.target


    def _apply_scaler(self):
        '''Apply the scaler to the inputs and targets of the dataset.'''
        if self.transform is not None:
            self.transformed_inputs = self.transform(self.inputs.to_numpy())
        else: 
            self.transformed_inputs = self.inputs.to_numpy()
            print('No Standardization applied to inputs!')
        if self.target_transform is not None:
            self.transformed_targets = self.target_transform(self.target.to_numpy())
        else:
            self.transformed_targets = self.target.to_numpy()
            print('No Standardization applied to targets!')

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input = torch.from_numpy(self.transformed_inputs[idx]).to(torch.float32)
        target = torch.from_numpy(self.transformed_targets[idx]).to(torch.float32)

        return input, target, idx
    




# Datamodule

class PL_GlacierDataset(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for handling glacier datasets.

    Args:
        config (dict): Configuration parameters for the dataset.

    Attributes:
        train_size (float): Proportion of data to use for training.
        test_size (float): Proportion of data to use for testing.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses to use for data loading.
        config (dict): Configuration parameters for the dataset.
        scaler (StandardScaler): Scaler for input features.
        target_scaler (StandardScaler): Scaler for target values.
        labeled_data (pd.DataFrame): Labeled data.
        unlabeled_data (pd.DataFrame): Unlabeled data.
        train_data (pd.DataFrame): Training data.
        val_data (pd.DataFrame): Validation data.
        train_dataset (GriddedDataset): Training dataset.
        val_dataset (GriddedDataset): Validation dataset.
        dataset (GriddedDataset): Dataset for prediction.

    Methods:
        setup(stage: str) -> None:
            Set up the data for a specific stage (fit, validate, or predict).
        prepare_data() -> None:
            Prepare the data by applying filters and dropping missing values.
        fit_scaler_to_traindata() -> None:
            Fit the scaler to the training dataset and set the transformers.
        fit_scaler_with_config(dataset) -> None:
            Fit the scaler to the dataset using the configuration parameters.
        train_dataloader() -> DataLoader:
            Return a DataLoader for the training dataset.
        val_dataloader() -> DataLoader:
            Return a DataLoader for the validation dataset.
        predict_dataloader() -> DataLoader:
            Return a DataLoader for the prediction dataset.
    """

    def __init__(self, config: dict) -> None: 
        super().__init__()
        
        self.train_size, self.test_size = config["ds"]["train_size"], config["ds"]["test_size"]
        self.batch_size = config["dataloader"]["batch_size"]
        self.num_workers = config["dataloader"]["num_workers"]
        self.ds_config = copy.deepcopy(config["ds"])
        if "save_dir" in config["experiment"]:
            self.save_dir = config["experiment"]["save_dir"]
        else: self.save_dir = None
        self.scaler = StandardScaler()
        self.target_scaler = StandardScaler()

        self.labeled_data = pd.concat([pd.read_csv(path, low_memory=False) for path in config["ds"]["data_dir_labeled"]])
        self.unlabeled_data = pd.concat([pd.read_csv(path, low_memory=False) for path in config["ds"]["data_dir_unlabeled"]])

        self.prepare_data()


    def setup(self, stage: str) -> None:
        """
        Set up the data for a specific stage (fit, validate, or predict).

        Args:
            stage (str): The stage of the training process.

        Raises:
            Warning: If no specific validation dataset is set.
        """
        if stage == 'fit':
            self.train_data, self.val_data = train_test_split(self.labeled_data, test_size=self.test_size, random_state=42)
            
            common_elements = self.train_data.merge(self.val_data, on=self.ds_config["input_features"])
            
            if common_elements.shape[0] > 0:
                print(common_elements.shape[0], ' common elements found! They will be removed from the validation set.') 
                self.val_data = self.val_data[~self.val_data['Unnamed: 0'].isin(common_elements['Unnamed: 0_y'])]
            
            if 'repeat_train_data' in self.ds_config:
                enhanced_train_data = pd.concat([self.train_data for _ in range(self.ds_config["repeat_train_data"])]) 
                self.train_dataset = GriddedDataset(self.ds_config, enhanced_train_data, self.unlabeled_data, dataset_type='training')
            else:
                self.train_dataset = GriddedDataset(self.ds_config, self.train_data, self.unlabeled_data, dataset_type='training')
            
            self.val_dataset = GriddedDataset(self.ds_config, self.val_data,dataset_type='validation')
            
            if self.save_dir:
                try:
                    
                    #shouldnt i be saving the indices of the dataframes instead of the dataframes themselves?
                    self.train_dataset.labeled_dataset.to_csv(os.path.join(self.save_dir,'labelled_train_data.csv'),mode='x')
                    self.val_dataset.labeled_dataset.to_csv(os.path.join(self.save_dir,'labelled_val_data.csv'),mode='x')
                except FileExistsError:
                    print('File already exists, so we are not saving the data again.')
            
            self.fit_scaler_to_traindata()
            self.train_dataset._apply_scaler()
            self.val_dataset._apply_scaler()
        
        if stage =='validate':
            if not hasattr(self, 'val_dataset'):
                warnings.warn('No specific validation dataset has been set, the whole dataset is used as validation.')
                self.val_dataset = GriddedDataset(self.ds_config, self.labeled_data, self.unlabeled_data)
            
            if self.val_dataset.transform is None:
                warnings.warn('Dataset will be scaled from config file!') 
                print('Get scaling from config file.')
                self.fit_scaler_with_config(self.val_dataset)
                self.val_dataset.transform = self.scaler.transform
                self.val_dataset.target_transform = self.target_scaler.transform
                
                self.val_dataset._apply_scaler()
            else:
                warnings.warn('Validation dataset has scaler already and will not be scaled from config file.')
                

        if stage == 'predict':
            predict_config = copy.deepcopy(self.ds_config)
            del predict_config['num_points']
            self.dataset = GriddedDataset(self.ds_config, self.labeled_data)
            
            if self.dataset.transform is None:
                warnings.warn('Dataset will be scaled from config file!')
                self.fit_scaler_with_config(self.dataset)
                self.dataset.transform = self.scaler.transform
                self.dataset.target_transform = self.target_scaler.transform
                self.dataset._apply_scaler()
            else:
                warnings.warn('Prediction dataset has scaler already and will not be scaled from config file.')
                

    def prepare_data(self) -> None:
        pass

    def fit_scaler_to_traindata(self) -> None:
        """
        Fit the scaler to the training dataset and set the transformers for inputs and targets in the dataset class.
        """
        self.scaler.fit(self.train_dataset.inputs.to_numpy()) 
        self.target_scaler.fit(self.train_dataset.target.to_numpy())
        self.train_dataset.transform = self.scaler.transform
        self.train_dataset.target_transform = self.target_scaler.transform
        self.val_dataset.transform = self.scaler.transform
        self.val_dataset.target_transform = self.target_scaler.transform

    def fit_scaler_with_config(self, dataset) -> None:
        """
        Fit the scaler to the dataset using the configuration parameters.

        Args:
            dataset (GriddedDataset): The dataset to fit the scaler to.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.target_scaler.fit(dataset.target.to_numpy())
            self.scaler.fit(dataset.inputs.to_numpy()) 
        
        try: 
            self.scaler.mean_ = self.ds_config["transformation_features_mean"]
            self.target_scaler.mean_ = self.ds_config["transformation_target_mean"]
            
            self.scaler.scale_ = self.ds_config["transformation_features_var"]
            self.target_scaler.scale_ = self.ds_config["transformation_target_var"]
            
        except KeyError:
            print('Could not find mean and variance for scaler in config file.')
        
        
        
    def train_dataloader(self):
        """
        Returns:
            DataLoader: The DataLoader for the training dataset.
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
  
    def val_dataloader(self):
        """
        Returns:
            DataLoader: The DataLoader for the validation dataset.
        """
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
    
    def predict_dataloader(self):
        """
        Returns:
            DataLoader: The DataLoader for the prediction dataset.
        """
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)





