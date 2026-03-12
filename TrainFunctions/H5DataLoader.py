import h5py
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch

class H5DataLoader:
    def __init__(self,dataset_name,h5_file_path):
        self.dataset_name = dataset_name
        self.h5_file_path = h5_file_path
        self._load_data_to_np_arrays()
                
    def get_signals(self):
        return self.signals
    def get_coarse_masks(self):
        return self.general_masks_tensor
    
    def get_fiducial_states(self):
        return self.fiducials_states
    
    def get_fiducial_masks(self):
        return self.fiducials
    def get_post_input(self):
        return self.in_concat
    def get_all_data(self):
        return self.signals, self.general_masks, self.fiducials
    
    def _load_data_to_np_arrays(self):
        """
        Function that takes a h5 file and appends to list. Then converted to
        numpy arrays that should be used in network

        Returns
        -------
        None.

        """
        sigs = []
        masks = []
        self.subject_names = []
        
        
        with h5py.File(self.h5_file_path, 'r') as h5f:
            for subject_name, subject_group in h5f.items():
                #print(f"Reading data for subject: {subject_name}")
                self.subject_names.append(subject_name)
                # Iterate over all datasets in the subject group
                for dataset_name, dataset in subject_group.items():
                    #print(f"  Dataset: {dataset_name} | Shape: {dataset.shape} | Data type: {dataset.dtype}")
                    if dataset_name == 'input':
                        sigs.append(dataset[:])
                    elif dataset_name == 'masks':
                        masks.append(dataset[:])
        signals = np.stack(sigs)
        self.signals = signals.reshape(signals.shape[0],1,signals.shape[1])
        self.masks = np.stack(masks).transpose(0,2,1)
    
    def prepare_unet_tensor_dataset(self,batch_size,shuffle=True):
        self.input_data_tensor = torch.tensor(self.signals,dtype=torch.float32)
        self.masks_tensor = torch.tensor(self.masks,dtype=torch.float32)
        self.tensor_dataset_unet = TensorDataset(self.input_data_tensor,self.masks_tensor)
        self.tensor_data_loader_unet = DataLoader(self.tensor_dataset_unet,batch_size = batch_size,shuffle=True)