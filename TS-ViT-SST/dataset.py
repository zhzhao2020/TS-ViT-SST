import numpy as np
from torch.utils.data import Dataset
import xarray as xr
from pathlib import Path
import random
import torch
from config import configs
import pdb
import numpy as np
import h5py
import os

    
def prepare_data(ds_dir, train_type):  

    data = xr.open_dataset(ds_dir).transpose('time', 'lat_index', 'lon_index')
    tiw_data = data.pp_sst.values   
    tiw_data = tiw_data - 273
    tiw_data = (tiw_data - 24) / 8.0
    tiw_data = tiw_data[:, :, :, None]
    
    assert len(tiw_data.shape) == 4
    assert not np.any(np.isnan(tiw_data))

    data.close()
    return tiw_data
    

def prepare_inputs_targets(len_time, input_gap, input_length, pred_shift, pred_length, samples_gap):

    assert pred_shift >= pred_length
    input_span = input_gap * (input_length - 1) + 1 
    pred_gap = pred_shift // pred_length 
    input_ind = np.arange(0, input_span, input_gap)
    target_ind = np.arange(0, pred_shift, pred_gap) + input_span + pred_gap - 1
    ind = np.concatenate([input_ind, target_ind]).reshape(1, input_length + pred_length)
    max_n_sample = len_time - (input_span+pred_shift-1)
    ind = ind + np.arange(max_n_sample)[:, np.newaxis] @ np.ones((1, input_length+pred_length), dtype=int)
    idx_inputs = ind[::samples_gap, :input_length]
    idx_targets = ind[::samples_gap, input_length:]
    return idx_inputs, idx_targets
    
    
def cat_over_last_dim(data):
    return np.concatenate(np.moveaxis(data, -1, 0), axis=0)


class tiw_dataset(Dataset):
    def __init__(self, tiw_sst, samples_gap):
        super().__init__()

        input_tiw = []
        target_tiw = []
       
        assert len(tiw_sst.shape) == 4
        idx_input_tiw, idx_target_tiw = prepare_inputs_targets(tiw_sst.shape[0], input_gap=configs.input_gap, input_length=configs.input_length, 
                                                pred_shift=configs.output_length, pred_length=configs.pred_shift, samples_gap=samples_gap)
                                                                   
        input_tiw.append(cat_over_last_dim(tiw_sst[idx_input_tiw]))   
        target_tiw.append(cat_over_last_dim(tiw_sst[idx_target_tiw]))  

        self.input_tiw = np.concatenate(input_tiw, axis=0)[:, :, None]  
        self.target_tiw = np.concatenate(target_tiw, axis=0)  

        assert self.input_tiw.shape[0] == self.target_tiw.shape[0]
        assert self.input_tiw.shape[1] == configs.input_length
        assert self.target_tiw.shape[1] == configs.pred_shift

    def GetDataShape(self):
        return {'tiw input': self.input_tiw.shape, 
                'tiw target': self.target_tiw.shape}

    def __len__(self,):
        return self.input_tiw.shape[0]

    def __getitem__(self, idx):
        return self.input_tiw[idx], self.target_tiw[idx]
       
