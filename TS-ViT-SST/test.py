import pickle
import numpy as np
from model import SpaceTimeTransformer
from torch.utils.data import DataLoader
import torch
import os
from dataset import *
import shutil
import pdb
from torch.utils.data import Dataset
import xarray as xr
from tqdm import tqdm

save_path = './result/results

def loss_tiw(y_pred, y_true):
    mse = torch.mean((y_pred - y_true)**2, dim=[2, 3])
    rmse = mse.sqrt()
    rmse_mean = rmse.mean(dim=0)
    return rmse, rmse_mean


def main():
    ### load the config
    with open(save_path+'/config_train.pkl', 'rb') as config_test:
        configs = pickle.load(config_test)

    configs.output_length = 5
    configs.pred_shift = 5
    configs.device = torch.device('cuda:0')
        
    model = SpaceTimeTransformer(configs).to(configs.device)
    model.load_state_dict(torch.load(save_path+'/checkpoint.chk')['net'])
    model.eval()
    
    data_root = 'path_to_DATASET'
    test_data_name = 'name_to_DATASET'

    test_tiw = prepare_data(data_root+test_data_name, 'test')
    dataset_test = tiw_dataset(test_tiw, samples_gap=1)
    
    dataloader_test = DataLoader(dataset_test, batch_size=configs.batch_size_test, shuffle=False)
    
    tiw_pred = []
    with torch.no_grad():
        for input_tiw, tiw_true in tqdm(dataloader_test):
            tiw = model(src=input_tiw.float().to(configs.device), tgt=None, train=False)
            tiw_pred.extend(tiw.tolist())
    
    tiw_true = torch.from_numpy(dataset_test.target_tiw).float()
    tiw_pred = torch.Tensor(tiw_pred)
    tiw_pred = tiw_pred * 8 + 24.0
    tiw_true = tiw_true * 8 + 24.0

    rmse, rmse_mean = loss_tiw(tiw_pred, tiw_true)

    print(rmse_mean)


if __name__ == '__main__':
    main()
    