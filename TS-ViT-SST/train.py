from model import SpaceTimeTransformer
import torch
import torch.nn as nn
from config import configs
from torch.utils.data import DataLoader
import pickle
from dataset import *
import math
import os
import pdb
import os.path as osp


class GDL(nn.Module):
    def __init__(self, alpha=1, temporal_weight=None):

        super().__init__()
        self.alpha = alpha
        self.temporal_weight = temporal_weight

    def __call__(self, gt, pred):

        gt_shape = gt.shape
        if len(gt_shape) == 5:
            B, T, _, _, _ = gt.shape
        elif len(gt_shape) == 6:  
            B, T, TP, _, _, _ = gt.shape
        gt = gt.flatten(0, -4)
        pred = pred.flatten(0, -4)

        gt_i1 = gt[:, :, 1:, :]
        gt_i2 = gt[:, :, :-1, :]
        gt_j1 = gt[:, :, :, :-1]
        gt_j2 = gt[:, :, :, 1:]

        pred_i1 = pred[:, :, 1:, :]
        pred_i2 = pred[:, :, :-1, :]
        pred_j1 = pred[:, :, :, :-1]
        pred_j2 = pred[:, :, :, 1:]

        term1 = torch.abs(gt_i1 - gt_i2)
        term2 = torch.abs(pred_i1 - pred_i2)
        term3 = torch.abs(gt_j1 - gt_j2)
        term4 = torch.abs(pred_j1 - pred_j2)

        if self.alpha != 1:
            gdl1 = torch.pow(torch.abs(term1 - term2), self.alpha)
            gdl2 = torch.pow(torch.abs(term3 - term4), self.alpha)
        else:
            gdl1 = torch.abs(term1 - term2)
            gdl2 = torch.abs(term3 - term4)

        if self.temporal_weight is not None:
            assert self.temporal_weight.shape[0] == T, "Mismatch between temporal_weight and predicted sequence length"
            w = self.temporal_weight.to(gdl1.device)
            _, C, H, W = gdl1.shape
            _, C2, H2, W2 = gdl2.shap
            if len(gt_shape) == 5:
                gdl1 = gdl1.reshape(B, T, C, H, W)
                gdl2 = gdl2.reshape(B, T, C2, H2, W2)
                gdl1 = gdl1 * w[None, :, None, None, None]
                gdl2 = gdl2 * w[None, :, None, None, None]
            elif len(gt_shape) == 6:
                gdl1 = gdl1.reshape(B, T, TP, C, H, W)
                gdl2 = gdl2.reshape(B, T, TP, C2, H2, W2)
                gdl1 = gdl1 * w[None, :, None, None, None, None]
                gdl2 = gdl2 * w[None, :, None, None, None, None]

        gdl1 = gdl1.mean()
        gdl2 = gdl2.mean()
        gdl_loss = gdl1 + gdl2

        return gdl_loss

class NoamOpt:

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))


class Trainer:
    def __init__(self, configs):
        self.configs = configs
        self.device = configs.device
        torch.manual_seed(5)
        self.network = SpaceTimeTransformer(configs).to(configs.device)
        adam = torch.optim.Adam(self.network.parameters(), lr=0, weight_decay=configs.weight_decay)
        factor = math.sqrt(configs.d_model*configs.warmup)*0.0014
        self.opt = NoamOpt(configs.d_model, factor, warmup=configs.warmup, optimizer=adam)

    def loss_tiw(self, y_pred, y_true):
        rmse = torch.mean((y_pred - y_true)**2, dim=[2, 3])
        rmse = torch.sum(rmse.sqrt().mean(dim=0))
        mae = torch.mean(torch.abs(y_pred - y_true), dim=[2, 3])
        mae = torch.mean(mae.mean(dim=1))
        return rmse + mae

    def train_once(self, input_tiw, tiw_true, ssr_ratio):
        gdl_loss = GDL(alpha=1)
        tiw_pred = self.network(src=input_tiw.float().to(self.device),
                                tgt=tiw_true[:, :, None].float().to(self.device),
                                train=True, ssr_ratio=ssr_ratio)
        self.opt.optimizer.zero_grad()
        loss_tiw = self.loss_tiw(tiw_pred, tiw_true.float().to(self.device))
        T_GDL_loss = gdl_loss(tiw_true.float().to(self.device), tiw_pred)
        loss_tiw += T_GDL_loss * 0.2
        loss_tiw.backward()
        if configs.gradient_clipping:
            nn.utils.clip_grad_norm_(self.network.parameters(), configs.clipping_threshold)
        self.opt.step()
        return loss_tiw.item()

    def test(self, dataloader_test):
        tiw_pred = []
        with torch.no_grad():
            for input_tiw, tiw_true in dataloader_test:
                tiw = self.network(src=input_tiw.float().to(self.device), tgt=None, train=False)
                tiw_pred.append(tiw)
        return torch.cat(tiw_pred, dim=0)

    def infer(self, dataset, dataloader):
        self.network.eval()
        with torch.no_grad():
            tiw_pred = self.test(dataloader)
            tiw_true = torch.from_numpy(dataset.target_tiw).float().to(self.device)
            loss_tiw = self.loss_tiw(tiw_pred, tiw_true).item()
        return loss_tiw

    def train(self, dataset_train, dataset_test, chk_path):
        torch.manual_seed(0)
        print('loading train dataloader')
        dataloader_train = DataLoader(dataset_train, batch_size=self.configs.batch_size, shuffle=True)
        print('loading test dataloader')
        dataloader_test = DataLoader(dataset_test, batch_size=self.configs.batch_size_test, shuffle=False)

        count = 0
        best = 1000
        ssr_ratio = 1
        for i in range(self.configs.num_epochs):
            print('\nepoch: {0}'.format(i+1))
            # train
            self.network.train()
            for j, (input_tiw, tiw_true) in enumerate(dataloader_train):

                if ssr_ratio > 0:
                    ssr_ratio = max(ssr_ratio - self.configs.ssr_decay_rate, 0)
                loss_tiw = self.train_once(input_tiw, tiw_true, ssr_ratio)  # y_pred for one batch

                if j % self.configs.display_interval == 0:
                    print('batch training loss: {:.2f}, ssr: {:.5f}, lr: {:.5f}'.format(loss_tiw, ssr_ratio, self.opt.rate()))

            # evaluation
            eval_rmse = self.infer(dataset=dataset_test, dataloader=dataloader_test)
            log_str = 'Eval epoch: {:02d}, rmse: {:.4f}'.format(i+1, eval_rmse)
            print(log_str)
            out_file.write(log_str + "\n")
            out_file.flush()
            
            if eval_rmse >= best:
                count += 1
                print('eval score is not improved for {} epoch'.format(count))
            else:
                count = 0
                print('eval score is improved from {:.5f} to {:.5f}, saving model'.format(best, eval_rmse))
                self.save_model(chk_path)
                best = eval_rmse

            if count == self.configs.patience:
                print('early stopping reached, best score is {:5f}'.format(best))
                break


    def save_configs(self, config_path):
        with open(config_path, 'wb') as path:
            pickle.dump(self.configs, path)

    def save_model(self, path):
        torch.save({'net': self.network.state_dict(), 
                    'optimizer': self.opt.optimizer.state_dict()}, path)


if __name__ == '__main__':
    print(configs.__dict__)

    print('\nreading data')
    data_root = 'path_to_DATASET'
    train_data_name = 'name_to_train_data'
    test_data_name = 'name_to_test_data'
    
    train_tiw = prepare_data(data_root+train_data_name, 'train')
    test_tiw = prepare_data(data_root+test_data_name, 'test')

    print('processing training set')
    dataset_train = tiw_dataset(train_tiw, samples_gap=1)
    print(dataset_train.GetDataShape())
    del train_tiw
    
    print('processing test set')
    dataset_test = tiw_dataset(test_tiw, samples_gap=1)
    print(dataset_test.GetDataShape())
    del test_tiw
    
    if not os.path.exists(configs.save_path):
        os.mkdir(configs.save_path)
    
    out_file = open(osp.join(configs.save_path, "log.txt"), "w")
    out_file.write(str(dataset_train.GetDataShape()) + "\n")
    out_file.write(str(dataset_test.GetDataShape()) + "\n")
    out_file.flush()
    
    # pdb.set_trace()
    trainer = Trainer(configs)
    
    trainer.save_configs(configs.save_path + '/config_train.pkl')
    trainer.train(dataset_train, dataset_test, configs.save_path + '/checkpoint.chk')
