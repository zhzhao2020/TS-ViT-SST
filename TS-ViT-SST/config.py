import torch


class Configs:
    def __init__(self):
        pass


configs = Configs()

# trainer related
configs.n_cpu = 0
configs.device = torch.device('cuda:0')
configs.batch_size_test = 4
configs.batch_size = 4
configs.weight_decay = 0
configs.display_interval = 120
configs.num_epochs = 100
configs.early_stopping = True
configs.patience = 6
configs.gradient_clipping = False
configs.clipping_threshold = 1.

# lr warmup
configs.warmup = 2000

# data related
configs.input_dim = 1
configs.output_dim = 1

configs.input_length = 14
configs.output_length = 5

configs.input_gap = 5
configs.pred_shift = 5

# model
configs.d_model = 256
configs.patch_size = (8, 8)
configs.emb_spatial_size = 15*45
configs.nheads = 8
configs.dim_feedforward = 512
configs.dropout = 0.2
configs.num_encoder_layers = 2
configs.num_decoder_layers = 2

configs.ssr_decay_rate = 3.e-4
configs.save_path = './result/results'
