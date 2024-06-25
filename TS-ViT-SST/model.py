
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import pdb


spa_rto = 0.8


class SpaceTimeTransformer(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        d_model = configs.d_model
        self.device = configs.device
        self.input_dim = configs.input_dim * configs.patch_size[0] * configs.patch_size[1]
        self.src_conv_emb = input_conv_embedding(configs.input_dim, d_model, configs.emb_spatial_size,
                                       configs.patch_size, configs.input_length, self.device)
        self.tgt_conv_emb = input_conv_embedding(configs.input_dim, d_model, configs.emb_spatial_size,
                                       configs.patch_size, configs.output_length, self.device)
        encoder_layer = EncoderLayer(d_model, configs.nheads, configs.dim_feedforward, configs.dropout)
        decoder_layer = DecoderLayer(d_model, configs.nheads, configs.dim_feedforward, configs.dropout)
        self.encoder = Encoder(encoder_layer, num_layers=configs.num_encoder_layers)
        self.decoder = Decoder(decoder_layer, num_layers=configs.num_decoder_layers)

        self.xdeconvs = nn.Sequential(
            nn.ConvTranspose2d(d_model, d_model//2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(d_model//2, d_model//4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(d_model//4, configs.input_dim, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, src, tgt, src_mask=None, memory_mask=None,
                train=True, ssr_ratio=0):

        memory = self.encode(src, src_mask)
        if train:
            with torch.no_grad():
                tgt_mask = self.generate_square_subsequent_mask(tgt.size(1))
                sic_pred = self.decode(torch.cat([src[:, -1:], tgt[:, :-1]], dim=1),
                                       memory, tgt_mask, memory_mask)  # (N, T_tgt, C, H, W)

            if ssr_ratio > 1e-6:
                teacher_forcing_mask = torch.bernoulli(ssr_ratio *
                        torch.ones(tgt.size(0), tgt.size(1) - 1, 1, 1, 1)).to(self.device)
            else:
                teacher_forcing_mask = 0
            tgt = teacher_forcing_mask * tgt[:, :-1] + (1 - teacher_forcing_mask) * sic_pred[:, :-1]
            tgt = torch.cat([src[:, -1:], tgt], dim=1)
            sic_pred = self.decode(tgt, memory, tgt_mask, memory_mask)
        else:
            if tgt is None:
                tgt = src[:, -1:]  # use last src as the input during test
            else:
                assert tgt.size(1) == 1
            for t in range(self.configs.output_length):
                tgt_mask = self.generate_square_subsequent_mask(tgt.size(1))
                sic_pred = self.decode(tgt, memory, tgt_mask, memory_mask)
                tgt = torch.cat([tgt, sic_pred[:, -1:]], dim=1)

        sic_pred = sic_pred[:, :, 0]

        return sic_pred

    def encode(self, src, src_mask):
        
        # conv2D embedding
        src = self.src_conv_emb(src) 

        memory = self.encoder(src, src_mask)  # (N, S, T_src, D)
        return memory

    def decode(self, tgt, memory, tgt_mask, memory_mask):

        B, T, C, H, W = tgt.shape

        # conv2D embedding
        tgt = self.tgt_conv_emb(tgt) 

        # decode 
        output = self.decoder(tgt, memory, tgt_mask, memory_mask)

        # conv output  
        output = output.permute(0, 2, 3, 1) 
        output = output.reshape(B*T, 256, 15, 45)
        output = self.xdeconvs(output)
        output = output.reshape(B, T, 1, H, W)
        
        return output

    def generate_square_subsequent_mask(self, sz: int):
    
        mask = (torch.triu(torch.ones(sz, sz)) == 0).T
        return mask.to(self.configs.device)


class Encoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])

        d_model = 256
        self.pruning_loc = [1]
        self.token_ratio = [spa_rto]
        predictor_list = [PredictorLG(d_model) for _ in range(len(self.pruning_loc))]
        self.score_predictor = nn.ModuleList(predictor_list)

    def forward(self, x, mask=None):
        B = x.shape[0] * x.shape[2]
        init_n = 15 * 45
        p_count = 0
        out_pred_prob = []
        prev_decision = torch.ones(B, init_n, 1, dtype=x.dtype, device=x.device)
        k = x
        for i, layer in enumerate(self.layers):
            if i in self.pruning_loc:
                spatial_x = k.permute(0, 2, 1, 3).reshape(B, -1, 256)
                pred_score = self.score_predictor[p_count](spatial_x, prev_decision).reshape(B, -1, 2)
                if self.training:
                    hard_keep_decision = F.gumbel_softmax(pred_score, hard=True)[:, :, 0:1] * prev_decision
                    out_pred_prob.append(hard_keep_decision.reshape(B, init_n))
                    policy = hard_keep_decision
                    x = layer(x, x, mask, policy=policy)
                    prev_decision = hard_keep_decision
                else:
                    score = pred_score[:, :, 0]
                    num_keep_node = int(init_n * self.token_ratio[p_count])
                    keep_policy = torch.argsort(score, dim=1, descending=True)[:, :num_keep_node]
                    now_policy = keep_policy + 1
                    k = batch_index_select(x, now_policy)
                    prev_decision = batch_index_select(prev_decision, keep_policy)
                    x = layer(x, k, mask)
                p_count += 1
            else:
                x = layer(x, x, mask)
        return x
        

class Decoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])

        d_model = 256
        self.pruning_loc = [1]
        self.token_ratio = [spa_rto]
        predictor_list = [PredictorLG(d_model) for _ in range(len(self.pruning_loc))]
        self.score_predictor = nn.ModuleList(predictor_list)

    def forward(self, x, memory, tgt_mask, memory_mask):
        B = x.shape[0] * x.shape[2]
        init_n = 15 * 45
        p_count = 0
        out_pred_prob = []
        prev_decision = torch.ones(B, init_n, 1, dtype=x.dtype, device=x.device)
        k = x
        for i, layer in enumerate(self.layers):
            if i in self.pruning_loc:
                spatial_x = k.permute(0, 2, 1, 3).reshape(B, -1, 256)
                pred_score = self.score_predictor[p_count](spatial_x, prev_decision).reshape(B, -1, 2)
                if self.training:
                    hard_keep_decision = F.gumbel_softmax(pred_score, hard=True)[:, :, 0:1] * prev_decision
                    out_pred_prob.append(hard_keep_decision.reshape(B, init_n))
                    policy = hard_keep_decision
                    x = layer(x, x, memory, tgt_mask, memory_mask, policy=policy)  # torch.Size([16, 128, 48, 256])
                    prev_decision = hard_keep_decision
                else:
                    score = pred_score[:, :, 0]
                    num_keep_node = int(init_n * self.token_ratio[p_count])
                    keep_policy = torch.argsort(score, dim=1, descending=True)[:, :num_keep_node]
                    now_policy = keep_policy + 1
                    k = batch_index_select(x, now_policy)
                    prev_decision = batch_index_select(prev_decision, keep_policy)
                    x = layer(x, k, memory, tgt_mask, memory_mask)
                p_count += 1
            else:
                x = layer(x, x, memory, tgt_mask, memory_mask)  # torch.Size([16, 128, 48, 256])
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, nheads, dim_feedforward, dropout):
        super().__init__()
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)
        self.encoder_attn = MultiHeadedAttention(d_model, nheads, SpaceAttention, dropout)
        self.time_attn = MultiHeadedAttention(d_model, nheads, TimeAttention, dropout)
        self.space_attn = MultiHeadedAttention(d_model, nheads, SpaceAttention, dropout)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
            )

    def divided_space_time_attn(self, query, key, value, mask, policy):

        m = self.space_attn(query, key, value, mask, policy) 
        return self.time_attn(m, m, m, mask)
        
    def forward(self, x, k, mask=None, policy=None):
        x = self.sublayer[0](x, lambda x: self.divided_space_time_attn(x, k, k, mask, policy))
        return self.sublayer[1](x, self.feed_forward)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, nheads, dim_feedforward, dropout):
        super().__init__()
        self.sublayer = clones(SublayerConnection(d_model, dropout), 3)
        self.encoder_attn = MultiHeadedAttention(d_model, nheads, TimeAttention, dropout)
        self.time_attn = MultiHeadedAttention(d_model, nheads, TimeAttention, dropout)
        self.space_attn = MultiHeadedAttention(d_model, nheads, SpaceAttention, dropout)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
            )

    def divided_space_time_attn(self, query, key, value, mask=None, policy=None):

        m = self.space_attn(query, key, value, mask, policy=None)
        return self.time_attn(m, m, m, mask)
    
    def forward(self, x, k, memory, tgt_mask, memory_mask, policy=None):
        x = self.sublayer[0](x, lambda x: self.divided_space_time_attn(x, k, k, tgt_mask, policy))
        x = self.sublayer[1](x, lambda x: self.encoder_attn(x, memory, memory, memory_mask))
        return self.sublayer[2](x, self.feed_forward)


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return self.norm(x + self.dropout(sublayer(x)))


class input_conv_embedding(nn.Module):
    def __init__(self, input_dim, d_model, emb_spatial_size, patch_size, max_len, device):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe_time = pe[None, None].to(device)  # (1, 1, T, D)
        self.spatial_pos = torch.arange(emb_spatial_size)[None, :, None].to(device)
        self.emb_space = nn.Embedding(emb_spatial_size, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model
        self.xconvs = nn.Sequential(
            nn.Conv2d(input_dim, d_model//4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model//4, d_model//2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model//2, d_model, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):

        
        assert len(x.size()) == 5 
        embedded_space = self.emb_space(self.spatial_pos)  # (1, S, 1, D)

        B, T, C, H, W = x.shape
        x = x.reshape(B*T, C, H, W)

        x = self.xconvs(x)
        x = x.reshape(B, T, self.d_model, -1).permute(0, 3, 1, 2)

        x = x + self.pe_time[:, :, :x.size(2)] + embedded_space  # (N, S, T, D)
        return self.norm(x)


def TimeAttention(query, key, value, mask=None, dropout=None):

    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)  # (N, h, S, T, T)
    if mask is not None:
        assert mask.dtype == torch.bool
        assert len(mask.size()) == 2
        scores = scores.masked_fill(mask[None, None, None], float("-inf"))
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    output = torch.matmul(p_attn, value) # (N, h, S, T, D)
    return output  


def SpaceAttention(query, key, value, mask=None, dropout=None, policy=None):

    d_k = query.size(-1)
    query = query.transpose(2, 3)  # (N, h, T, S, D)
    key = key.transpose(2, 3)  # (N, h, T, S, D)
    value = value.transpose(2, 3)  # (N, h, T, S, D)
    scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)  # (N, h, T, S, S)
    if policy is not None:
        p_attn = softmax_with_policy(scores, policy)
    else:
        p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    output = torch.matmul(p_attn, value).transpose(2, 3)  # (N, h, S, T_q, D)
    return output


class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model, nheads, attn, dropout):
        super().__init__()
        assert d_model % nheads == 0
        self.d_k = d_model // nheads
        self.nheads = nheads
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.dropout = nn.Dropout(p=dropout)
        self.attn = attn

    def forward(self, query, key, value, mask=None, policy=None):

        nbatches = query.size(0)
        nspace = query.size(1)
        ntime = query.size(2)

        query, key, value = \
            [l(x).view(x.size(0), x.size(1), x.size(2), self.nheads, self.d_k).permute(0, 3, 1, 2, 4)
             for l, x in zip(self.linears, (query, key, value))]

        if str(self.attn)[10:15] == 'Space':
            x = self.attn(query, key, value, mask=mask, dropout=self.dropout, policy=policy)
        else:
            x = self.attn(query, key, value, mask=mask, dropout=self.dropout)

        # (N, S, T, D)
        x = x.permute(0, 2, 3, 1, 4).contiguous() \
             .view(nbatches, nspace, ntime, self.nheads * self.d_k)
        return self.linears[-1](x)

class PredictorLG(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, embed_dim=384):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )

        self.out_conv = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 2),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x, policy):
        x = self.in_conv(x)
        B, N, C = x.size()
        local_x = x[:,:, :C//2]
        global_x = (x[:,:, C//2:] * policy).sum(dim=1, keepdim=True) / torch.sum(policy, dim=1, keepdim=True)
        x = torch.cat([local_x, global_x.expand(B, N, C//2)], dim=-1)
        return self.out_conv(x)


def softmax_with_policy(attn, policy, eps=1e-6):
    B, N, _ = policy.size()
    B, H, T, N, N = attn.size()
    attn = attn.permute(0,2,1,3,4).reshape(B*T, H, N, N)
    attn_policy = policy.reshape(B*T, 1, 1, N)  # * policy.reshape(B, 1, N, 1)
    eye = torch.eye(N, dtype=attn_policy.dtype, device=attn_policy.device).view(1, 1, N, N)
    attn_policy = attn_policy + (1.0 - attn_policy) * eye
    max_att = torch.max(attn, dim=-1, keepdim=True)[0]
    attn = attn - max_att

    # for stable training
    attn = attn.to(torch.float32).exp_() * attn_policy.to(torch.float32)
    attn = (attn + eps/N) / (attn.sum(dim=-1, keepdim=True) + eps)
    attn = attn.reshape(B, T, H, N, N).permute(0,2,1,3,4)
    return attn.type_as(max_att)


def batch_index_select(x, idx):
    if len(x.size()) == 4:
        B, N, T, C = x.size()
        x = x.permute(0,2,1,3).reshape(B*T*N, C)

        N_new = idx.size(1)
        offset = torch.arange(B*T, dtype=torch.long, device=x.device).view(B*T, 1) * N
        idx = (idx + offset -1).reshape(-1)
        out = x[idx]
        out = out.reshape(B, T, N_new, C).permute(0,2,1,3)
        return out
    elif len(x.size()) == 3:
        B, N, T = x.size()
        x = x.permute(0,2,1).reshape(B*T*N)
        N_new = idx.size(1)
        offset = torch.arange(B*T, dtype=torch.long, device=x.device).view(B*T, 1) * N
        idx = (idx + offset -1).reshape(-1)
        out = x[idx]
        out = out.reshape(B, T, N_new).permute(0,2,1)
        return out