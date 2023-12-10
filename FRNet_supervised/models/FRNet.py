import math
from math import sqrt

from models import FDNet

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from scipy import signal
import numpy as np
from layers.PatchTST_layers import series_decomp
from layers.RevIN import RevIN

from layers.PatchTST_backbone import PatchTST_backbone
from utils.dct_func import FFT_for_Period
from utils.losses import context_sampling, hierarchical_contrastive_loss
def FFT_for_Period(x, top_k=2):
    # [B, C, T]
    x = x.permute(0, 2, 1)
    xf = torch.fft.rfft(x, dim=-1)
    # find period by amplitudes
    frequency_list = abs(xf)
    peak_indexes = signal.argrelextrema(frequency_list.cpu().detach().numpy(), np.greater, axis=-1)
    zy = torch.zeros_like(frequency_list).cuda()
    zy[peak_indexes] = 1
    zy[...,0:10] = 0
    top_list = torch.topk(frequency_list * zy, top_k, dim=-1)[0]
    top_list_pos = torch.topk(frequency_list * zy, top_k, dim=-1)[1]    
    period = x.shape[-1] // top_list_pos
    return period, top_list

class TrendsBlock(nn.Module):
    def __init__(self, c_in, seq_len, pred_len):
        super(TrendsBlock, self).__init__()
        # RevIn
        self.revin_layer = RevIN(c_in, affine=True, subtract_last=False)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len_list = [24]
        self.stride_list = [12]
        self.emb = 32
        # #FITS
        # self.fixed_period_list = [12, 24, 48]
        # self.dominance_freq=int(self.seq_len // self.fixed_period_list[0] + 1) * 2 + 10 # configs.cut_freq # 720/24
        # self.length_ratio = (self.pred_len)/self.seq_len 
        # self.freq_Linear = nn.Linear(self.dominance_freq, int(self.dominance_freq*self.length_ratio)).to(torch.cfloat)
        self.Trend_pred = nn.ModuleList()
        self.Trend_linear_projection = nn.ModuleList()
        self.W_P = nn.ModuleList() 
        for k in range(len(self.patch_len_list)):
            self.period_input_len = int((seq_len - self.patch_len_list[k])/self.stride_list[k] + 1)
            self.period_output_len = int((pred_len - self.patch_len_list[k])/self.stride_list[k] + 1)
            # self.frequency_num = self.patch_len_list[k] // 2 + 1
            self.channel_indivial = False
            self.W_P.append(nn.Linear(self.patch_len_list[k], self.emb))
            

            self.frequency_num = self.emb //2 +1
            self.Trend_pred.append( nn.Sequential(Period_Frequency_Mixer_Predictor(self.period_input_len, self.period_input_len, self.frequency_num, self.channel_indivial), nn.Linear(self.period_input_len, self.period_output_len).to(torch.cfloat)))
            self.Trend_linear_projection.append(
                nn.Sequential(
                    nn.Linear(self.emb * self.period_output_len, self.pred_len)
                    )
                )
            
                                                
        
    def forward(self, x):
        B, T, N = x.size()
        # # norm
        x = self.revin_layer(x, 'norm')
        # # # FITS
        # low_specx = torch.fft.rfft(x, dim=1)
        # low_specx[:,self.dominance_freq:]=0 # LPF截断
        # low_specx = low_specx[:,0:self.dominance_freq,:] # LPF
        # # print(low_specx.permute(0,2,1).shape, self.freq_Linear)
        # low_specxy_ = self.freq_Linear(low_specx.permute(0,2,1)).permute(0,2,1)
        # low_specxy = torch.zeros([low_specxy_.size(0),int((self.pred_len)/2+1),low_specxy_.size(2)],dtype=low_specxy_.dtype).to(low_specxy_.device)

        # low_specxy[:,0:low_specxy_.size(1),:]=low_specxy_ # zero padding
        # low_xy=torch.fft.irfft(low_specxy, dim=1)
        # low_xy=low_xy * self.length_ratio # compemsate the length change
        
        x = x.permute(0,2,1)
        res = []
        for k in range(len(self.patch_len_list)):   
            x_k = x  
            x_k = x_k.unfold(dimension=-1, size=self.patch_len_list[k], step=self.stride_list[k])
            x_k = self.W_P[k](x_k)
            # print(x.shape)
            period_fft = torch.fft.rfft(x_k, dim=-1)    
            W_pos = nn.Parameter(torch.empty(period_fft.shape)).to(period_fft.device) 
            nn.init.uniform_(W_pos, -0.02, 0.02)
            period_fft = period_fft + W_pos
            # print(period_fft.shape)
            period_pred = self.Trend_pred[k](period_fft.permute(0,1,3,2)).permute(0,1,3,2)    
            period_pred = torch.fft.irfft(period_pred, dim=-1)
            period_pred = period_pred.reshape(B, N, -1)
            # print(period_pred.shape)
            res.append(self.Trend_linear_projection[k](period_pred).permute(0,2,1))
        res = torch.stack(res, dim=-1)
        res = torch.sum(res * (1 / len(self.patch_len_list)), -1)
        # p = 1
        # res = res *p + (1-p)*low_xy
        res = self.revin_layer(res, 'denorm')
        return res
        

class TimesBlock(nn.Module):
    def __init__(self, c_in, seq_len, pred_len):
        super(TimesBlock, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = c_in
        self.channel_indivial = False
        self.fixed_period_list = [24]
        self.period_fixed = True
        self.emb =64

        #FITS
        self.dominance_freq=int(self.seq_len // self.fixed_period_list[0] + 1) * 2 + 10 # configs.cut_freq # 720/24
        self.length_ratio = (self.pred_len)/self.seq_len 
        self.freq_Linear = nn.Linear(self.dominance_freq, int(self.dominance_freq*self.length_ratio)).to(torch.cfloat)
        #Complex Rotate
        self.Period_pred = nn.ModuleList()
        self.W_P = nn.ModuleList() 
        self.predic_head = nn.ModuleList() 

        for k in range(len(self.fixed_period_list)):
            self.W_P.append(nn.Linear(self.fixed_period_list[k], self.emb))
            
            self.period_input_len = math.floor(self.seq_len/self.fixed_period_list[k]) 
            self.period_output_len =  math.ceil(self.pred_len/self.fixed_period_list[k]) 
            self.frequency_num = self.emb //2 +1
            self.predic_head.append(nn.Linear(self.period_output_len*self.emb, self.pred_len))
            self.Period_pred.append(nn.Sequential(Period_Frequency_Mixer_Predictor(self.period_input_len, self.period_input_len, self.frequency_num, self.channel_indivial), nn.Linear(self.period_input_len, self.period_output_len).to(torch.cfloat)))            
        # adaptive aggregation
        self.aggregation = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        
        # RIN
        # x, var, mean  = self.RIN(x)
        # enc_out = self.revin_layer(x, 'norm')
        
        B, T, N = x.size()
        # print(x.shape)
        period_list = self.fixed_period_list   
        # #FITS
        low_specx = torch.fft.rfft(x, dim=1)
        low_specx[:,self.dominance_freq:]=0 # LPF截断
        low_specx = low_specx[:,0:self.dominance_freq,:] # LPF
        # print(low_specx.permute(0,2,1).shape, self.freq_Linear)
        low_specxy_ = self.freq_Linear(low_specx.permute(0,2,1)).permute(0,2,1)
        low_specxy = torch.zeros([low_specxy_.size(0),int((self.pred_len)/2+1),low_specxy_.size(2)],dtype=low_specxy_.dtype).to(low_specxy_.device)

        low_specxy[:,0:low_specxy_.size(1),:]=low_specxy_ # zero padding
        low_xy=torch.fft.irfft(low_specxy, dim=1)
        low_xy=low_xy * self.length_ratio # compemsate the length change


        res = [] #存放不同topk周期
        for k in range(len(period_list)):
            length = self.seq_len
            
            #由于每个序列周期不一样，所以需要分维度
            pred_k = torch.zeros([B, self.pred_len, N],dtype=x.dtype).to(x.device)

            period_i = self.fixed_period_list[k]
            # reshape [b, l , c]
            if length % period_i != 0:
                new_length = (length // period_i) * period_i
                x_k = x[:, -new_length:, :]
            else:
                new_length = length
                x_k = x
            x_k = x_k.permute(0,2,1) #[b, c, l] 
            x_k = x_k.unfold(dimension=-1, size=period_i, step=period_i) #[b,c,n,p]
            
            #这里必须映射到高维空间 不然表示能力有限
            x_k = self.W_P[k](x_k)
            

            period_fft = torch.fft.rfft(x_k, dim=-1)
            
            W_pos = nn.Parameter(torch.empty(period_fft.shape)).to(period_fft.device) 
            nn.init.uniform_(W_pos, -0.02, 0.02)
            period_fft = period_fft + W_pos
            
            period_pred = self.Period_pred[k](period_fft.permute(0,1,3,2)).permute(0,1,3,2)
            
            period_pred = torch.fft.irfft(period_pred, dim=-1)

            period_pred = period_pred.reshape(B, N, -1)  #[b, l, c]
            
            pred_k = self.predic_head[k]( period_pred).permute(0,2,1)
            pred_k = pred_k [:, :self.pred_len, :] #[b,c, pred_length]
            res.append(pred_k)
        # res.append(low_xy)
        res = torch.stack(res, dim=-1)
        res = torch.sum(res * (1 / len(self.fixed_period_list)), -1)
        # # adaptive aggregation
        # linear_res = self.aggregation(x.permute(0,2,1)).permute(0,2,1)
        p = 0.7
        res = res *p + (1-p)*low_xy
        # res=(res) * torch.sqrt(var) + mean
        return res
    
    def RIN(self, x):
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x = x - x_mean
        x_var=torch.var(x, dim=1, keepdim=True)+ 1e-5
        x = x / torch.sqrt(x_var)
        return x, x_var, x_mean 


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.c_in = configs.enc_in
        self.seasonal_model_layer = 0
        self.trend_model_layer = 0
        self.seasonal_model = nn.ModuleList([TimesBlock(c_in = self.c_in, seq_len = self.seq_len, pred_len = self.seq_len)
                                    for _ in range(self.seasonal_model_layer)] + [TimesBlock(c_in = self.c_in, seq_len = self.seq_len, pred_len = self.pred_len)])
        
        self.trend_model = nn.ModuleList([TrendsBlock(c_in = self.c_in, seq_len = self.seq_len, pred_len = self.seq_len)
                                    for _ in range(self.trend_model_layer)] + [TrendsBlock(c_in = self.c_in, seq_len = self.seq_len, pred_len = self.pred_len)])                         
        
        self.linear_projection =  nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.pred_len, self.pred_len)
        )
        self.decomposition = configs.decomposition
        kernel_size = configs.kernel_size
        self.decomp_module = series_decomp(kernel_size)
        self.revin_layer = RevIN(configs.enc_in, affine=configs.affine, subtract_last=configs.subtract_last)
        self.trend_revin_layer = RevIN(configs.enc_in, affine=configs.affine, subtract_last=configs.subtract_last)
        # # load parameters
        # c_in = configs.enc_in
        # context_window = configs.seq_len
        # target_window = configs.pred_len
        
        # n_layers = configs.e_layers
        # n_heads = configs.n_heads
        # d_model = configs.d_model
        # d_ff = configs.d_ff
        # dropout = configs.dropout
        # fc_dropout = configs.fc_dropout
        # head_dropout = configs.head_dropout
        
        # individual = configs.individual
    
        # patch_len = configs.patch_len
        # stride = configs.stride
        # padding_patch = configs.padding_patch
        # revin = configs.revin
        # affine = configs.affine
        # subtract_last = configs.subtract_last
        # kernel_size = configs.kernel_size
        # self.patchTST_trend_2 = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
        #                           max_seq_len=1024, n_layers=n_layers, d_model=d_model,
        #                           n_heads=n_heads, d_k=None, d_v=None, d_ff=d_ff, norm='BatchNorm', attn_dropout=0,
        #                           dropout=dropout, act="gelu", key_padding_mask='auto', padding_var=None, 
        #                           attn_mask=None, res_attention=True, pre_norm=False, store_attn=False,
        #                           pe='zeros', learn_pe=True, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
        #                           pretrain_head=False, head_type='flatten', individual=individual, revin=revin, affine=affine,
        #                           subtract_last=subtract_last, verbose=False)
        
        # self.FDNet_trend = FDNet.Model(configs = self.configs)
        
        
        # self.linear_trend_blocks =  nn.Sequential(
        #     nn.Linear(self.seq_len, self.seq_len),
        #     nn.Linear(self.seq_len, self.seq_len),
        #     nn.Linear(self.seq_len, self.pred_len)
        # )

    
    def forward(self, x):
        x = self.revin_layer(x, 'norm')
        # seasonal_init, trend_init = self.decomp_module(x)
        
          
        
        # # linear_trend
        # trend_init = self.trend_revin_layer(trend_init, 'norm')
        # trend = self.linear_trend_blocks(trend_init.permute(0,2,1)).permute(0,2,1)
        # trend = self.trend_revin_layer(trend, 'denorm')
        
        #patchTST_trend
        # trend = self.patchTST_trend_2(trend_init.permute(0,2,1)).permute(0,2,1)
        
        # FDNet_trend
        # trend = self.FDNet_trend(trend_init)
        
        # enc_out = trend_init
        # for i in range(self.trend_model_layer + 1):           
        #     enc_out = self.trend_model[i](enc_out) + enc_out if i<self.trend_model_layer else self.trend_model[i](enc_out)
        # trend = enc_out
        
        enc_out = x
        for i in range(self.seasonal_model_layer + 1):    
            enc_out = self.seasonal_model[i](enc_out) + enc_out if i<self.seasonal_model_layer else self.seasonal_model[i](enc_out)

        y = enc_out
        
            
        # y = self.linear_projection((seasonal + trend).permute(0,2,1)).permute(0,2,1)
        y = self.revin_layer(y, 'denorm')
        # y = trend
        return y
    
class Period_Frequency_Mixer_Predictor(nn.Module):
    def __init__(self, in_features, out_features, frequency_num, channel_indivial):
        super(Period_Frequency_Mixer_Predictor, self).__init__()
        self.linear_1 =  nn.Linear(in_features, out_features).to(torch.cfloat) 
        self.frequency_mix = nn.Linear(frequency_num, frequency_num).to(torch.cfloat) 
        self.channel_indivial = channel_indivial
        
    
    def forward(self, input):
        # print(input.shape, self.linear_1)
        output = self.linear_1(input) + input
        if self.channel_indivial:
            output = self.frequency_mix(output.permute(0, 2, 1)).permute(0, 2, 1) + output
        else:
            output = self.frequency_mix(output.permute(0, 1, 3, 2)).permute(0, 1, 3, 2) + output
        return  output

    
