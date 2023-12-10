import torch.nn as nn

from layers.ConvBlock import ConvBlock
from layers.FDNetEmbed import DataEmbedding
from layers.RevIN import RevIN

class Decomposed_block(nn.Module):
    def __init__(self, enc_in, d_model, seq_kernel, dropout, attn_nums, ICOM, h_nums, seq_len, pred_len, timebed):
        super(Decomposed_block, self).__init__()
        self.enc_in = enc_in - timebed
        self.pred_len = pred_len
        self.embed = DataEmbedding(d_model, dropout)
        self.ICOM = ICOM
        if self.ICOM:
            pro_conv2d = [ConvBlock(d_model, d_model, seq_kernel, ICOM, pool=True, dropout=dropout)
                          for _ in range(h_nums)]
        else:
            pro_conv2d = [ConvBlock(d_model, d_model, seq_kernel, ICOM, pool=False, dropout=dropout)
                          for _ in range(attn_nums)]
        self.pro_conv2d = nn.ModuleList(pro_conv2d)

        self.F = nn.Flatten(start_dim=2)
        final_len = seq_len // (2 ** h_nums) if ICOM else seq_len
        self.FC = nn.Linear(final_len * d_model, pred_len)

        self.timebed = timebed
        if self.timebed:
            self.time_layer = nn.Linear(self.timebed, self.enc_in * d_model)

    def forward(self, x):
        if self.timebed:
            time = x[:, :, -self.timebed:, :]
            x = x[:, :, :-self.timebed, :]
            x = self.embed(x)
            B, C, S, V = x.shape
            time = self.time_layer(time[:, :, :, 0]).contiguous().view(B, S, V, C).permute(0, 3, 1, 2)
            x = x + time
        else:
            x = self.embed(x)

        x_2d = x.clone()
        for conv2d in self.pro_conv2d:
            x_2d = conv2d(x_2d)
        x_2d_out = self.F(x_2d.transpose(1, -1))
        x_out = self.FC(x_2d_out).transpose(1, 2)
        return x_out


class Model(nn.Module):
    def __init__(self, configs,
                 seq_kernel=3, attn_nums=5, timebed='None',
                 d_model=8, pyramid=4, ICOM=False, dropout=0.1):
        super(Model, self).__init__()
        type_bed = {'None': 0, 'hour': 1, 'day': 1, 'year': 6, 'year_min': 7}
        timebed = int(type_bed[timebed])
        self.enc_in = configs.enc_in
        self.timebed = timebed
        self.pyramid = pyramid

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = d_model
        FDNet_blocks = [Decomposed_block(self.enc_in, d_model, seq_kernel, dropout, attn_nums, ICOM, 1, # 多个List相加实现多层模块提前子序列
                                         self.seq_len // (2 ** pyramid), self.pred_len, self.timebed)] + \
                       [Decomposed_block(self.enc_in, d_model, seq_kernel, dropout, attn_nums - i, ICOM, i + 1,
                                         self.seq_len // (2 ** (pyramid - i)), self.pred_len, self.timebed)
                        for i in range(pyramid + 1)]
        self.FDNet_blocks = nn.ModuleList(FDNet_blocks)
        self.rev = RevIN(configs.enc_in)

    def forward(self, enc_input):
        # enc_input = x_enc[:, :self.seq_len, :]
        # enc_input = self.rev(enc_input, 'norm')
        enc_input_list = [enc_input[:, -self.seq_len // (2 ** self.pyramid):, :]]

        enc_out = 0
        num_output = 0
        for i in range(self.pyramid):
            enc_input_list.append(enc_input[:, -self.seq_len // (2 ** (self.pyramid - i - 1)):
                                               -self.seq_len // (2 ** (self.pyramid - i)), :])
        for curr_input, FD_b in zip(enc_input_list, self.FDNet_blocks):
            # print(curr_input.shape)
            enc_out += FD_b(curr_input.unsqueeze(-1)) # 不同子序列预测结果加起来
            num_output += 1
            
        return enc_out / num_output
        # return self.rev(enc_out / num_output, 'denorm')
