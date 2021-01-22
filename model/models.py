import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
from torch.nn import Parameter
from torch import Tensor
import torch.nn.functional as F

import math


class GRUCell(nn.Module):
    def __init__(self, num_units, n_var, gate_type):
        super(GRUCell, self).__init__()

        self._num_units = num_units
        self._linear = None
        self._input_dim = None

        # added by mv_rnn
        self._n_var = n_var
        self.gate_type = gate_type
        self.gate_w_I2H = Parameter(
            torch.zeros([n_var, 1, 2*int(num_units / n_var), 1],dtype=torch.float32),
            requires_grad=True)
        self.gate_w_H2H = Parameter(
            torch.zeros([n_var, 1, 2*int(num_units / n_var), int(num_units / n_var)],dtype=torch.float32),
            requires_grad=True)
        self.gate_bias = Parameter(
            torch.ones(2*num_units,dtype=torch.float32),
            requires_grad=True)
        self.update_w_I2H = Parameter(
            torch.zeros([n_var, 1, int(num_units / n_var), 1], dtype=torch.float32),
            requires_grad=True)
        self.update_w_H2H = Parameter(
            torch.zeros([n_var, 1, int(num_units / n_var), int(num_units / n_var)], dtype=torch.float32),
            requires_grad=True)
        self.update_bias = Parameter(
            torch.ones(num_units, dtype=torch.float32),
            requires_grad=True)
        self.reset_parameters()


    def reset_parameters(self):
        std = 1.0 / math.sqrt(self._num_units)
        for w in self.parameters():
            w.data.uniform_(-std, std)


    def forward(self, inputs, hidden):
        if self.gate_type == 'tensor':
            # reshape input
            # [B H]
            blk_input = inputs.unsqueeze(2).permute(1, 0, 2)
            mv_input = blk_input.unsqueeze(2)
            # reshape hidden
            B = hidden.size()[0]
            blk_h = torch.unsqueeze(hidden, dim=2)
            blk_h2 = blk_h.view(B, -1, self._n_var)
            mv_h = blk_h2.permute(2, 0, 1).unsqueeze(2)

            gate_tmp_I2H = (mv_input * self.gate_w_I2H).sum(3)
            gate_tmp_H2H = (mv_h * self.gate_w_H2H).sum(3)
            # [V 1 B 2D]
            gate_tmp_x = torch.chunk(gate_tmp_I2H, self._n_var, dim=0)
            gate_tmp_h = torch.chunk(gate_tmp_H2H, self._n_var, dim=0)
            # [1 B 2H]
            g_res_x = torch.cat(gate_tmp_x, dim=2)
            g_res_h = torch.cat(gate_tmp_h, dim=2)

            gate_res_x = g_res_x.squeeze(0)
            gate_res_h = g_res_h.squeeze(0)
            # [B 2H;
            res_gate = gate_res_x + gate_res_h + self.gate_bias
            z, r = torch.chunk(res_gate, 2, dim=1)

            blk_r = torch.unsqueeze(F.sigmoid(r), dim=2)
            blk_r2 = blk_r.view(B, -1, self._n_var)
            mv_r = blk_r2.permute(2, 0, 1).unsqueeze(2)

            update_tmp_I2H = (mv_input * self.update_w_I2H).sum(3)
            update_tmp_H2H = ((mv_h * mv_r) * self.update_w_H2H).sum(3)

            update_tmp_x = torch.chunk(update_tmp_I2H, self._n_var, dim=0)
            update_tmp_h = torch.chunk(update_tmp_H2H, self._n_var, dim=0)

            u_res_x = torch.cat(update_tmp_x, dim=2)
            u_res_h = torch.cat(update_tmp_h, dim=2)

            update_res_x = u_res_x.squeeze(0)
            update_res_h = u_res_h.squeeze(0)

            g = update_res_x + update_res_h + self.update_bias
            new_h = F.sigmoid(z) * hidden + (1 - F.sigmoid(z)) * F.tanh(g)
        else:
            print('[ERROR]    mv-lstm cell type')
            new_h = 'nan'

        return new_h


class GRUModel(nn.Module):
    def __init__(self, hidden_dim, input_dim, layer_dim, gate_type, qz_para_dim):
        super(GRUModel,self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.gru_cell = GRUCell(hidden_dim, input_dim, gate_type)
        self.fc_q = nn.Linear(hidden_dim, hidden_dim * 2)
        #self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(x.size(0), self.hidden_dim).cuda(0))
        else:
            h0 = Variable(torch.zeros(x.size(0), self.hidden_dim))

        outs = []

        hn = h0
        # reverse time step for input series
        for seq in reversed(range(x.size(1))):
            hn = self.gru_cell(x[:, seq, :], hn)
            outs.append(hn)

        h_first = outs[-1].squeeze()
        qz_out = self.fc_q(h_first)
        # h_last = outs[-1].squeeze()
        #out = self.fc(h_last)
        return outs, h_first, qz_out


class ODEfunc(nn.Module):
    def __init__(self, hidden_size, bias=True):
        super().__init__()
        self.lin_hh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hz = nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hr = nn.Linear(hidden_size, hidden_size, bias=False)
        self.nfe = 0

    def forward(self, t, h):
        self.nfe += 1
        x = torch.zeros_like(h)
        r = torch.sigmoid(x + self.lin_hr(h))
        z = torch.sigmoid(x + self.lin_hz(h))
        u = torch.tanh(x + self.lin_hh(r * h))

        dh = (1 - z) * (u - h)
        return dh

class attention_mix(nn.Module):
    def __init__(self, hidden_dim, input_dim, seq_dim, n_class, per_vari_dim,
                 temp_attention_type, vari_attention_type):
        super(attention_mix, self).__init__()

        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.steps = seq_dim
        self.n_class = n_class
        self.per_vari_dim = per_vari_dim
        self.temp_type = temp_attention_type
        self.vari_type = vari_attention_type
        self.att_w_temp = Parameter(torch.zeros([input_dim, 1, 1, per_vari_dim], dtype=torch.float32), requires_grad=True)
        self.bias_temp = Parameter(torch.ones([input_dim, 1, 1], dtype=torch.float32), requires_grad=True)
        self.att_w_vari = Parameter(torch.zeros([2 * per_vari_dim, 1], dtype=torch.float32), requires_grad=True)
        self.bias_vari = Parameter(torch.ones(1, dtype=torch.float32), requires_grad=True)
        self.aug_w_vari = Parameter(torch.zeros([1, 1, 2 * per_vari_dim], dtype=torch.float32), requires_grad=True)
        self.att_w_mul = Parameter(torch.zeros([1, input_dim, 1], dtype=torch.float32), requires_grad=True)
        self.reset_parameters()


    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_dim)
        for w in self.parameters():
            w.data.uniform_(-std, std)


    def forward(self,att_inputs ):
        #att_h_tensor = torch.stack(att_inputs, dim=1)
        att_h_tensor = att_inputs
        att_h_list = torch.split(att_h_tensor, [self.per_vari_dim] * self.input_dim, 2)

        h_att_temp_input = att_h_list
        # temporal attention
        # [V B T D]
        tmph = torch.stack(h_att_temp_input, dim=0)
        # [V B T-1 D], [V, B, 1, D]
        tmph_before, tmph_last = torch.split(tmph, [self.steps - 1, 1], 2)
        # -- temporal logits
        if self.temp_type == 'temp_loc':
            # [V, B, T-1]
            temp_logit = torch.tanh((tmph_before * self.att_w_temp).sum(3) + self.bias_temp)
        else:
            print('\n [ERROR] temporal attention type \n')
            temp_logit = 'nan'

        temp_weight = torch.softmax(temp_logit, dim=-1)
        # temp_before [V B T-1 D], temp_weight [V B T-1]
        tmph_cxt = (tmph_before * temp_weight.unsqueeze(-1)).sum(2)
        tmph_last = tmph_last.squeeze(2)

        # [V B 2D]
        h_temp = torch.cat((tmph_last, tmph_cxt), 2)

        # variable attention
        if self.vari_type == 'vari_loc_all':
            vari_logits = torch.tanh(((h_temp * self.aug_w_vari).sum(2, keepdim=True) + self.bias_vari).permute(1, 0, 2))
            # vari_logits = torch.tanh(((h_temp * self.aug_w_vari).sum(2, keepdim=True)).permute(1, 0, 2))
            vari_weight = torch.softmax(vari_logits, dim=1)
            # [B V D]
            h_trans = h_temp.permute(1, 0, 2)
            # [B D]
            h_weighted = (h_trans * vari_weight).sum(1)

        elif self.vari_type == 'vari_mlp_all':
            # [B V 1] <- [V B 1] = [V B D]*[D 1]
            vari_logits = self.att_w_mul * F.tanh((torch.matmul(h_temp, self.att_w_vari) + self.bias_vari).permute(1, 0, 2))
            # [B V 1]
            vari_weight = torch.softmax(vari_logits, dim=1)
            # [B V D]
            h_trans = h_temp.permute(1, 0, 2)
            # [B D]
            h_weighted = (h_trans * vari_weight).sum(1)

        else:
            print('\n [ERROR] variable attention type \n')
            vari_weight = 'nan'

        return h_temp, temp_weight, vari_weight

class attention2_mix(nn.Module):
    def __init__(self, hidden_dim, input_dim, seq_dim, n_class, per_vari_dim,
                 temp_attention_type, vari_attention_type):
        super(attention2_mix, self).__init__()

        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.steps = seq_dim
        self.n_class = n_class
        self.per_vari_dim = per_vari_dim
        self.temp_type = temp_attention_type
        self.vari_type = vari_attention_type
        self.att_w_temp = Parameter(torch.zeros([input_dim, 1, 1, per_vari_dim], dtype=torch.float32), requires_grad=True)
        self.bias_temp = Parameter(torch.ones([input_dim, 1, 1], dtype=torch.float32), requires_grad=True)
        self.att_w_vari = Parameter(torch.zeros([per_vari_dim, 1], dtype=torch.float32), requires_grad=True)
        self.bias_vari = Parameter(torch.ones(1, dtype=torch.float32), requires_grad=True)
        self.aug_w_vari = Parameter(torch.zeros([1, 1, per_vari_dim], dtype=torch.float32), requires_grad=True)
        self.att_w_mul = Parameter(torch.zeros([1, input_dim, 1], dtype=torch.float32), requires_grad=True)
        self.fc = nn.Linear(input_dim, input_dim * 2)
        #self.fc = nn.Linear(input_dim, 20)
        #self.fc_mu = nn.Linear(input_dim,input_dim)
        #self.fc_var = nn.Linear(input_dim,input_dim)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_dim)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, att_inputs ):
        #att_h_tensor = torch.stack(att_inputs, dim=1)
        att_h_tensor = att_inputs
        att_h_list = torch.split(att_h_tensor, [self.per_vari_dim] * self.input_dim, 2)

        h_att_temp_input = att_h_list
        # temporal attention
        # [V B T D]
        tmph = torch.stack(h_att_temp_input, dim=0)
        # [V B T-1 D], [V, B, 1, D]
        #tmph_before, tmph_last = torch.split(tmph, [self.steps - 1, 1], 2)
        # -- temporal logits
        if self.temp_type == 'temp_loc':
            # [V, B, T-1]
            temp_logit = torch.tanh((tmph * self.att_w_temp).sum(3) + self.bias_temp)
        else:
            print('\n [ERROR] temporal attention type \n')
            temp_logit = 'nan'

        temp_weight = torch.softmax(temp_logit, dim=-1)
        # temp_before [V B T-1 D], temp_weight [V B T-1]
        tmph_cxt = (tmph * temp_weight.unsqueeze(-1)).sum(2)
        #tmph_last = tmph_last.squeeze(2)
        v_temp = (tmph * temp_weight.unsqueeze(-1)).sum(3).permute(1, 0, 2)

        # [V B 2D]
        #h_temp = torch.cat((tmph_last, tmph_cxt), 2)
        h_temp = tmph_cxt

        # variable attention
        if self.vari_type == 'vari_loc_all':
            vari_logits = torch.tanh(((h_temp * self.aug_w_vari).sum(2, keepdim=True) + self.bias_vari).permute(1, 0, 2))
            # vari_logits = torch.tanh(((h_temp * self.aug_w_vari).sum(2, keepdim=True)).permute(1, 0, 2))
            vari_weight = torch.softmax(vari_logits, dim=1)
            # [B V D]
            h_trans = h_temp.permute(1, 0, 2)
            # [B D]
            h_weighted = (h_trans * vari_weight).sum(1)

        elif self.vari_type == 'vari_mlp_all':
            # [B V 1] <- [V B 1] = [V B D]*[D 1]
            vari_logits = self.att_w_mul * F.tanh((torch.matmul(h_temp, self.att_w_vari) + self.bias_vari).permute(1, 0, 2))
            # [B V 1]
            vari_weight = torch.softmax(vari_logits, dim=1)
            # [B V D]
            h_trans = h_temp.permute(1, 0, 2)
            # [B D]
            h_weighted = (h_trans * vari_weight).sum(1)

        else:
            print('\n [ERROR] variable attention type \n')
            vari_weight = 'nan'

        c_t = (v_temp * vari_weight).sum(2)
        qz_out = self.fc(c_t)
        #ct_mean = (v_temp * vari_weight).mean(2)
        #mu = self.fc_mu(ct_mean)
        #var = self.fc_var(c_t)

        #return c_t, qz_out
        #return mu, var
        return c_t, qz_out, temp_weight, vari_weight


class mv_dense(nn.Module):
    def __init__(self, dim_per_vari, num_vari, dim_to, activation_type):
        super(mv_dense, self).__init__()
        self.num_vari = num_vari
        self.activation_type = activation_type
        # [V 1 D d]
        self.dense_w_ = Parameter(torch.zeros([num_vari, 1, 2*dim_per_vari, dim_to], dtype=torch.float32), requires_grad=True)
        # [V 1 1 d]
        self.dense_bias = Parameter(torch.ones([num_vari, 1, 1, dim_to], dtype=torch.float32), requires_grad=True)

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.num_vari)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, input):
        h_expand = input.unsqueeze(-1)
        # [V B D 1] * [V 1 D d] -> [V B d]
        tmp_h = (h_expand * self.dense_w_ + self.dense_bias).sum(2)
        if self.activation_type == "":
            dense_h_temp = tmp_h
        elif self.activation_type == "softplus":
            dense_h_temp = nn.softplus(tmp_h)
        else:
            print("\n [ERROR] activation in mv_dense \n")
            dense_h_temp = 'nan'

        return dense_h_temp


class multi_mv_dense(nn.Module):
    def __init__(self, num_layer, keep_prob, dim_vari, num_vari, activation_type):
        super(multi_mv_dense, self).__init__()
        self.in_dim_vari = dim_vari
        self.out_dim_vari = int(dim_vari / 2)
        self.num_layer = num_layer
        self.activation_type = activation_type
        self.drop = nn.Dropout(keep_prob)
        self.dense = mv_dense(self.in_dim_vari, num_vari, self.out_dim_vari, activation_type)

    def forward(self, input):
        for i in range(self.num_layers):
            h_mv_input = self.drop(input)
            # h_mv [V B d]
            h_mv_input = self.dense(h_mv_input)
            self.in_dim_vari = self.out_dim_vari
            self.out_dim_vari = int(self.out_dim_vari / 2)

        return h_mv_input, self.in_dim_vari


class Decoder(nn.Module):
    def __init__(self, latent_dim, fc_dim1, fc_dim2, target_dim=1):
        super(Decoder, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, fc_dim1)
        self.fc2 = nn.Linear(fc_dim1, fc_dim2)
        self.fc3 = nn.Linear(fc_dim2, target_dim)

    def forward(self, z):
        out = self.fc1(z)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

class Decoder2(nn.Module):
    def __init__(self, latent_dim, fc_dim, target_dim=1):
        super(Decoder2, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, fc_dim)
        self.fc2 = nn.Linear(fc_dim, target_dim)

    def forward(self, z):
        out = self.fc1(z)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class fcc(nn.Module):
    def __init__(self, z_hidden, target_dim=1, d_hidden=170):
        super(fcc, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(z_hidden, d_hidden)
        self.fc2 = nn.Linear(d_hidden, z_hidden)
        self.fc3 = nn.Linear(z_hidden, target_dim)

    def forward(self, z):
        out = self.fc1(z)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


class lstm_decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_lstm_layers, target_size, device):
        super(lstm_decoder, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=input_size, num_layers=num_lstm_layers, batch_first=True)
        #self.fc_in = nn.Linear(hidden_size, 1)
        #self.fc_hid = nn.Linear(input_size, hidden_size)
        self.fch = nn.Linear(hidden_size, input_size)
        #self.fc = nn.Linear(input_size, 1)
        self.target_size = target_size
        self.device = device
        self.out = nn.Linear(input_size, 1)
        self.hidden_size = hidden_size

    def forward(self, ct, z, h_in):
        #h = self.fc_hid(ct).unsqueeze(0)
        #c = self.fc_hid(ct).unsqueeze(0)
        h = self.fch(h_in).unsqueeze(0)
        c = self.fch(h_in).unsqueeze(0)
        z_in = z
        #decoder_output = z_in[:, 0, :].unsqueeze(1)
        decoder_output = ct.unsqueeze(1)
        outputs = torch.zeros([z.shape[0], self.target_size, 1]).to(self.device)
        for di in range(self.target_size):
            #decoder_input = (decoder_output + z_in[:, di, :].unsqueeze(1)) / 2
            decoder_input = decoder_output + z_in[:, di, :].unsqueeze(1)
            decoder_output, (h, c) = self.lstm(decoder_input, (h, c))
            decoder_output = self.out(decoder_output)
            outputs[:, di:di+1, :] = decoder_output
            #decoder_output = self.fc_in(decoder_output)
        #out = self.out(outputs)

        return outputs


class attention_step(nn.Module):
    def __init__(self, input_dim, hidden_dim, alphaHiddenDimSize, betaHiddenDimSize, per_vari_dim):
        super(attention_step, self).__init__()
        self.hidden_dim = hidden_dim
        self.alphahid_size = alphaHiddenDimSize
        self.betahid_size = betaHiddenDimSize
        self.input_dim = input_dim
        self.per_vari_dim = per_vari_dim

        self.gru_alpha = nn.GRU(self.hidden_dim, self.alphahid_size)
        self.gru_beta = nn.GRU(self.hidden_dim, self.betahid_size)
        self.alpha_att = nn.Linear(self.alphahid_size, 1)
        self.beta_att = nn.Linear(self.betahid_size, self.input_dim)
        #self.fc = nn.Linear(self.input_dim, per_vari_dim * 2)

    def forward(self, x):
        h_a = Variable(torch.zeros(1, x.shape[1], self.alphahid_size).cuda(0))
        h_b = Variable(torch.zeros(1, x.shape[1], self.betahid_size).cuda(0))

        rnn_alpha, _ = self.gru_alpha(x, h_a)
        rnn_beta, _ = self.gru_beta(x, h_b)
        alpha_att = self.alpha_att(rnn_alpha)
        alpha = F.softmax(alpha_att, dim=0)
        beta = torch.tanh(self.beta_att(rnn_beta))

        emb_ = torch.split(x, [self.per_vari_dim] * self.input_dim, 2)
        emb_v = torch.sum(torch.stack(emb_, dim=0), dim=-1)
        emb_v = emb_v.permute(1, 2, 0)

        var = torch.sum((alpha * beta * emb_v), dim=0)
        mean = torch.mean((alpha * beta * emb_v), dim=0)

        return mean, var


class attention3_mix(nn.Module):
    def __init__(self, hidden_dim, input_dim, seq_dim, n_class, per_vari_dim,
                 temp_attention_type, vari_attention_type):
        super(attention3_mix, self).__init__()

        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.steps = seq_dim
        self.n_class = n_class
        self.per_vari_dim = per_vari_dim
        self.temp_type = temp_attention_type
        self.vari_type = vari_attention_type
        self.att_w_temp = Parameter(torch.zeros([input_dim, 1, 1, per_vari_dim], dtype=torch.float32), requires_grad=True)
        self.bias_temp = Parameter(torch.ones([input_dim, 1, 1], dtype=torch.float32), requires_grad=True)
        self.att_w_vari = Parameter(torch.zeros([per_vari_dim, 1], dtype=torch.float32), requires_grad=True)
        self.bias_vari = Parameter(torch.ones(1, dtype=torch.float32), requires_grad=True)
        self.aug_w_vari = Parameter(torch.zeros([1, 1, per_vari_dim], dtype=torch.float32), requires_grad=True)
        self.att_w_mul = Parameter(torch.zeros([1, input_dim, 1], dtype=torch.float32), requires_grad=True)
        self.fc_mu = nn.Linear(per_vari_dim, 1)
        self.fc_sigma = nn.Linear(per_vari_dim, 1)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_dim)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, att_inputs ):
        #att_h_tensor = torch.stack(att_inputs, dim=1)
        att_h_tensor = att_inputs
        att_h_list = torch.split(att_h_tensor, [self.per_vari_dim] * self.input_dim, 2)

        h_att_temp_input = att_h_list
        # temporal attention
        # [V B T D]
        tmph = torch.stack(h_att_temp_input, dim=0)
        # [V B T-1 D], [V, B, 1, D]
        #tmph_before, tmph_last = torch.split(tmph, [self.steps - 1, 1], 2)
        # -- temporal logits
        if self.temp_type == 'temp_loc':
            # [V, B, T-1]
            temp_logit = torch.tanh((tmph * self.att_w_temp).sum(3) + self.bias_temp)
        else:
            print('\n [ERROR] temporal attention type \n')
            temp_logit = 'nan'

        temp_weight = torch.softmax(temp_logit, dim=-1)
        # temp_before [V B T-1 D], temp_weight [V B T-1]
        h_temp = (tmph * temp_weight.unsqueeze(-1)).sum(2)
        #tmph_last = tmph_last.squeeze(2)
        v_temp = (tmph * temp_weight.unsqueeze(-1)).sum(3).permute(1, 0, 2)

        # [V B 2D]
        #h_temp = torch.cat((tmph_last, tmph_cxt), 2)

        # variable attention
        if self.vari_type == 'vari_loc_all':
            vari_logits = torch.tanh(((h_temp * self.aug_w_vari).sum(2, keepdim=True) + self.bias_vari).permute(1, 0, 2))
            # vari_logits = torch.tanh(((h_temp * self.aug_w_vari).sum(2, keepdim=True)).permute(1, 0, 2))
            vari_weight = torch.softmax(vari_logits, dim=1)
            # [B V D]
            h_trans = h_temp.permute(1, 0, 2)
            # [B D]
            h_weighted = (h_trans * vari_weight).sum(1)

        elif self.vari_type == 'vari_mlp_all':
            # [B V 1] <- [V B 1] = [V B D]*[D 1]
            vari_logits = self.att_w_mul * F.tanh((torch.matmul(h_temp, self.att_w_vari) + self.bias_vari).permute(1, 0, 2))
            # [B V 1]
            vari_weight = torch.softmax(vari_logits, dim=1)
            # [B V D]
            h_trans = h_temp.permute(1, 0, 2)
            # [B D]
            h_weighted = (h_trans * vari_weight).sum(1)

        else:
            print('\n [ERROR] variable attention type \n')
            vari_weight = 'nan'

        mu = self.fc_mu(h_temp.permute(1, 0, 2))
        logvar = self.fc_sigma(h_temp.permute(1, 0, 2))
        #mu = (mu * vari_weight)
        #logvar = (logvar * vari_weight)
        #mu = (mu * vari_weight).sum(1)
        #logvar = (logvar * vari_weight).sum(1)
        return mu.squeeze(2), logvar.squeeze(2), vari_weight.permute(0, 2, 1)
        #return mu, logvar

class paralle_attention(nn.Module):
    def __init__(self, alpha_dim, latent_hid):
        super(paralle_attention, self).__init__()
        #self.hidden_dim = hidden_dim
        self.alphahid_size = alpha_dim
        #self.betahid_size = beta_dim
        self.input_dim = 1
        #self.per_vari_dim = per_vari_dim

        self.gru_alpha = nn.GRU(1, alpha_dim)

        #self.gru_alpha = GRUModel(self.alphahid_size, self.hidden_dim, 1, 'tensor', 10)
        #self.gru_beta = GRUModel(self.betahid_size, self.hidden_dim, 1, 'tensor', 10)
        self.alpha_att = nn.Linear(self.alphahid_size, latent_hid)
        #self.beta_att = nn.Linear(self.betahid_size, self.input_dim)
        #self.fc = nn.Linear(self.input_dim, per_vari_dim * 2)
        #self.mu_h = nn.Linear(5, qz_dim)
        #self.sigma_h = nn.Linear(5, qz_dim)

    def forward(self, x):
        h_a = Variable(torch.zeros(1, x.shape[0], self.alphahid_size).cuda(0))
        #h_b = Variable(torch.zeros(1, x.shape[1], self.betahid_size).cuda(0))
        x_trans = x.permute(1, 0, 2)

        rnn_alpha, hn = self.gru_alpha(x_trans, h_a)
        #rnn_beta, _ = self.gru_beta(x)

        alpha_att = self.alpha_att(rnn_alpha)
        alpha = F.softmax(alpha_att, dim=0).permute(1, 0, 2)

        # beta_list = torch.split(torch.stack(rnn_beta, dim=0), [self.per_vari_dim] * self.input_dim, dim=2)
        # beta_gi = torch.sum(torch.stack(beta_list, dim=0), dim=-1)
        # beta = torch.tanh(beta_gi.permute(1, 2, 0))

        # emb_ = torch.split(x, [self.per_vari_dim] * self.input_dim, 2)
        # emb_v = torch.sum(torch.stack(emb_, dim=0), dim=-1)
        # emb_v = emb_v.permute(2, 1, 0)

        c = torch.sum((alpha * x), dim=1)
        #mean = torch.mean((alpha * beta * emb_v), dim=1)
        #mu = self.mu_h(c)
        #sigma = self.sigma_h(c)

        return c



class without_att(nn.Module):
    def __init__(self, input_dim, num_per_vari):
        super(without_att, self).__init__()
        self.per_vari_dim = num_per_vari
        self.input_dim = input_dim
        self.fc = nn.Linear(input_dim, input_dim * 2)

    def forward(self, h):
        h_list = torch.split(h, [self.per_vari_dim] * self.input_dim, 2)
        none_h = torch.stack(h_list, dim=0)
        h_d = torch.mean(none_h,3)
        h_t = torch.mean(h_d,2)
        h_out = h_t.transpose(0, 1)
        z_out = self.fc(h_out)
        return z_out


class LSTMCell(nn.Module):
    def __init__(self, num_units, n_var, gate_type):
        super(LSTMCell, self).__init__()

        self._num_units = num_units
        self._linear = None
        self._input_dim = None

        # added by mv_rnn
        self._n_var = n_var
        self.gate_type = gate_type
        self.gate_w_I2H = Parameter(
            torch.zeros([n_var, 1, 4*int(num_units / n_var), 1],dtype=torch.float32),
            requires_grad=True)
        self.gate_w_H2H = Parameter(
            torch.zeros([n_var, 1, 4*int(num_units / n_var), int(num_units / n_var)],dtype=torch.float32),
            requires_grad=True)
        self.gate_bias = Parameter(
            torch.ones(4*num_units,dtype=torch.float32),
            requires_grad=True)
        # self.update_w_I2H = Parameter(
        #     torch.zeros([n_var, 1, int(num_units / n_var), 1], dtype=torch.float32),
        #     requires_grad=True)
        # self.update_w_H2H = Parameter(
        #     torch.zeros([n_var, 1, int(num_units / n_var), int(num_units / n_var)], dtype=torch.float32),
        #     requires_grad=True)
        # self.update_bias = Parameter(
        #     torch.ones(num_units, dtype=torch.float32),
        #     requires_grad=True)
        self.reset_parameters()


    def reset_parameters(self):
        std = 1.0 / math.sqrt(self._num_units)
        for w in self.parameters():
            w.data.uniform_(-std, std)


    def forward(self, inputs, h, c):
        if self.gate_type == 'tensor':
            # reshape input
            # [B H]
            blk_input = inputs.unsqueeze(2).permute(1, 0, 2)
            mv_input = blk_input.unsqueeze(2)
            # reshape hidden
            B = h.size()[0]
            blk_h = torch.unsqueeze(h, dim=2)
            blk_h2 = blk_h.view(B, -1, self._n_var)
            mv_h = blk_h2.permute(2, 0, 1).unsqueeze(2)

            # --- gates of input, output, forget
            gate_tmp_I2H = (mv_input * self.gate_w_I2H).sum(3)
            gate_tmp_H2H = (mv_h * self.gate_w_H2H).sum(3)
            # [V 1 B 4D]
            gate_tmp_x = torch.chunk(gate_tmp_I2H, self._n_var, dim=0)
            gate_tmp_h = torch.chunk(gate_tmp_H2H, self._n_var, dim=0)
            # [1 B 4H]
            g_res_x = torch.cat(gate_tmp_x, dim=2)
            g_res_h = torch.cat(gate_tmp_h, dim=2)

            gate_res_x = g_res_x.squeeze(0)
            gate_res_h = g_res_h.squeeze(0)
            # [B 4H]
            res_gate = gate_res_x + gate_res_h + self.gate_bias
            i, f, o, g = torch.chunk(res_gate, 4, dim=1)

            new_c = (c * F.sigmoid(f)) + F.sigmoid(i) * F.tanh(g)
            new_h = F.tanh(new_c) * F.sigmoid(o)

            #new_state = (new_c, new_h)

        else:
            print('[ERROR]    mv-lstm cell type')
            new_h = 'nan'
            new_c = 'nan'
            #new_state = 'nan'

        return new_h, new_c


class LSTMModel(nn.Module):
    def __init__(self, hidden_dim, input_dim, layer_dim, gate_type, qz_para_dim):
        super(LSTMModel,self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm_cell = LSTMCell(hidden_dim, input_dim, gate_type)
        #self.fc_q = nn.Linear(hidden_dim, hidden_dim * 2)
        #self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        device = x.device
        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(x.size(0), self.hidden_dim).to(device))
            c0 = Variable(torch.zeros(x.size(0), self.hidden_dim).to(device))
        else:
            h0 = Variable(torch.zeros(x.size(0), self.hidden_dim))
            c0 = Variable(torch.zeros(x.size(0), self.hidden_dim))

        outs = []

        hn = h0
        cn = c0
        # reverse time step for input series
        for seq in reversed(range(x.size(1))):
            hn, cn = self.lstm_cell(x[:, seq, :], hn, cn)
            outs.append(hn)

        #h_first = outs[-1].squeeze()
        #qz_out = self.fc_q(h_first)
        # h_last = outs[-1].squeeze()
        #out = self.fc(h_last)
        return outs


class paralle_attention2(nn.Module):
    def __init__(self, input_dim, alpha_dim, beta_dim, ode_dim):
        super(paralle_attention2, self).__init__()
        #self.hidden_dim = hidden_dim
        self.alphahid_size = alpha_dim
        self.betahid_size = beta_dim
        self.input_dim = input_dim
        self.emb_dim = input_dim
        #self.per_vari_dim = per_vari_dim

        # self.gru_alpha = GRUModel(self.alphahid_size, self.hidden_dim, 1, 'tensor', 10)
        # self.gru_beta = GRUModel(self.betahid_size, self.hidden_dim, 1, 'tensor', 10)
        # self.alpha_att = nn.Linear(self.alphahid_size, 1)
        self.embedding = nn.Linear(self.input_dim, self.emb_dim)
        self.gru_alpha = nn.LSTMCell(self.emb_dim,self.alphahid_size)
        self.gru_beta = nn.LSTMCell(self.emb_dim, self.betahid_size)
        self.alpha_att = nn.Linear(self.alphahid_size, 1)
        self.beta_att = nn.Linear(self.betahid_size, self.emb_dim)
        self.fc_mu = nn.Linear(self.emb_dim, ode_dim)
        self.fc_sigma = nn.Linear(self.emb_dim, ode_dim)
        #self.fc = nn.Linear(self.input_dim, per_vari_dim * 2)

    def forward(self, x):
        #h_a = Variable(torch.zeros(1, x.shape[1], self.alphahid_size).cuda(0))
        #h_b = Variable(torch.zeros(1, x.shape[1], self.betahid_size).cuda(0))

        emb = self.embedding(x)
        emb_v = emb.permute(1, 0, 2)
        reverse_emb = torch.flip(emb_v, [0])
        step = reverse_emb.shape[0]
        alpha_hx = Variable(torch.randn(x.size(0), self.alphahid_size).cuda(0))
        alpha_cx = Variable(torch.randn(x.size(0), self.alphahid_size).cuda(0))
        alpha_out = []
        for i in range(step):
            alpha_hx, alpha_cx = self.gru_alpha(reverse_emb[i], (alpha_hx, alpha_cx))
            alpha_out.append(alpha_hx)

        hi = torch.flip(torch.stack(alpha_out, dim=0), [0])
        alpha_att = self.alpha_att(hi.permute(1, 0, 2))
        alpha = torch.softmax(alpha_att, dim=1)

        beta_hx = Variable(torch.randn(x.size(0), self.betahid_size).cuda(0))
        beta_cx = Variable(torch.randn(x.size(0), self.betahid_size).cuda(0))
        beta_out = []
        for i in range(step):
            beta_hx, beta_cx = self.gru_beta(reverse_emb[i], (beta_hx, beta_cx))
            beta_out.append(alpha_hx)

        gi = torch.flip(torch.stack(beta_out, dim=0), [0])
        beta_att = self.beta_att(gi.permute(1, 0, 2))
        beta = torch.tanh(beta_att)

        var = torch.sum((alpha * beta * emb), dim=1)
        mean = torch.mean((alpha * beta * emb), dim=1)
        mu = self.fc_mu(mean)
        sigma = self.fc_sigma(var)

        return mu, sigma

class attention_driving(nn.Module):
    def __init__(self, hidden_dim, input_dim, seq_dim, n_class, per_vari_dim,
                 temp_attention_type, vari_attention_type, latent_hid):
        super(attention_driving, self).__init__()

        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.steps = seq_dim
        self.n_class = n_class
        self.per_vari_dim = per_vari_dim
        self.temp_type = temp_attention_type
        self.vari_type = vari_attention_type
        self.att_w_temp = Parameter(torch.zeros([input_dim, 1, 1, per_vari_dim], dtype=torch.float32), requires_grad=True)
        self.bias_temp = Parameter(torch.ones([input_dim, 1, 1], dtype=torch.float32), requires_grad=True)
        self.att_w_vari = Parameter(torch.zeros([per_vari_dim, 1], dtype=torch.float32), requires_grad=True)
        self.bias_vari = Parameter(torch.ones(1, dtype=torch.float32), requires_grad=True)
        self.aug_w_vari = Parameter(torch.zeros([1, 1, per_vari_dim], dtype=torch.float32), requires_grad=True)
        self.att_w_mul = Parameter(torch.zeros([1, input_dim, 1], dtype=torch.float32), requires_grad=True)
        self.hx = nn.Linear(input_dim, latent_hid)
        #self.fc = nn.Linear(input_dim, input_dim * 2)
        #self.fc = nn.Linear(input_dim, qz_dim * 2)
        #self.fc_mu = nn.Linear(input_dim,input_dim)
        #self.fc_var = nn.Linear(input_dim,input_dim)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_dim)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, att_inputs ):
        #att_h_tensor = torch.stack(att_inputs, dim=1)
        att_h_tensor = att_inputs
        att_h_list = torch.split(att_h_tensor, [self.per_vari_dim] * self.input_dim, 2)

        h_att_temp_input = att_h_list
        # temporal attention
        # [V B T D]
        tmph = torch.stack(h_att_temp_input, dim=0)
        # [V B T-1 D], [V, B, 1, D]
        #tmph_before, tmph_last = torch.split(tmph, [self.steps - 1, 1], 2)
        # -- temporal logits
        if self.temp_type == 'temp_loc':
            # [V, B, T-1]
            temp_logit = torch.tanh((tmph * self.att_w_temp).sum(3) + self.bias_temp)
        else:
            print('\n [ERROR] temporal attention type \n')
            temp_logit = 'nan'

        temp_weight = torch.softmax(temp_logit, dim=-1)
        # temp_before [V B T-1 D], temp_weight [V B T-1]
        tmph_cxt = (tmph * temp_weight.unsqueeze(-1)).sum(2)
        #tmph_last = tmph_last.squeeze(2)
        v_temp = (tmph * temp_weight.unsqueeze(-1)).sum(3).permute(1, 0, 2)

        # [V B 2D]
        #h_temp = torch.cat((tmph_last, tmph_cxt), 2)
        h_temp = tmph_cxt

        # variable attention
        if self.vari_type == 'vari_loc_all':
            vari_logits = torch.tanh(((h_temp * self.aug_w_vari).sum(2, keepdim=True) + self.bias_vari).permute(1, 0, 2))
            # vari_logits = torch.tanh(((h_temp * self.aug_w_vari).sum(2, keepdim=True)).permute(1, 0, 2))
            vari_weight = torch.softmax(vari_logits, dim=1)
            # [B V D]
            h_trans = h_temp.permute(1, 0, 2)
            # [B D]
            h_weighted = (h_trans * vari_weight).sum(1)

        elif self.vari_type == 'vari_mlp_all':
            # [B V 1] <- [V B 1] = [V B D]*[D 1]
            vari_logits = self.att_w_mul * F.tanh((torch.matmul(h_temp, self.att_w_vari) + self.bias_vari).permute(1, 0, 2))
            # [B V 1]
            vari_weight = torch.softmax(vari_logits, dim=1)
            # [B V D]
            h_trans = h_temp.permute(1, 0, 2)
            # [B D]
            h_weighted = (h_trans * vari_weight).sum(1)

        else:
            print('\n [ERROR] variable attention type \n')
            vari_weight = 'nan'

        c_t = (v_temp * vari_weight).sum(2)
        h_x = self.hx(c_t)
        #qz_out = self.fc(c_t)
        #ct_mean = (v_temp * vari_weight).mean(2)
        #mu = self.fc_mu(ct_mean)
        #var = self.fc_var(c_t)

        return h_x
        #return mu, var
        #return c_t, qz_out, temp_weight, vari_weight


class con_GRUCell(nn.Module):
    def __init__(self, num_units, n_var, gate_type):
        super(con_GRUCell, self).__init__()

        self._num_units = num_units
        self._linear = None
        self._input_dim = None

        # added by mv_rnn
        self._n_var = n_var
        self.gate_type = gate_type
        self.gate_w_I2H = Parameter(
            torch.zeros([n_var, 1, 2*int(num_units / n_var), 1],dtype=torch.float32),
            requires_grad=True)
        self.gate_w_H2H = Parameter(
            torch.zeros([n_var, 1, 2*int(num_units / n_var), int(num_units / n_var)],dtype=torch.float32),
            requires_grad=True)
        self.y_w_I2H = Parameter(
            torch.zeros([1, 1, 2*int(num_units / n_var), 1], dtype=torch.float32),
            requires_grad=True)
        self.gate_bias = Parameter(
            torch.ones(2*num_units,dtype=torch.float32),
            requires_grad=True)
        self.update_w_I2H = Parameter(
            torch.zeros([n_var, 1, int(num_units / n_var), 1], dtype=torch.float32),
            requires_grad=True)
        self.update_w_H2H = Parameter(
            torch.zeros([n_var, 1, int(num_units / n_var), int(num_units / n_var)], dtype=torch.float32),
            requires_grad=True)
        self.y_w_H2H = Parameter(
            torch.zeros([1, 1, int(num_units / n_var), 1], dtype=torch.float32),
            requires_grad=True)
        self.update_bias = Parameter(
            torch.ones(num_units, dtype=torch.float32),
            requires_grad=True)
        self.reset_parameters()


    def reset_parameters(self):
        std = 1.0 / math.sqrt(self._num_units)
        for w in self.parameters():
            w.data.uniform_(-std, std)


    def forward(self, inputs, y, hidden):
        if self.gate_type == 'tensor':
            # reshape input
            # [B H]
            blk_input = inputs.unsqueeze(2).permute(1, 0, 2)
            blk_y = y.unsqueeze(2).permute(1, 0, 2)
            mv_input = blk_input.unsqueeze(2)
            mv_y = blk_y.unsqueeze(2)
            # reshape hidden
            B = hidden.size()[0]
            blk_h = torch.unsqueeze(hidden, dim=2)
            blk_h2 = blk_h.view(B, -1, self._n_var)
            mv_h = blk_h2.permute(2, 0, 1).unsqueeze(2)

            gate_tmp_I2H = (mv_input * self.gate_w_I2H).sum(3)
            gate_y_I2H = (mv_y * self.y_w_I2H).sum(3)
            gate_tmp_H2H = (mv_h * self.gate_w_H2H).sum(3)
            # [V 1 B 2D]
            gate_tmp_x = torch.chunk(gate_tmp_I2H, self._n_var, dim=0)
            gate_tmp_h = torch.chunk(gate_tmp_H2H, self._n_var, dim=0)
            # [1 B 2H]
            g_res_x = torch.cat(gate_tmp_x, dim=2)
            g_res_h = torch.cat(gate_tmp_h, dim=2)

            gate_res_x = g_res_x.squeeze(0)
            gate_res_y = gate_y_I2H.squeeze(0).repeat(1, self._n_var)
            gate_res_h = g_res_h.squeeze(0)
            # [B 2H;
            res_gate = gate_res_x + gate_res_h + gate_res_y + self.gate_bias
            z, r = torch.chunk(res_gate, 2, dim=1)

            blk_r = torch.unsqueeze(F.sigmoid(r), dim=2)
            blk_r2 = blk_r.view(B, -1, self._n_var)
            mv_r = blk_r2.permute(2, 0, 1).unsqueeze(2)

            update_tmp_I2H = (mv_input * self.update_w_I2H).sum(3)
            update_y_I2H = (mv_y * self.y_w_H2H).sum(3)
            update_tmp_H2H = ((mv_h * mv_r) * self.update_w_H2H).sum(3)

            update_tmp_x = torch.chunk(update_tmp_I2H, self._n_var, dim=0)
            update_tmp_h = torch.chunk(update_tmp_H2H, self._n_var, dim=0)

            u_res_x = torch.cat(update_tmp_x, dim=2)
            u_res_h = torch.cat(update_tmp_h, dim=2)

            update_res_x = u_res_x.squeeze(0)
            update_res_y = update_y_I2H.squeeze(0).repeat(1, self._n_var)
            update_res_h = u_res_h.squeeze(0)

            g = update_res_x + update_res_h + update_res_y + self.update_bias
            new_h = F.sigmoid(z) * hidden + (1 - F.sigmoid(z)) * F.tanh(g)
        else:
            print('[ERROR]    mv-lstm cell type')
            new_h = 'nan'

        return new_h


class con_GRUModel(nn.Module):
    def __init__(self, hidden_dim, input_dim, layer_dim, gate_type, qz_para_dim):
        super(con_GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.gru_cell = con_GRUCell(hidden_dim, input_dim, gate_type)
        self.fc_q = nn.Linear(hidden_dim, hidden_dim * 2)
        #self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, y):
        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(x.size(0), self.hidden_dim).cuda(0))
        else:
            h0 = Variable(torch.zeros(x.size(0), self.hidden_dim))

        outs = []

        hn = h0
        # reverse time step for input series
        for seq in reversed(range(x.size(1))):
            hn = self.gru_cell(x[:, seq, :], y[:, seq, :], hn)
            outs.append(hn)

        h_first = outs[-1].squeeze()
        qz_out = self.fc_q(h_first)
        # h_last = outs[-1].squeeze()
        #out = self.fc(h_last)
        return outs, h_first, qz_out


class samp(nn.Module):
    def __init__(self, latent_hid, qz_dim, bias=True):
        super(samp, self).__init__()
        self.sam = nn.Linear(latent_hid, qz_dim * 2)

    def forward(self, hx, hy):
        qzpara = self.sam(hx * hy)

        return qzpara