import torch
import numpy.random as npr
import os
import argparse
import logging
import time
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from model.config_hyper_para import *
from data_prepro import *
from model.models import *
import model.regular as regul
import math
import random
import matplotlib
matplotlib.use('agg')

parser = argparse.ArgumentParser()
parser.add_argument('--adjoint', '-a', type=eval, default=False)
# parser.add_argument('--dataset', '-d', help = "dataset", type = str)
parser.add_argument('--train_dir', type=str, default=None)
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

data_name = 'elec'
if 'energy' in data_name:
    data_path = 'raw_data/energydata_complete.csv'
elif 'nasdaq' in data_name:
    data_path = 'raw_data/nasdaq100_padding.csv'
elif 'pm25' in data_name:
    data_path = 'raw_data/pm25_rawdata.csv'
elif 'sml' in data_name:
    data_path = 'raw_data/sml_rawdata.csv'
elif 'elec' in data_name:
    data_path = 'raw_data/electricity_rawdata.csv'
else:
    print("wrong dataset name")

args.train_dir = 'model_elec3'
print('current datset: {}'.format(data_name))
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

target_col = target_col_dic[data_name]
indep_col = indep_col_dic[data_name]
win_size = 20
pre_T = 5
norm = True
alpha_loss = 1

train_share = 0.8
is_stateful = False
normalize_pattern = 2
generator = getGenerator(data_path)
datagen = generator(data_path, target_col, indep_col, win_size, pre_T,
                    train_share, is_stateful, normalize_pattern)

#train_input, test_input, train_target, test_target, y_mean, y_std = datagen.with_target()
#train_input, test_input, train_target, test_target, y_mean, y_std = datagen.getdata()
train_x, train_y, val_x, val_y, test_x, test_y, y_mean, y_std = datagen.data_extract_val()
ystd = torch.Tensor(y_std).to(device)
ymean = torch.Tensor(y_mean).to(device)

# seed = random.randint(0, 200)
# print(seed)
# torch.manual_seed(seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(seed)
seed = 150
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

#writer = SummaryWriter(comment=data_name)

print(" --- Data shapes: ", np.shape(train_x), np.shape(train_y), np.shape(val_x), np.shape(val_y), np.shape(test_x), np.shape(test_y))

batch_size = batch_size_dic[data_name]
dataset_train = subDataset(train_x, train_y)
dataset_val = subDataset(val_x, val_y)
dataset_test = subDataset(test_x, test_y)
train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)
val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True)
test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True)

def train(train_loader):
    loss_list = []
    loss_mse = []
    loss_kl = []
    loss_px = []
    vari_w = []
    for i, (inputs, target) in enumerate(train_loader):

        inputs = Variable(inputs.view(-1, step, input_dim).to(device))
        target = Variable(target.to(device))

        # # Reversed input in time step, modify in GRUModel
        # h, h_first, qz_para = encoder(inputs)
        # qz0_mean, qz0_logvar = qz_para[:, :hidden_dim], qz_para[:, hidden_dim:]
        # epsilon = torch.randn(qz0_mean.size()).to(device)
        # z0 = epsilon * torch.exp(0.5 * qz0_logvar) + qz0_mean
        #
        # # forward in time and solve ode for reconstructions
        # pred_z = odeint(func, z0, total_T).permute(1, 0, 2)
        # # attention
        # pre_att, temp_weight, vari_weight = att(pred_z)
        # h_mean = fc(pre_att)
        # h_mv_trans = h_mean.permute(1, 0, 2)
        # pred_y = (h_mv_trans * vari_weight).sum(1)

        # change oder of attention and ode part
        h, _, _ = encoder(inputs)
        att_h_input = torch.stack(h, dim=1)
        c_t, qz_para, _, _ = att(att_h_input)
        #qz0_mean, qz0_logvar = qz_para[:, :qz_dim], qz_para[:, qz_dim:]
        qz0_mean, qz0_logvar = qz_para[:, :input_dim], qz_para[:, input_dim:]
        epsilon = torch.randn(qz0_mean.size()).to(device)
        z0 = epsilon * torch.exp(0.5 * qz0_logvar) + qz0_mean
        # forward in time and solve ode for reconstructions
        #pred_z = odeint(func, z0, total_T).permute(1, 0, 2)
        pred_z = odeint(func, z0, total_T).permute(1, 0, 2)
        if norm:
            pred_ = fc(pred_z).squeeze(2)
            pred_y = pred_ * ystd + ymean
        else:
            pred_y = fc(pred_z).squeeze(2)

        # Calculate Loss
        noise_std_ = torch.zeros(pred_y.size()).to(device) + noise_std
        noise_logvar = 2. * torch.log(noise_std_).to(device)
        logpx = log_normal_pdf(target, pred_y, noise_logvar).sum(-1)
        pz0_mean = pz0_logvar = torch.zeros(z0.size()).to(device)
        analytic_kl = normal_kl(qz0_mean, qz0_logvar, pz0_mean, pz0_logvar).sum(-1)
        # loss1 = torch.mean(-logpx + analytic_kl, dim=0)
        l_px = torch.mean(-logpx, dim=0)
        l_kl = -torch.mean(analytic_kl, dim=0)
        loss2 = criterion(target, pred_y)
        # loss = loss1 + loss2
        loss = alpha_loss * (l_px + l_kl) + loss2

        # if weight_decay > 0:
        #     loss = loss1 + loss2 + reg_loss(encoder)
        # train_loss = loss.item()

        if torch.cuda.is_available():
            loss.to(device)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #loss_meter.update(loss.item())

        loss_px.append(l_px.item())
        loss_kl.append(l_kl.item())
        # loss_kl.append(loss1.item())
        loss_mse.append(loss2.item())
        loss_list.append(loss.item())
    #     vari_w.append(vari_weight)
    # vari_w_cate = torch.cat(vari_w, dim=0).squeeze(2)
    # vari_w_prior = vari_w_cate.sum(0)
    # sum_w_prior = vari_w_prior.sum()
    # ke_vari_prior = 1.0 * vari_w_prior / sum_w_prior
    train_mse = np.array(loss_mse).mean()
    kl_loss = np.array(loss_kl).mean()
    total_loss = np.array(loss_list).mean()
    px_loss = np.array(loss_px).mean()
    return px_loss, kl_loss, train_mse
    #return total_loss, ke_vari_prior

def eval(x, y):
    # Calculate Accuracy
    total = 0
    mae = []
    rmse = []
    # Iterate through test dataset
    with torch.no_grad():
        if torch.cuda.is_available():
            inputs = Variable(torch.Tensor(x).to(device))
            target = Variable(torch.Tensor(y).to(device))
        else:
            inputs = Variable(x)

        h, _, _ = encoder(inputs)
        att_h_input = torch.stack(h, dim=1)
        c_t, qz_para, _, _ = att(att_h_input)
        #qz0_mean, qz0_logvar = qz_para[:, :qz_dim], qz_para[:, qz_dim:]
        qz0_mean, qz0_logvar = qz_para[:, :input_dim], qz_para[:, input_dim:]
        epsilon = torch.randn(qz0_mean.size()).to(device)
        z0 = epsilon * torch.exp(0.5 * qz0_logvar) + qz0_mean
        # forward in time and solve ode for reconstructions
        pred_z = odeint(func, z0, total_T).permute(1, 0, 2)
        if norm:
            pred_ = fc(pred_z).squeeze(2)
            pred_y = pred_ * ystd + ymean
        else:
            pred_y = fc(pred_z).squeeze(2)

        test_mse = criterion(target, pred_y)
        test_rmse = torch.sqrt(test_mse)
        test_mae = criterionL1(target, pred_y)
        # mae.append(test_mae.item())
        # rmse.append(test_rmse.item())
        # mae_test = np.array(mae).mean()
        # rmse_test = np.array(rmse).mean()
        return test_mae, test_rmse

def log_test(text_env, errors):
    text_env.write("\n testing error: %s \n\n" % (errors))
    return

input_dim = train_x.shape[-1]
#qz_dim = input_dim * 10
pre_step = train_y.shape[-1]
step = train_x.shape[1]
target_size = train_y.shape[1]
#total_T = torch.linspace(1., 1., 1).to(device)
#total_T = torch.linspace(1., 2., 2).to(device)
total_T = torch.linspace(1., 5., 5).to(device)
#total_T = torch.linspace(1., 8., 8).to(device)
#total_T = torch.linspace(1., 10., 10).to(device)
num_epochs = 100
temp_attention_type = 'temp_loc'
vari_attention_type = 'vari_loc_all'
hidden_dim = hidden_dim_dic[data_name]
qz_para_dim = 10
layer_dim = 1  # ONLY CHANGE IS HERE FROM ONE LAYER TO TWO LAYER
gate_type = 'tensor'
loss_type = 'mse'
activation_type = ''
criterion = torch.nn.MSELoss()
criterionL1 = torch.nn.L1Loss()
learning_rate = lr_dic[data_name]
weight_decay = 0.01
noise_std = 0.1
per_vari_dim = int(hidden_dim / input_dim)
path = '/opt/data/private/torch_gru/model_elec/'
#path = '/home/penglei/odegru/best_model/'

best_log_dir = path + str(pre_T) + data_name + '_ode_att_model_5.pth'
log_err_file = path + str(pre_T) + data_name + '_test_35.txt'
# encoder = GRUModel(hidden_dim, input_dim, layer_dim, gate_type, qz_para_dim).to(device)
# func = ODEfunc(input_dim).to(device)
# # decoder = gru.Decoder(qz_para_dim, target_dim=1, d_hidden=170).to(device)
# att = attention2_mix(hidden_dim, input_dim, step, pre_step, per_vari_dim, temp_attention_type,
#                         vari_attention_type).to(device)
# # att = attention_mix(hidden_dim, input_dim, pre_step, pre_step, per_vari_dim, temp_attention_type,
# #                         vari_attention_type).to(device)
# #fc = mv_dense(per_vari_dim, input_dim, pre_step, activation_type).to(device)
# fc = nn.Linear(input_dim, 1).to(device)

# if weight_decay > 0:
#     reg_loss = regul.Regularization(encoder, weight_decay, p=2).to(device)
# else:
#     print("no regularization")

# params = (list(encoder.parameters()) + list(func.parameters()) + list(att.parameters()) + list(fc.parameters()))
# optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
# loss_meter = RunningAverageMeter()

# best_test_mae = float("inf")
# best_test_rmse = float("inf")
# iter = 0
# start_epoch = 0
w_vari = []

if args.train_dir is not None:
    if not os.path.exists(args.train_dir):
        os.makedirs(args.train_dir)
    # ckpt_path = os.path.join(args.train_dir, 'ckpt_5.pth')
    # if os.path.exists(ckpt_path):
    #     checkpoint = torch.load(ckpt_path)
    #     encoder.load_state_dict(checkpoint['encoder_state_dict'])
    #     func.load_state_dict(checkpoint['func_state_dict'])
    #     att.load_state_dict(checkpoint['att_state_dict'])
    #     fc.load_state_dict(checkpoint['fc_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     start_epoch = checkpoint['epoch']
    #     print('Loaded ckpt from epoc {}'.format(start_epoch))
    # else:
    #     start_epoch = 0
    #     print('no saved model, start from epoc 1')

for n in range(2):
    number_of_train = n
    print('train number in:', number_of_train)
    encoder = GRUModel(hidden_dim, input_dim, layer_dim, gate_type, qz_para_dim).to(device)
    func = ODEfunc(input_dim).to(device)
    att = attention2_mix(hidden_dim, input_dim, step, pre_step, per_vari_dim, temp_attention_type,
                         vari_attention_type).to(device)
    fc = nn.Linear(input_dim, 1).to(device)

    params = (list(encoder.parameters()) + list(func.parameters()) + list(att.parameters()) + list(fc.parameters()))
    optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
    #optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
    loss_meter = RunningAverageMeter()

    best_val_mae = float("inf")
    best_val_rmse = float("inf")
    test_mae = 0
    test_rmse = 0
    iter = 0
    start_epoch = 0
    for epoch in range(start_epoch + 1, num_epochs + 1):
        #train_loss, w_variable = train(train_loader)
        px_loss, kl_loss, mse_loss = train(train_loader)
        #w_vari.append(w_variable.unsqueeze(0))
        val_mae, val_rmse = eval(val_x, val_y)
        print('Epoch: {}. Loss: {}. Loss_kl: {}. Loss_mse: {}'.format(epoch, px_loss, kl_loss, mse_loss))
        if val_rmse < best_val_rmse:
            best_val_mae = val_mae
            best_val_rmse = val_rmse
            test_mae, test_rmse = eval(test_x, test_y)

            # state = {'encoder_state_dict':encoder.state_dict(),
            #          'func_state_dict':func.state_dict(),
            #          'att_state_dict':att.state_dict(),
            #          'fc_state_dict':fc.state_dict(),
            #          'optimizer_state_dict':optimizer.state_dict(),
            #          'epoch':epoch}
            # torch.save(state, best_log_dir)
            print('best_val_mae: {}. best_val_rmse: {}'.format(best_val_mae,best_val_rmse))
            print('test_mae: {}. test_rmse: {}'.format(test_mae, test_rmse))
        # writer.add_scalar('Eval_mae', test_mae, epoch)
        # writer.add_scalar('Eval_rmse', test_rmse, epoch)
        # writer.add_scalar('Train_loss', train_loss, epoch)
        #writer.add_scalar('Train_mse', mse_loss, epoch)
        #writer.add_scalar('KL_loss', kl_loss, epoch)

        #print('Epoch: {}. Loss: {}'.format(epoch, train_loss))
        print('Validation mae: {}. Validation rmse: {}'.format(val_mae, val_rmse))
        # iter += 1
        # if iter % 20 == 0:
        #     if args.train_dir is not None:
        #         ckpt_path = os.path.join(args.train_dir, 'ckpt_5.pth')
        #         torch.save({
        #             'encoder_state_dict': encoder.state_dict(),
        #             'func_state_dict': func.state_dict(),
        #             'att_state_dict': att.state_dict(),
        #             'fc_state_dict': fc.state_dict(),
        #             'optimizer_state_dict': optimizer.state_dict(),
        #             'epoch': iter
        #         }, ckpt_path)
        #         print('Stored ckpt at {}'.format(ckpt_path))

    print('best_val_mae: {}. best_val_rmse: {}'.format(best_val_mae, best_val_rmse))

    # save = torch.cat(w_vari, dim=0).cpu().detach().numpy()
    # #np.save(path + 'vari_weight.npy', save)
    # df = pd.DataFrame(save)
    # df.to_csv(path + 'vari.csv')
    # np_data = np.array(torch.cat(w_vari).cpu())
    # save = pd.DataFrame(np_data)
    # save.to_csv('w_weight.csv')
    with open(log_err_file, "a") as text_file:
        log_test(text_file, [best_val_rmse, best_val_mae, norm, alpha_loss, hidden_dim, learning_rate, noise_std])
        log_test(text_file, [test_rmse, test_mae, seed])