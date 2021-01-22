#!/usr/bin/python

# -- mv RNN --

# size of recurrent layers    
hidden_dim_dic = {}

hidden_dim_dic.update( {"energy": 125} )
hidden_dim_dic.update( {"nasdaq": 810} )
hidden_dim_dic.update( {"sml": 140} )
hidden_dim_dic.update( {"pm25": 60 } )
hidden_dim_dic.update( {"elec": 320} )

# learning rate increases as network size 
lr_dic = {}

lr_dic.update( {"energy": 0.01} )
lr_dic.update( {"sml": 0.01} )
lr_dic.update( {"nasdaq": 0.01} )
lr_dic.update( {"pm25": 0.01} )
lr_dic.update( {"elec": 0.001} )

# target col
target_col_dic = {}

target_col_dic.update( {"energy": 0} )
target_col_dic.update( {"nasdaq": 81} )
target_col_dic.update( {"sml": 13} )
target_col_dic.update( {"pm25": 0} )
target_col_dic.update( {"elec": 15} )

# indep col
indep_col_dic = {}

indep_col_dic.update( {"energy": [1,25]} )
indep_col_dic.update( {"nasdaq": [0,80]} )
indep_col_dic.update( {"sml": [0,12]} )
indep_col_dic.update( {"pm25": [1,6]} )
indep_col_dic.update( {"elec": [0,14]} )

# batch size 
batch_size_dic = {}
batch_size_dic.update( {"energy": 128} )
batch_size_dic.update( {"nasdaq": 128} )
batch_size_dic.update( {"sml": 128} )
batch_size_dic.update( {"pm25": 128} )
batch_size_dic.update( {"elec": 128} )

# noise_std
noise_std_dic = {}
noise_std_dic.update( {"energy": 0.5} )
noise_std_dic.update( {"nasdaq": 0.5} )
noise_std_dic.update( {"sml": 0.01} )
noise_std_dic.update( {"pm25": 0.5} )

#qz_dim_dic
qz_dim_dic = {}
qz_dim_dic.update( {"energy": 10} )
qz_dim_dic.update( {"nasdaq": 5} )
qz_dim_dic.update( {"sml": 10} )
qz_dim_dic.update( {"pm25": 10} )

#alpha_dim_dic
alpha_dim_dic = {}
alpha_dim_dic.update( {"energy": 10} )
alpha_dim_dic.update( {"nasdaq": 20} )
alpha_dim_dic.update( {"sml": 20} )
alpha_dim_dic.update( {"pm25": 20} )

#num_epoch
num_epoch_dic = {}
num_epoch_dic.update( {"energy": 50} )
num_epoch_dic.update( {"nasdaq": 100} )
num_epoch_dic.update( {"sml": 200} )
num_epoch_dic.update( {"pm25": 100} )

#if norm for target
norm_dic = {}
norm_dic.update( {"energy": False} )
norm_dic.update( {"nasdaq": False} )
norm_dic.update( {"sml": False} )
norm_dic.update( {"pm25": True} )

# max_norm contraints
maxnorm_dic = {}

maxnorm_dic.update( {"energy": 5.0} )
maxnorm_dic.update( {"sml": 5.0} )
maxnorm_dic.update( {"nasdaq": 5.0} )
maxnorm_dic.update( {"pm25": 5.0} )
maxnorm_dic.update( {"elec": 5.0} )

# attention type
attention_dic = {}

attention_dic.update( {"mv_full": "both-att"} )
attention_dic.update( {"mv_tensor": "both-att"} )

# loss type
loss_dic = {}

loss_dic.update( {"energy": "mse"} )
loss_dic.update( {"nasdaq": "mse"} )
loss_dic.update( {"sml": "mse"} )
loss_dic.update( {"pm25": "mse"} )
loss_dic.update( {"elec": "mse"} )