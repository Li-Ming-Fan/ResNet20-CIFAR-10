# -*- coding: utf-8 -*-


import model_data_cifar_10 as model_data
import model_resnet_20 as model_wrap


#
# activation flag
#
model_flag = model_wrap.FLAG_RELU
#

#
model = model_wrap.ResNetModel()
#
if model_flag == model_wrap.FLAG_RELU:
    model.z_model_dir = './model_relu'
    model.z_model_name = 'model_relu'
elif model_flag == model_wrap.FLAG_SPLUS:
    model.z_model_dir = './model_splus'
    model.z_model_name = 'model_splus'
elif model_flag == model_wrap.FLAG_PRELU:
    model.z_model_dir = './model_prelu'
    model.z_model_name = 'model_prelu'
elif model_flag == model_wrap.FLAG_ELU:
    model.z_model_dir = './model_elu'
    model.z_model_name = 'model_elu'
else:
    print('error, not defined activation function, use relu instead')
    model.z_model_dir = './model_relu'
    model.z_model_name = 'model_relu'
#


#
# data
data_train = model_data.load_train_data()
data_valid = model_data.load_test_data()
#


#
# train
model.train_and_valid(data_train, data_valid, model_flag)
#



