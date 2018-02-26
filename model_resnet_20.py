# -*- coding: utf-8 -*-


import os
import time
import random

import numpy as np

import tensorflow as tf
from tensorflow.python.framework import graph_util

import zoo_layers as layers



#
# meta data
#
TRAINING_STEPS = 32000*4
BATCH_SIZE = 128
REG_LAMBDA = 0.0001
#
LEARNING_RATE_BASE = 0.001 # 1e-5
DECAY_RATE = 0.1
DECAY_STAIRCASE = True
DECAY_STEPS = 32000
MOMENTUM = 0.9
#
VALID_FREQ = 10
#
MODEL_DIR = './model_resnet'
MODEL_NAME = 'model_resnet'
#
# FLAG_DEBUG = 1
#


#
# activation functions
#
FLAG_RELU = 0
FLAG_SPLUS = 1
FLAG_PRELU = 3
FLAG_ELU = 4
#


def parametric_relu(x, name = None):
    alphas = tf.Variable(tf.zeros(x.get_shape()[-1], dtype = tf.float32), \
                         name = 'alpha')
    #
    return tf.nn.relu(x) + tf.multiply(alphas, (x - tf.abs(x))) * 0.5
    #
    '''
    weights = tf.Variable(tf.random_normal([784, 200], stddev=0.35), name=”weights”) 
    biases = tf.Variable(tf.zeros([200]), name=”biases”) 
    '''


def activation_layer(inputs, flag):
    #
    if flag == FLAG_RELU: outputs = tf.nn.relu(inputs)
    elif flag == FLAG_SPLUS: outputs = tf.nn.softplus(inputs)
    elif flag == FLAG_PRELU: outputs = parametric_relu(inputs)
    elif flag == FLAG_ELU: outputs = tf.nn.elu(inputs) 
    else:
        print('error, not defined activation function, use relu instead')
        outputs = tf.nn.relu(inputs)
    #
    return outputs
    #

#
# model
#
def resnet_model(inputs, flag_activation, training):
    #
    # [ batch, height, width, channel]
    #
    # [ 16, (3,3), (1,1),  'same', True, False, 'conv1']
    #
    
    #
    ch_n = 16
    #
    # params = [filters, kernel_size, strides, padding, batch_norm, relu, name]                             
    #
    
    item1 = [ ch_n, (3,3), (1,1), 'same', True, False, 'conv1']
    inputs = layers.conv_layer(inputs, item1, training)
    inputs = activation_layer(inputs, flag_activation)
    
    
    inputs = layers.block_resnet(inputs, ch_n, 1, False, training, 'res11')  # 32x32
    inputs = activation_layer(inputs, flag_activation)
    inputs = layers.block_resnet(inputs, ch_n, 1, False, training, 'res12')  # 32x32
    inputs = activation_layer(inputs, flag_activation)
    inputs = layers.block_resnet(inputs, ch_n, 1, False, training, 'res13')  # 32x32
    inputs = activation_layer(inputs, flag_activation)

    
    inputs = layers.block_resnet(inputs, 2*ch_n, 2, False, training, 'res21')  # 16x16
    inputs = activation_layer(inputs, flag_activation)
    inputs = layers.block_resnet(inputs, 2*ch_n, 1, False, training, 'res22')  # 16x16
    inputs = activation_layer(inputs, flag_activation)
    inputs = layers.block_resnet(inputs, 2*ch_n, 1, False, training, 'res23')  # 16x16
    inputs = activation_layer(inputs, flag_activation)
    
    
    inputs = layers.block_resnet(inputs, 4*ch_n, 2, False, training, 'res31')  # 8x8
    inputs = activation_layer(inputs, flag_activation)
    inputs = layers.block_resnet(inputs, 4*ch_n, 1, False, training, 'res32')  # 8x8
    inputs = activation_layer(inputs, flag_activation)
    inputs = layers.block_resnet(inputs, 4*ch_n, 1, False, training, 'res33')  # 8x8
    inputs = activation_layer(inputs, flag_activation)

    inputs = layers.averpool_layer(inputs, (8,8), (1,1), 'valid', name = 'last_pool')
    
    inputs = tf.squeeze(inputs, [1,2], name = 'features')
    
    
    #inputs = tf.reshape(inputs,(-1,2*2*ch_n*8))   #  32*32*3

    
    # fc
    weight_initializer = tf.contrib.layers.xavier_initializer()
    bias_initializer = tf.constant_initializer(value=0.0)
    #
    inputs = tf.layers.dense(inputs, 10, activation = tf.nn.sigmoid,
                             kernel_initializer = weight_initializer,
                             bias_initializer = bias_initializer,
                             name = 'fc_last')
    #
    # softmax
    outputs = tf.nn.softmax(inputs, name = 'softmax')
    #
    
    #
    return outputs
    #


#
# class
#
class ResNetModel():
    def __init__(self):
        #        
        self.z_sess_config = tf.ConfigProto()
        self.z_sess_config.gpu_options.per_process_gpu_memory_fraction = 0.99
        #
        self.z_valid_freq = VALID_FREQ
        self.z_valid_option = False
        #
        self.z_batch_size = BATCH_SIZE
        #
        self.z_model_dir = MODEL_DIR
        self.z_model_name = MODEL_NAME
        #
    
    def prepare_for_prediction(self, pb_file_path = None, portion = None):
        #
        if pb_file_path == None:
            pb_file_path = os.path.join(self.z_model_dir, self.z_model_name + '.pb')
        #
        if not os.path.exists(pb_file_path):
            print('ERROR: %s NOT exists, when load_pb_for_prediction()' % pb_file_path)
            return -1
        #
        # graph
        self.graph = tf.Graph()
        #
        with self.graph.as_default():
            #
            with open(pb_file_path, "rb") as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                #
                tf.import_graph_def(graph_def, name="")
                #
            #
            ''' input and output tensors '''
            #
            self.x = self.graph.get_tensor_by_name('x-input:0')
            #
            self.results = self.graph.get_tensor_by_name('softmax:0')
            #
        #
        print('graph loaded for prediction')
        #
        # sess
        if portion:
            self.z_sess_config.gpu_options.per_process_gpu_memory_fraction = portion 
        #
        with self.graph.as_default():
            sess = tf.Session(config = self.z_sess_config)
        #
        self.sess = sess
        #
        return 0
        #
    
    def predict(self, x):
        #
        with self.graph.as_default():
            #
            ''' tensors and run '''
            #
            feed_dict = {self.x: x}
            results = self.sess.run([self.results], feed_dict)
            #            
            return results
        #
    
    def z_define_graph_all(self, graph, flag_activation, train = True):
        #
        with graph.as_default():
            #
            x = tf.placeholder(tf.float32, (None, None, None, 3), name = 'x-input')
            y = tf.placeholder(tf.float32, (None, None), name = 'y-input')
            #
            results = resnet_model(x, flag_activation, train)
            #
            if REG_LAMBDA > 0:
                #
                trainable_vars = tf.trainable_variables() 
                loss_reg = tf.add_n([ tf.nn.l2_loss(v) for v in trainable_vars if 'bias' not in v.name ]) \
                                                                                 * REG_LAMBDA
                #
                # loss_infer
                # diff_sq = tf.square(y - results)
                # loss_infer = tf.reduce_mean(tf.reduce_sum(diff_sq, axis = 1), name = 'loss_infer') 
                loss_infer = tf.reduce_mean(-tf.reduce_sum(y * tf.log(results), axis = 1), name = 'loss_infer')
                #
                loss_all = tf.add(loss_infer, loss_reg, name = 'loss')
                #
            else:
                loss_all = tf.reduce_mean(-tf.reduce_sum(y * tf.log(results), axis = 1), name = 'loss')
                #
            #
            print(results.op.name)
            print(loss_all.op.name)
            #
            print('forward graph defined, training = %s' % train)
            #            
            # print(self.graph.get_operations())
            #
            global_step = tf.train.get_or_create_global_step()
            #
            # Update batch norm stats [http://stackoverflow.com/questions/43234667]
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            # 
            with tf.control_dependencies(extra_update_ops):
                #
                learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
                                                           tf.train.get_global_step(),
                                                           DECAY_STEPS,
                                                           DECAY_RATE,
                                                           staircase = DECAY_STAIRCASE,
                                                           name = 'learning_rate')
                # optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
                # optimizer = tf.train.MomentumOptimizer(learning_rate, MOMENTUM)
                # optimizer = tf.train.AdagradOptimizer(learning_rate)
                # optimizer = tf.train.AdadeltaOptimizer(learning_rate)
                optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate, beta1 = MOMENTUM)
                train_op = tf.contrib.layers.optimize_loss(loss = loss_all,
                                                           global_step = tf.train.get_global_step(),
                                                           learning_rate = learning_rate,
                                                           optimizer = optimizer,
                                                           name = 'train_op')  #variables = rnn_vars)
                #                
                print(global_step.op.name)
                print(train_op.op.name)
                #
                print('train graph defined, training = %s' % train)
                #
                
    def prepare_for_validation(self, flag_act):
        #
        # validation graph
        self.graph = tf.Graph()
        self.z_define_graph_all(self.graph, flag_act, self.z_valid_option)
        #
        # sess
        with self.graph.as_default():
            sess = tf.Session(config = self.z_sess_config)
        #
        self.sess = sess
        #

    def validate(self, data_valid):
        #
        with self.graph.as_default():
            #
            saver = tf.train.Saver()
            #
            with tf.Session(config = self.z_sess_config) as sess:
                #
                tf.global_variables_initializer().run()
                #
                # restore with saved data
                ckpt = tf.train.get_checkpoint_state(self.z_model_dir)
                #
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                #
                # pb
                constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, \
                                                                           output_node_names = \
                                                                           ['softmax'])
                pb_file = os.path.join(self.z_model_dir, self.z_model_name + '.pb')
                #
                with tf.gfile.FastGFile(pb_file, mode='wb') as f:
                    f.write(constant_graph.SerializeToString())
                #
                #
                # variables
                #
                x = self.graph.get_tensor_by_name('x-input:0')
                y = self.graph.get_tensor_by_name('y-input:0')
                #
                results = self.graph.get_tensor_by_name('softmax:0')
                #
                loss_ts = self.graph.get_tensor_by_name('loss:0')
                #
                # validate
                num_samples = len(data_valid['x'])
                num_right = 0
                num_batchs = np.ceil(num_samples * 1.0 / self.z_batch_size)
                loss_sum = 0.0
                #
                curr = 0
                batch_start = 0
                batch_end = self.z_batch_size
                while batch_start < num_samples:
                    #
                    data = data_valid['x'][batch_start:batch_end]
                    labels = data_valid['y'][batch_start:batch_end]
                    #
                    feed_dict = {x: data, y: labels}
                    #
                    batch_start = batch_start + self.z_batch_size
                    batch_end = min(batch_end + self.z_batch_size, num_samples)
                    #
                    outputs, loss = sess.run([results, loss_ts], feed_dict)
                    #
                    curr += 1                    
                    #
                    out_list = [np.argmax(item) for item in outputs]
                    label_list = [np.argmax(item) for item in labels]
                    diff = [ x[0]-x[1] for x in zip(out_list, label_list)]
                    #
                    num_right += diff.count(0)
                    loss_sum += loss
                    #
                    if curr % 10 == 0:
                        print('curr: %d / %d, loss_mean: %f' % (curr, num_batchs, loss_sum/curr) )
                    #
                #
                acc = num_right * 1.0 / num_samples
                loss_mean = loss_sum / num_batchs
                #
                print('validation finished, accuracy: %f' % acc)
                #
                return acc, loss_mean
                #
    
    def train_and_valid(self, data_train, data_valid, flag_act):
        #
        # model save-path
        if not os.path.exists(self.z_model_dir): os.mkdir(self.z_model_dir)
        #   
        # training graph
        self.z_graph = tf.Graph()
        self.z_define_graph_all(self.z_graph, flag_act)
        #
        # validation graph and sess
        self.prepare_for_validation(flag_act)
        #
        # begin to train
        with self.z_graph.as_default():
            #
            saver = tf.train.Saver()
            #
            with tf.Session(config = self.z_sess_config) as sess:
                #
                tf.global_variables_initializer().run()
                #
                # restore with saved data
                ckpt = tf.train.get_checkpoint_state(self.z_model_dir)
                #
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                #
                #
                # variables
                #
                x = self.z_graph.get_tensor_by_name('x-input:0')
                y = self.z_graph.get_tensor_by_name('y-input:0')
                #
                loss_ts = self.z_graph.get_tensor_by_name('loss:0')
                #
                #
                global_step = self.z_graph.get_tensor_by_name('global_step:0')
                learning_rate = self.z_graph.get_tensor_by_name('learning_rate:0')
                train_op = self.z_graph.get_tensor_by_name('train_op/control_dependency:0')
                #
                #
                print(' ')
                print('start training ...')
                #
                num_samples = len(data_train['x'])
                #
                # start training
                start_time = time.time()
                begin_time = start_time 
                step = 0
                #
                for curr_iter in range(TRAINING_STEPS):
                    #
                    # save and validate
                    if step % self.z_valid_freq == 0:
                        #
                        # ckpt
                        print('save model to ckpt ...')
                        saver.save(sess, os.path.join(self.z_model_dir, self.z_model_name), global_step = step)
                        #
                        # validate
                        print('validating ...')
                        acc_t, loss_mt = self.validate(data_train)
                        #acc_t = 0.0, loss_mt = 0.0
                        #
                        print('validating ...')
                        acc_v, loss_mv = self.validate(data_valid)
                        #
                        file_path = self.z_model_dir + '/valiation_accuracy.txt'
                        with open(file_path, 'a') as f:
                            #
                            f.write('%d: %g, %g; %g, %g\n' % (step, loss_mt, acc_t, loss_mv, acc_v))
                            #
                        #
                        # acc = self.test(data_train)
                        #
                    #
                    # train
                    index_batch = random.sample(range(num_samples), self.z_batch_size)
                    #
                    data = [data_train['x'][i] for i in index_batch] 
                    labels = [data_train['y'][i] for i in index_batch]
                    #
                    feed_dict = {x: data, y: labels}
                    #
                    _, loss, step, lr = sess.run([train_op, loss_ts, global_step, learning_rate], \
                                                 feed_dict)
                    #
                    if step % 1 == 0:
                        #
                        curr_time = time.time()            
                        #
                        print('step: %d, loss: %g, lr: %g, sect_time: %.1f, total_time: %.1f' %
                              (step, loss, lr, curr_time - begin_time, curr_time - start_time))
                        #
                        begin_time = curr_time
                        #
                    
                            
    def test(self, data_test):
        #
        self.prepare_for_prediction()
        #
        # test
        num_samples = len(data_test['x'])
        num_right = 0
        num_batchs = np.ceil(num_samples * 1.0 / self.z_batch_size)
        #
        curr = 0
        batch_start = 0
        batch_end = self.z_batch_size
        while batch_start < num_samples:
            #
            data = data_test['x'][batch_start:batch_end]
            labels = data_test['y'][batch_start:batch_end]
            #
            batch_start = batch_start + self.z_batch_size
            batch_end = min(batch_end + self.z_batch_size, num_samples)
            #
            outputs = self.predict(data)
            #
            curr += 1                    
            #
            out_list = [np.argmax(item) for item in outputs]
            label_list = [np.argmax(item) for item in labels]
            diff = [ x[0]-x[1] for x in zip(out_list, label_list)]
            #
            print(out_list)
            print(label_list)
            print(diff)
            #
            num_right += diff.count(0)
            #
            print('curr: %d / %d, right: %d / %d' % (curr, num_batchs, diff.count(0), BATCH_SIZE))
            #
        #
        acc = num_right * 1.0 / num_samples
        #
        print('test finished, accuracy: %f' % acc)
        #
        return acc
        #

    
