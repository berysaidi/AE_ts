# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 10:43:29 2016

@author: Rob Romijnders
"""
import argparse
import numpy as np
import tensorflow as tf
import os
from tensorflow.contrib.tensorboard.plugins import projector
from AE_ts_model import Model, open_data, plot_data, plot_z_run
tf.logging.set_verbosity(tf.logging.ERROR)


"""Hyperparameters"""
# Please download data here: https://www.cs.ucr.edu/~eamonn/time_series_data/
LOG_DIR = "./foo.log"  # Directory for the logging


config = dict()  # Put all configuration information into the dict
config['num_layers'] = 2  # number of layers of stacked RNN's
config['hidden_size'] = 90  # memory cells in a layer
config['max_grad_norm'] = 5  # maximum gradient norm during training
config['batch_size'] = batch_size = 64
config['learning_rate'] = .005
config['crd'] = 1  # Hyperparameter for future generalization
config['num_l'] = 20  # number of units in the latent space


#change me
datapath = './ecg_data.csv'
#constant bullshit
dataset="ECG5000"
datadir = datapath + '/' + dataset + '/' + dataset

#real input should be this
test_path = datadir + '_TEST'
train_path = datadir + '_TRAIN'

def api_open_data(test_p, train_p, ratio_train=0.8):
    """Input:
    ratio_train: ratio to split training and testset
    """
    data_test_val = np.loadtxt(test_p, delimiter=',')[:-1]
    data_train = np.loadtxt(train_p, delimiter=',')

    data = np.concatenate((data_train, data_test_val), axis=0)

    N, D = data.shape

    ind_cut = int(ratio_train * N)
    ind = np.random.permutation(N)
    return data[ind[:ind_cut], 1:], data[ind[ind_cut:], 1:], data[ind[:ind_cut], 0], data[ind[ind_cut:], 0]






def api_call(config)
     
# Load the data
    X_train, X_val, y_train, y_val = api_open_data(test_path, train_path, 0.8)

    Nval = X_val.shape[0]
    D = X_train.shape[1]
    # Organize the classes
    num_classes = len(np.unique(y_train))
    base = np.min(y_train)  # Check if data is 0-based
    if base != 0:
        y_train -= base
        y_val -= base



    config['sl'] = sl = D  # sequence length
    print('We have %s observations with %s dimensions' % (N, D))

    """Training time!"""
    model = Model(config)
    sess = tf.Session()
    perf_collect = np.zeros((2, int(np.floor(max_iterations / plot_every))))
    saver = tf.train.Saver()
    saver.restore(sess, save_path=os.path.join(LOG_DIR, "model.ckpt-10"))

    # Extract the latent space coordinates of the validation set
    start = 0
    label = []  # The label to save to visualize the latent space
    z_run = []

    while start + batch_size < Nval:
        run_ind = range(start, start + batch_size)
        z_mu_fetch = sess.run(model.z_mu, feed_dict={model.x: X_val[run_ind], model.keep_prob: 1.0})
        z_run.append(z_mu_fetch)
        start += batch_size

    z_run = np.concatenate(z_run, axis=0)
    label = y_val[:start]

    plot_z_run(z_run, label)

    sess.close()

    return z_run
   

if __name__ == '__main__':

    result = api_call(config)
