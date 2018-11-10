from __future__ import print_function

import keras
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, Activation, LeakyReLU, Add
from keras.optimizers import SGD, Adam
from keras import regularizers
from keras import initializers
import keras.backend as K

import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Only error will be shown

class NeuralNetwork:
    '''
    Deep Q network with Intrinsic Curiosity Module (ICM)
    '''
    def __init__(self, input_shape, output_dim, network_structure, feature_dim=30, 
        learning_rate=1e-3, l2_const=1e-4, verbose=False
    ):
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.network_structure = network_structure
        self.feature_dim = feature_dim

        self.learning_rate = learning_rate
        self.l2_const = l2_const
        self.verbose = verbose

        self.action_model = self.build_action_model()
        self.icm_model = self.build_icm_model()

    def build_action_model(self):

    def build_icm_model(self):

    def __conv_block(self, x, filters, kernel_size):
        out = Conv2D(
            filters,
            kernel_size=kernel_size,
            padding='same',
            activation='linear',
            kernel_initializer=initializers.lecun_normal(),
            kernel_constraints=regularizers.l2(self.l2_const)
        )(x)
        out = BatchNormalization(axis=1, momentum=0.9)(out)
        out = LeakyReLU(alpha=0.1)(out)
        return out

    def __res_block(self, x, filters, kernel_size):
        out = Conv2D(
            filters,
            kernel_size=kernel_size,
            padding='same',
            activation='linear',
            kernel_initializer=initializers.glorot_normal(),
            kernel_constraints=regularizers.l2(self.l2_const)
        )(x)
        out = BatchNormalization(axis=1, momentum=0.9)(out)
        out = Add()([out, x])
        out = LeakyReLU(alpha=0.1)(out)
        return out

    def __dense_block(self, x, units):
        out = Dense(
            units,
            activation=None, 
            use_bias=False, 
            kernel_initializer='glorot_uniform', 
            kernel_regularizer=regularizers.l2(self.l2_const)
        )(x)
        out = LeakyReLU(alpha=0.1)(out)
        return out

    def __action_Q_block(self, x):

    def __feature_block(self, x):

    def __intrinsic_block(self, x, y):

    def fit(self, dataset, epochs, batch_size):

    def update(self, dataset):

    def predict(self, state):

    def save_model(self, filename):

    def load_model(self, filename):

    def plot_model(self, filename):

class AI:
    '''
    The main AI model class
    '''
    def __init__(self, state_shape, action_dim, verbose=False):
        self.state_shape = state_shape
        self.action_dim = action_dim
        self.verbose = verbose

    def train(self, dataset):

    def update(self, dataset):

    def evaluate_function(self, state):

    def play(self, state):

    def save_nnet(self, filename):

    def load_nnet(self, filename):

    def plot_nnet(self, filename):