from __future__ import print_function

import keras
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, Activation, LeakyReLU, Add, Concatenate
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
        learning_rate=1e-3, l2_const=1e-4, eta=1.0, verbose=False
    ):
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.network_structure = network_structure
        self.feature_dim = feature_dim

        self.learning_rate = learning_rate
        self.l2_const = l2_const
        self.eta = eta
        self.verbose = verbose

        self.action_model = self.build_action_model()
        self.feature_model = self.build_feature_model()
        self.forward_model = self.build_forward_model()
        self.inverse_model = self.build_inverse_model()

        self.action_model.compile(
            loss='mse',
            optimizer='adam'
        )

        state_tensor_t = Input(shape=self.input_shape)
        state_tensor_t1 = Input(shape=self.input_shape)
        action_tensor = Input(shape=(self.output_dim,))

        feature_tensor_t = self.feature_model(state_tensor_t)
        feature_tensor_t1 = self.feature_model(state_tensor_t1)

        predict_action_tensor = self.inverse_model([feature_tensor_t, feature_tensor_t1])

        self.train_inverse_model = Model(inputs=[state_tensor_t, state_tensor_t1], outputs=action_tensor)
        self.train_inverse_model.compile(
            loss='mse',
            optimizer='adam'
        )

        feature_tensor_t = Input(shape=(self.feature_dim,))
        action_tensor = Input(shape=(self.action_dim,))

        predict_feature_tensor_t1 = self.forward_model([action_tensor, feature_tensor_t])

        self.train_forward_model = Model(inputs=[action_tensor, feature_tensor_t], outputs=predict_feature_tensor_t1)
        self.train_forward_model.compile(
            loss='mse',
            optimizer='adam'
        )

    def build_action_model(self):
        state_tensor = Input(shape=self.input_shape)

        out = self.__conv_block(state_tensor, filters=64, kernel_size=3)
        out = self.__res_block(out, filters=64, kernel_size=3)
        out = self.__res_block(out, filters=64, kernel_size=3)
        out = self.__res_block(out, filters=64, kernel_size=3)
        out = self.__res_block(out, filters=64, kernel_size=3)

        out = Flatten()(out)
        out = self.__dense_block(units=200)
        out = self.__dense_block(units=200)
        action_tensor = Dense(
            self.output_dim,
            self.feature_dim,
            activation='sigmoid', 
            use_bias=False, 
            kernel_initializer='glorot_uniform', 
            kernel_regularizer=regularizers.l2(self.l2_const)
        )(out)
        
        return Model(inputs=state_tensor, outputs=action_tensor)

    def build_feature_model(self):
        state_tensor = Input(shape=self.input_shape)

        out = self.__conv_block(state_tensor, filters=64, kernel_size=3)
        out = self.__conv_block(out, filters=64, kernel_size=3)
        out = self.__conv_block(out, filters=64, kernel_size=3)
        out = Flatten()(out)
        out = self.__dense_block(out, units=180)
        feature_tensor = Dense(
            self.feature_dim, 
            activation='sigmoid', 
            use_bias=False, 
            kernel_initializer='glorot_uniform', 
            kernel_regularizer=regularizers.l2(self.l2_const)
        )(out)

        return Model(inputs=state_tensor, outputs=feature_tensor)

    def build_forward_model(self):
        feature_tensor = Input(shape=(self.feature_dim,))
        action_tensor = Input(shape=(self.output_dim,))

        out = Concatenate()([feature_tensor, action_tensor])
        out = self.__dense_block(out, 180)
        out = self.__dense_block(out, 180)
        
        predict_feature_tensor = Dense(
            self.feature_dim,
            activation='sigmoid', 
            use_bias=False, 
            kernel_initializer='glorot_uniform', 
            kernel_regularizer=regularizers.l2(self.l2_const)
        )(out)

        return Model(inputs=[action_tensor, feature_tensor], outputs=predict_feature_tensor)

    def build_inverse_model(self):
        feature_tensor_t = Input(shape=(self.feature_dim,))
        feature_tensor_t1 = Input(shape=(self.feature_dim,))

        out = Concatenate()([feature_tensor_t, feature_tensor_t1])
        out = self.__dense_block(out, 180)
        out = self.__dense_block(out, 180)
        out = self.__dense_block(out, 180)

        action_tensor = Dense(
            self.action_dim,
            activation='sigmoid', 
            use_bias=False, 
            kernel_initializer='glorot_uniform', 
            kernel_regularizer=regularizers.l2(self.l2_const)
        )(out)

        return Model(inputs=[feature_tensor_t, feature_tensor_t1], outputs=action_tensor)

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

    def fit(self, dataset, epochs, batch_size):
        states, action_probs, states_next = dataset

        if self.verbose:
            print("Updating neural network with epochs [{0}] and batch_size [{1}].".format(epochs, batch_size))

        if self.verbose:
            print("Updating neural network of action model...")
        self.action_model.fit(states, action_probs, epochs=epochs, batch_size=batch_size, verbose=self.verbose)

        if self.verbose:
            print("Updating neural network of inverse model...")
        self.train_inverse_model.fit([states, states_next], action_probs, epochs=epochs, batch_size=batch_size, verbose=self.verbose)

        if self.verbose:
            print("Updating neural network of forward model...")
        features = self.feature_model.predict(states)
        features_next = self.feature_model.predict(states_next)
        self.train_forward_model.fit([action_probs, features], features_next, epochs=epochs, batch_size=batch_size, verbose=self.verbose)

        if self.verbose:
            print("End of updating neural network.")

    def get_intrinsic_rewards(self, states_set):
        states, action_probs, states_next = dataset
        features = self.feature_model.predict(states)
        features_next = self.feature_model.predict(states_next)
        features_next_predict = self.forward_model.predict([action_probs, features])

        intrinsic_rewards = self.eta/2*np.sum(np.abs(features_next - features_next_predict), axis=1)
        return intrinsic_rewards

    def update(self, dataset):
        pass

    def predict(self, state):
        states = np.array(state).reshape(1, *self.input_shape)
        action_probs = self.action_model.predict(states)
        return action_probs[0]

    def save_model(self, filename):
        pass

    def load_model(self, filename):
        pass

    def plot_model(self, filename):
        from keras.utils import plot_model
        plot_model(self.action_model, show_shapes=True, to_file='{filename}_{modelname}.png'.format(filename=filename,modelname='action_model'))
        plot_model(self.feature_model, show_shapes=True, to_file='{filename}_{modelname}.png'.format(filename=filename,modelname='feature_model'))
        plot_model(self.forward_model, show_shapes=True, to_file='{filename}_{modelname}.png'.format(filename=filename,modelname='forward_model'))
        plot_model(self.inverse_model, show_shapes=True, to_file='{filename}_{modelname}.png'.format(filename=filename,modelname='inverse_model'))
        
        plot_model(self.train_forward_model, show_shapes=True, to_file='{filename}_{modelname}.png'.format(filename=filename,modelname='train_forward_model'))
        plot_model(self.train_inverse_model, show_shapes=True, to_file='{filename}_{modelname}.png'.format(filename=filename,modelname='train_inverse_model'))
        
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