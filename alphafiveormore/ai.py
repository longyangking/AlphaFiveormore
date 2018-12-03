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

        # The action model perform the Q values of actions base on the states 
        self.action_model = self.build_action_model()
        # The feature model abstract the states into the feature vectors
        self.feature_model = self.build_feature_model()
        # The forward model predict the feature vectors of the future states based on the current actions and states
        self.forward_model = self.build_forward_model()
        # The inverse model derive the selected actions based on the current states and the future states
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

        self.train_inverse_model = Model(inputs=[state_tensor_t, state_tensor_t1], outputs=predict_action_tensor)
        self.train_inverse_model.compile(
            loss='mse',
            optimizer='adam'
        )

        feature_tensor_t = Input(shape=(self.feature_dim,))
        action_tensor = Input(shape=(self.output_dim,))

        predict_feature_tensor_t1 = self.forward_model([action_tensor, feature_tensor_t])

        self.train_forward_model = Model(inputs=[action_tensor, feature_tensor_t], outputs=predict_feature_tensor_t1)
        self.train_forward_model.compile(
            loss='mse',
            optimizer='adam'
        )

    def build_action_model(self):
        network_structure = self.network_structure['action_model']

        state_tensor = Input(shape=self.input_shape)
        if len(network_structure) == 0:
            raise Exception("No structure for action model!")
        
        out = self.__conv_block(state_tensor, filters=network_structure[0]['filters'], kernel_size=network_structure[0]['kernel_size'])
        is_flatten = False
        if len(network_structure) > 1:
            for structure in network_structure:
                if 'filters' in structure:
                    out = self.__res_block(out, filters=structure['filters'], kernel_size=structure['kernel_size'])
                elif 'units' in structure:
                    if not is_flatten:
                        out = Flatten()(out)
                        is_flatten = True
                    out = self.__dense_block(out, units=structure['units'])

        action_tensor = Dense(
            self.output_dim,
            activation='sigmoid', 
            use_bias=False, 
            kernel_initializer='glorot_uniform', 
            kernel_regularizer=regularizers.l2(self.l2_const)
        )(out)
        
        return Model(inputs=state_tensor, outputs=action_tensor)

    def build_feature_model(self):
        network_structure = self.network_structure['feature_model']

        state_tensor = Input(shape=self.input_shape)
        if len(network_structure) == 0:
            raise Exception("No structure for feature model!")
        
        out = self.__conv_block(state_tensor, filters=network_structure[0]['filters'], kernel_size=network_structure[0]['kernel_size'])
        is_flatten = False
        if len(network_structure) > 1:
            for structure in network_structure:
                if 'filters' in structure:
                    out = self.__res_block(out, filters=structure['filters'], kernel_size=structure['kernel_size'])
                elif 'units' in structure:
                    if not is_flatten:
                        out = Flatten()(out)
                        is_flatten = True
                    out = self.__dense_block(out, units=structure['units'])

        feature_tensor = Dense(
            self.feature_dim, 
            activation='sigmoid', 
            use_bias=False, 
            kernel_initializer='glorot_uniform', 
            kernel_regularizer=regularizers.l2(self.l2_const)
        )(out)

        return Model(inputs=state_tensor, outputs=feature_tensor)

    def build_forward_model(self):
        network_structure = self.network_structure['forward_model']

        feature_tensor = Input(shape=(self.feature_dim,))
        action_tensor = Input(shape=(self.output_dim,))

        out = Concatenate()([feature_tensor, action_tensor])

        if len(network_structure) == 0:
            raise Exception("No structure for forward model!")
        for structure in network_structure:
            out = self.__dense_block(out, structure['units'])
        
        predict_feature_tensor = Dense(
            self.feature_dim,
            activation='sigmoid', 
            use_bias=False, 
            kernel_initializer='glorot_uniform', 
            kernel_regularizer=regularizers.l2(self.l2_const)
        )(out)

        return Model(inputs=[action_tensor, feature_tensor], outputs=predict_feature_tensor)

    def build_inverse_model(self):
        network_structure = self.network_structure['forward_model']
        
        feature_tensor_t = Input(shape=(self.feature_dim,))
        feature_tensor_t1 = Input(shape=(self.feature_dim,))

        out = Concatenate()([feature_tensor_t, feature_tensor_t1])

        if len(network_structure) == 0:
            raise Exception("No structure for feature model!")
        for structure in network_structure:
            out = self.__dense_block(out, structure['units'])

        action_tensor = Dense(
            self.output_dim,
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
            kernel_regularizer=regularizers.l2(self.l2_const))(x)
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
            kernel_regularizer=regularizers.l2(self.l2_const)
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
        history_action_model = self.action_model.fit(np.array(states), action_probs, epochs=epochs, batch_size=batch_size, verbose=self.verbose)

        if self.verbose:
            print("Updating neural network of inverse model...")
        N = len(states)
        history_inverse_model = self.train_inverse_model.fit([np.array(states[:N-1]), np.array(states_next[:N-1])], np.array(action_probs[:N-1]), epochs=epochs, batch_size=batch_size, verbose=self.verbose)

        if self.verbose:
            print("Updating neural network of forward model...")
        features = self.feature_model.predict(np.array(states[:N-1]))
        features_next = self.feature_model.predict(np.array(states_next[:N-1]))
        history_forward_model = self.train_forward_model.fit([np.array(action_probs[:N-1]), features], features_next, epochs=epochs, batch_size=batch_size, verbose=self.verbose)

        if self.verbose:
            print("End of updating neural network.")
        
        loss_action = history_action_model.history['loss'][-1]
        loss_inverse = history_inverse_model.history['loss'][-1]
        loss_forward = history_forward_model.history['loss'][-1]
        return loss_action, loss_inverse, loss_forward

    def get_intrinsic_rewards(self, dataset):
        states, action_probs, states_next = dataset
        N = len(states)
        features = self.feature_model.predict(np.array(states[:N-1]))
        features_next = self.feature_model.predict(np.array(states_next[:N-1]))
        features_next_predict = self.forward_model.predict([np.array(action_probs[:N-1]), features])

        intrinsic_rewards = self.eta/2*np.sum(np.abs(features_next - features_next_predict), axis=1)
        return intrinsic_rewards

    def update(self, dataset):
        states, action_probs, states_next = dataset

        if self.verbose:
            print("Updating neural network on batch (Usually for mini-batch trainning).")

        if self.verbose:
            print("Updating neural network of action model...")
        self.action_model.train_on_batch(states, action_probs)

        if self.verbose:
            print("Updating neural network of inverse model...")
        self.train_inverse_model.train_on_batch([states, states_next], action_probs)

        if self.verbose:
            print("Updating neural network of forward model...")
        features = self.feature_model.predict(states)
        features_next = self.feature_model.predict(states_next)
        self.train_forward_model.train_on_batch([action_probs, features], features_next)

        if self.verbose:
            print("End of updating neural network on batch.")

    def predict(self, state):
        states = np.array(state).reshape(1, *self.input_shape)
        action_probs = self.action_model.predict(states)
        return action_probs[0]

    def save_models(self, filename):
        self.action_model.save_weights('{filename}_{modelname}.h5'.format(filename=filename,modelname='action_model'))
        self.feature_model.save_weights('{filename}_{modelname}.h5'.format(filename=filename,modelname='feature_model'))
        self.forward_model.save_weights('{filename}_{modelname}.h5'.format(filename=filename,modelname='forward_model'))
        self.inverse_model.save_weights('{filename}_{modelname}.h5'.format(filename=filename,modelname='inverse_model'))
        
        self.train_forward_model.save_weights('{filename}_{modelname}.h5'.format(filename=filename,modelname='train_forward_model'))
        self.train_inverse_model.save_weights('{filename}_{modelname}.h5'.format(filename=filename,modelname='train_inverse_model'))

    def load_models(self, filename):
        self.action_model.load_weights('{filename}_{modelname}.h5'.format(filename=filename,modelname='action_model'))
        self.feature_model.load_weights('{filename}_{modelname}.h5'.format(filename=filename,modelname='feature_model'))
        self.forward_model.load_weights('{filename}_{modelname}.h5'.format(filename=filename,modelname='forward_model'))
        self.inverse_model.load_weights('{filename}_{modelname}.h5'.format(filename=filename,modelname='inverse_model'))
        
        self.train_forward_model.load_weights('{filename}_{modelname}.h5'.format(filename=filename,modelname='train_forward_model'))
        self.train_inverse_model.load_weights('{filename}_{modelname}.h5'.format(filename=filename,modelname='train_inverse_model'))

    def plot_models(self, filename):
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
    def __init__(self, state_shape, verbose=False):
        self.state_shape = state_shape
        Nx, Ny, channel = state_shape
        self.action_dim  = 2*Nx*Ny # action probability of start_index and end_index
        self.verbose = verbose

        network_structure = dict()
        network_structure['action_model'] = list()
        # Action model parameters
        network_structure['action_model'].append({"filters":64, "kernel_size":3})
        network_structure['action_model'].append({"filters":64, "kernel_size":3})
        network_structure['action_model'].append({"filters":64, "kernel_size":3})
        network_structure['action_model'].append({"filters":64, "kernel_size":3})
        network_structure['action_model'].append({"units":256})
        network_structure['action_model'].append({"units":256})
        # Feature model parameters
        network_structure['feature_model'] = list()
        network_structure['feature_model'].append({"filters":64, "kernel_size":3})
        network_structure['feature_model'].append({"filters":64, "kernel_size":3})
        network_structure['feature_model'].append({"filters":64, "kernel_size":3})
        network_structure['feature_model'].append({"units":256})
        network_structure['feature_model'].append({"units":256})
        # Forward model parameters
        network_structure['forward_model'] = list()
        network_structure['forward_model'].append({"units":256})
        network_structure['forward_model'].append({"units":256})
        network_structure['forward_model'].append({"units":256})
        network_structure['forward_model'].append({"units":256})
        # Inverse model parameters
        network_structure['inverse_model'] = list()
        network_structure['inverse_model'].append({"units":256})
        network_structure['inverse_model'].append({"units":256})
        network_structure['inverse_model'].append({"units":256})
        network_structure['inverse_model'].append({"units":256})
        
        self.nnet = NeuralNetwork(
            input_shape=self.state_shape,
            output_dim=self.action_dim,
            network_structure=network_structure,
            verbose=self.verbose
        )

    def get_action_dim(self):
        return self.action_dim

    def train(self, dataset, epochs, batch_size):
        loss = self.nnet.fit(dataset, epochs, batch_size)
        return loss

    def update(self, dataset):
        self.nnet.update(dataset)

    def get_intrinsic_rewards(self, states_set):
        intrinsic_rewards = self.nnet.get_intrinsic_rewards(states_set)
        return intrinsic_rewards

    def evaluate_function(self, state, availables):
        action_prob = self.nnet.predict(state)
        N = int(self.action_dim/2)
        start_prob = action_prob[:N]
        end_prob = action_prob[N:]

        # Only available positions can be applied
        eps = 1e-12
        have_chesses = [index for index in range(N) if index not in availables]
        start_prob[availables] = 0.0
        end_prob[have_chesses] = 0.0

        start_prob = start_prob/(np.sum(start_prob) + eps)
        end_prob = end_prob/(np.sum(end_prob) + eps)

        action_prob = np.concatenate([start_prob, end_prob])
        return action_prob

    def play(self, state, availables):
        action_prob = self.evaluate_function(state, availables)
        N = int(self.action_dim/2)
        start_prob = action_prob[:N]
        end_prob = action_prob[N:]

        start_index = np.argmax(start_prob)
        end_index = np.argmax(end_prob)
        return start_index, end_index

    def save_nnet(self, filename):
        self.nnet.save_models(filename)

    def load_nnet(self, filename):
        self.nnet.load_models(filename)

    def plot_nnet(self, filename):
        self.nnet.plot_models(filename)