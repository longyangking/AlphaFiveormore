from ai import AI
import time
import numpy as np
from game import GameEngine

class TrainAI:
    def __init__(self, state_shape, ai=None, verbose=False):
        self.state_shape = state_shape
        self.verbose = verbose

        if ai is None:
            self.ai = AI(
                state_shape=state_shape,
                verbose=verbose
            )
        else:
            self.ai = ai

        self.losses = list()

    def get_losses(self):
        return np.array(self.losses)

    def get_selfplay_data(self, n_rounds, epsilon=0.5, gamma=0.9):
        states = list()
        action_values = list()

        if self.verbose:
            starttime = time.time()
            print("Start self-play process with rounds [{0}]:".format(n_rounds))

        for i in range(n_rounds):
            if self.verbose:
                print("{0}th self-play round...".format(i+1))

            engine = GameEngine(
                state_shape=self.state_shape,
                ai=self.ai,
                verbose=self.verbose
            )

            _states, _action_values = engine.start_selfplay(epsilon=epsilon, gamma=gamma)
            for i in range(len(_action_values)):
                states.append(_states[i])
                action_values.append(_action_values[i])
        
        if self.verbose:
            endtime = time.time()
            print("End of self-play process with data size [{0}] and cost time [{1:.1f}s].".format(
                len(action_values),  (endtime - starttime)))

        #states = np.array(states)
        action_values = np.array(action_values)
        next_states = states[:-1]
        next_states.append(None)

        return states, action_values, next_states

    def update_ai(self, dataset):
        if self.verbose:
            print("Start to update the network of AI model...")

        loss = self.ai.train(dataset, epochs=3, batch_size=32)

        if self.verbose:
            print("End of updating with final loss [{0}]".format(loss))

        self.losses.append(loss)

    def start(self, filename):
        '''
        Main training process
        '''
        n_epochs = 1000
        n_rounds = 5
        n_checkpoints = 2

        if self.verbose:
            print("Train AI model with epochs: [{0}]".format(n_epochs))
        
        for i in range(n_epochs):
            if self.verbose:
                print("{0}th self-play training process ...".format(i+1))

            dataset = self.get_selfplay_data(n_rounds)
            self.update_ai(dataset)

            if self.verbose:
                print("End of training process.")

            if (i+1)%n_checkpoints == 0:
                if self.verbose:
                    print("Checkpoint: Saving AI model with filename [{0}] ...".format(filename),end="")

                self.ai.save_nnet(filename)

                if self.verbose:
                    print("OK!")