from ai import AI
from game import GameEngine

class TrainAI:
    def __init__(self, state_shape, ai=None, verbose=False):
        self.state_shape = state_shape
        Nx, Ny, channel = state_shape
        self.action_dim  = 2*Nx*Ny # action probability of start_index and end_index
        self.verbose = verbose

        if ai is None:
            self.ai = AI(
                state_shape=state_shape,
                action_dim=action_dim,
                verbose=verbose
            )
        else:
            self.ai = ai

        self.losses = list()

    def get_losses(self):
        return np.array(self.losses)

    def get_selfplay_data(self, n_rounds):

    def update_ai(self, dataset):

    def start(self, filename):