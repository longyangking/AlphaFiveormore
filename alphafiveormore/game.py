import numpy as np 

COLOR_TABLE = {
    'null':0,
    'red':1,
    'blue':2,
    'green':3,
    'yellow':4,
    'purple':5
}
N_COLORS = 5

class Block:
    def __init__(self, position, color):
        self.position = position
        self.color = color
    
    def get_position(self):
        return np.copy(self.position)

    def get_color(self):
        return self.color

    def set_position(self, position):
        self.position = position

    def set_color(self, color):
        self.color = color

class Board:
    def __init__(self, board_shape, n_init=3, n_release=3, verbose=False):
        self.board_shape = board_shape
        self.verbose = verbose

        self.n_init = n_init
        self.n_release = n_release

        self.board = np.zeros(self.board_shape)
        self.availables = list()
        self.scores = 0

    def get_board(self):
        return np.copy(self.board)

    def get_board_shape(self):
        return np.copy(self.board_shape)

    def init(self):
        Nx, Ny = self.board_shape
        self.board = np.zeros(self.board_shape)
        self.availables = list(range(Nx*Ny))
        self.scores = 0
        indexs = np.random.choice(self.availables, self.n_init, replace=False)
        for index in indexs:
            position = (index%Nx, int(index/Nx))
            color = np.random.choice(N_COLORS) + 1  # +1 means bias
            self.board[position] = color
            self.availables.remove(index)

    def release(self):
        indexs = np.random.choice(self.availables, self.n_release, replace=False)
        for index in indexs:
            position = (index%Nx, int(index/Nx))
            color = np.random.choice(N_COLORS) + 1  # +1 means bias
            self.board[position] = color
            self.availables.remove(index)

    def update(self):
        pass

    def play(self, action):  
        start_index, end_start = action
        pass

class GameEngine:
    def __init__(self, state_shape, ai=None, verbose=False):
        self.state_shape = state_shape
        self.board_shape = state_shape[:2]
        self.ai = ai
        self.verbose = verbose

        self.states = list()
        self.boards = list()
        self.actions = list()
        self.scores = list()

    def get_state(self):
        return self.states[-1]

    def update_states(self):
        Nx, Ny, channel = self.state_shape
        state = np.zeros((Nx, Ny, channel)) # Only current state due to the special feature of this game
        state[:,:,0] = self.boards[-1]
        self.states.append(state)

    def start(self):
        pass

    def start_ai(self):
        pass

    def start_selfplay(self, epsilon, gamma):
        '''
        Start self-play process to get train data for AI model
        '''
        action_Q = list()

        # Intrinsic reward and external reward

        return self.states, action_Q
