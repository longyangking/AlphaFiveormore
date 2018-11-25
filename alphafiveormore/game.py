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
        self.score = 0

    def get_score(self):
        return self.score

    def get_availables(self):
        return np.array(self.availables)

    def is_index_null(self, index):
        flag = (index in self.availables)
        return flag

    def get_board(self):
        return np.copy(self.board)

    def get_board_shape(self):
        return np.copy(self.board_shape)

    def init(self):
        Nx, Ny = self.board_shape
        self.board = np.zeros(self.board_shape)
        self.availables = list(range(Nx*Ny))
        self.score = 0
        indexs = np.random.choice(self.availables, self.n_init, replace=False)
        for index in indexs:
            position = (index%Nx, int(index/Nx))
            color = np.random.choice(N_COLORS) + 1  # +1 means bias
            self.board[position] = color
            self.availables.remove(index)

    def release(self):
        Nx, Ny = self.board_shape

        # bug: if the number of existing places is less than the release number

        indexs = np.random.choice(self.availables, self.n_release, replace=False)
        for index in indexs:
            position = (index%Nx, int(index/Nx))
            color = np.random.choice(N_COLORS) + 1  # +1 means bias
            self.board[position] = color
            self.availables.remove(index)

    def update(self):
        '''
        Return the current status of board, is_end?
        '''
        
        self.release()

        # Delete the satisfying blocks

        flag = (len(self.availables) == 0)
        return flag

    def play(self, action):  
        start_index, end_index = action
        Nx, Ny = self.board_shape

        start_position = (start_index%Nx, int(start_index/Nx))
        end_position = (end_index%Nx, int(end_index/Nx))

        if end_index in self.availables:
            self.availables.remove(end_index)
        if start_position not in self.availables:
            self.availables.append(start_index)

        self.board[start_position], self.board[end_position] = self.board[end_position], self.board[start_position]
        flag = self.update() 

        # Judge whether this operation is valid: A possible tranport path exist?

        return flag

class GameEngine:
    def __init__(self, state_shape, ai=None, verbose=False):
        self.state_shape = state_shape
        self.board_shape = state_shape[:2]
        self.ai = ai
        self.verbose = verbose

        self.gameboard = None
        self.flag = False

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

    def pressaction(self, action):
        if action != (-1, -1):
            self.flag = self.gameboard.play(action=action)

    def start(self):
        '''
        Start to play "Five or more" game for human
        '''
        Nx, Ny, channel = self.state_shape
        self.gameboard = Board(board_shape=(Nx, Ny))
        self.gameboard.init()

        if self.verbose:
            print("Initiating UI...")

        from ui import UI
        ui = UI(pressaction=self.pressaction, board=self.gameboard.get_board(), sizeunit=50)
        ui.start()

        while not self.flag:
            board = self.gameboard.get_board()
            ui.setboard(board)

        score = self.gameboard.get_score()
        ui.gameend(score)

    def pressaction_ai(self, action):
        if action == (-1, -1): # press space
            state = self.get_state()
            action = self.ai.play(state)
            self.flag = self.gameboard.play(action=action)

    def start_ai(self):
        '''
        Start to play "Five or more" game for human
        '''
        Nx, Ny, channel = self.state_shape
        self.gameboard = Board(board_shape=(Nx, Ny))
        self.gameboard.init()

        if self.verbose:
            print("Initiating UI...")

        from ui import UI
        ui = UI(pressaction=self.pressaction_ai, board=self.gameboard.get_board(), sizeunit=50)
        ui.start()

        while not self.flag:
            board = self.gameboard.get_board()
            ui.setboard(board)

        score = self.gameboard.get_score()
        ui.gameend(score)

    def start_selfplay(self, epsilon, gamma):
        '''
        Start self-play process to get train data for AI model
        '''
        self.gameboard = Board(board_shape=(Nx, Ny))
        self.gameboard.init()

        board = self.gameboard.get_board()
        self.boards.append(board)

        while not self.flag:
            self.update_states()

            state = self.get_state()
            availables = self.gameboard.get_availables()

            action = self.ai.play(state, availables)    
            # TODO 
            # 1. Choose action based on the deep Q-value
            # 2. Record the rewards and train networks

            self.actions.append(action)
            score = self.gameboard.get_score()
            self.scores.append(score)
            self.flag = self.gameboard.play(action)

            board = self.gameboard.get_board()
            self.boards.append(board)

        action_Q = list()
        # Intrinsic reward and external reward
        intrinsic_rewards = self.ai.get_intrinsic_rewards(self.states)
        for i in len(self.states):
            pass

        return self.states, action_Q
