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

        if len(self.availables) > self.n_release:
            indexs = np.random.choice(self.availables, self.n_release, replace=False)
        else:
            indexs = self.availables

        for index in indexs:
            position = (index%Nx, int(index/Nx))
            color = np.random.choice(N_COLORS) + 1  # +1 means bias
            self.board[position] = color
            self.availables.remove(index)

    def clear_board(self):
        Nx, Ny = self.board_shape
        indexs = list()
        score = 0

        # TODO The mechanism to eliminate chesses on board and calculate the score

        for j in range(Ny):
            for i in range(Nx-5):
                if self.board[i,j] != 0:
                    flag = (self.board[i,j] == self.board[i+1,j] == self.board[i+2,j] == self.board[i+3,j] == self.board[i+4,j])
                    if flag:
                        score += 1
                        for m in range(5):
                            self.board[i+m, j] = 0
                            index = i + m + j*Nx
                            indexs.append(index)

        for i in range(Nx):
            for j in range(Ny-5):
                if self.board[i,j] != 0:
                    flag = (self.board[i,j] == self.board[i,j+1] == self.board[i,j+2] == self.board[i,j+3] == self.board[i,j+4])
                    if flag:
                        score += 1
                        for m in range(5):
                            self.board[i, j+m] = 0
                            index = i + (j + m)*Nx
                            indexs.append(index)

        self.score += score            
        self.availables.extend(indexs)

    def update(self):
        '''
        Return the current status of board, is_end?
        '''
        self.release()
        self.clear_board()
    
        flag = (len(self.availables) == 0)
        return flag

    def play(self, action):  
        start_index, end_index = action
        Nx, Ny = self.board_shape

        start_position = (start_index%Nx, int(start_index/Nx))
        end_position = (end_index%Nx, int(end_index/Nx))

        if end_index in self.availables:
            self.availables.remove(end_index)
        if start_index not in self.availables:
            self.availables.append(start_index)

        self.board[start_position], self.board[end_position] = self.board[end_position], self.board[start_position]
        flag = self.update() 

        # TODO Judge whether this operation is valid: A possible tranport path exist?

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
        self.action_probs = list()
        self.scores = list()

        self.ui = None

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
            if self.flag:
                self.end_ui()
            else:
                board = self.gameboard.get_board()
                score = self.gameboard.get_score()
                self.ui.setboard(board)
                self.ui.setscore(score)

    def start(self):
        '''
        Start to play "Five or more" game for human
        '''
        Nx, Ny, channel = self.state_shape
        self.gameboard = Board(board_shape=(Nx, Ny))
        self.gameboard.init()

        if self.verbose:
            print("Initiating UI...")

        from ui import UI, init_ui
        app = init_ui()
        self.ui = UI(pressaction=self.pressaction, board=self.gameboard.get_board(), sizeunit=50)
        app.exec_()

    def end_ui(self):
        score = self.gameboard.get_score()
        self.ui.gameend(score)

    def pressaction_ai(self, code):
        if code == (-1, -1): # press space
            # Evaluate the play vector
            state = self.get_state()
            availables = self.gameboard.get_availables()
            action = self.ai.play(state, availables) 
            if self.verbose:
                print("AI action: [{0}]".format(action))
            
            self.flag = self.gameboard.play(action=action)

            # Game end?
            if self.flag:
                self.end_ui()
            else:
                board = self.gameboard.get_board()
                score = self.gameboard.get_score()
                self.ui.setboard(board)
                self.ui.setscore(score)
        
    def start_ai(self):
        '''
        Start to play "Five or more" game for human
        '''
        Nx, Ny, channel = self.state_shape
        self.gameboard = Board(board_shape=(Nx, Ny))
        self.gameboard.init()

        board = self.gameboard.get_board()
        self.boards.append(board)
        self.update_states()

        state = self.get_state()
        availables = self.gameboard.get_availables()
        action = self.ai.play(state, availables)

        if self.verbose:
            print("Initiating UI...")

        from ui import UI, init_ui
        app = init_ui()
        self.ui = UI(pressaction=self.pressaction_ai, board=self.gameboard.get_board(), sizeunit=50)
        app.exec_()

    def start_selfplay(self, epsilon, gamma, beta=0.1):
        '''
        Start self-play process to get train data for AI model
        '''
        Nx, Ny, channel = self.state_shape
        self.gameboard = Board(board_shape=(Nx, Ny))
        self.gameboard.init()

        board = self.gameboard.get_board()
        self.boards.append(board)

        while not self.flag:
            self.update_states()
            state = self.get_state()

            availables = self.gameboard.get_availables()

            v = np.random.random()
            if v > epsilon:
                # Sample an action randomly to explore the hidden or potential approaches
                action_prob = np.zeros(2*Nx*Ny)
                chess_availables = [index for index in range(Nx*Ny) if index not in availables]
                start_index = np.random.choice(chess_availables)
                action_prob[start_index] = 1
                end_index = np.random.choice(availables)
                action_prob[Nx*Ny + end_index] = 1
                action = start_index, end_index
            else:
                # Perform an action based on deep learning model
                action_prob = self.ai.evaluate_function(state, availables)
                
                action = self.ai.play(state, availables) 

            self.action_probs.append(action_prob)   
            self.actions.append(action)

            score = self.gameboard.get_score()
            self.scores.append(score)
            self.flag = self.gameboard.play(action)
            
            board = self.gameboard.get_board()
            self.boards.append(board)

        N = len(self.states)
        Nx, Ny, channel = self.state_shape

        states = self.states
        current_states = states
        next_states = states[1:]
        next_states.append(None)
        actions = list()
        for i in range(N):
            start_index, end_index = self.actions[i]
            action = np.zeros(2*Nx*Ny)
            action[start_index] = 1
            action[Nx*Ny + end_index] = 1
            actions.append(action)

        dataset = current_states, actions, next_states
        intrinsic_rewards = self.ai.get_intrinsic_rewards(dataset)
        external_rewards = self.scores

        action_probs = list()
        for i in range(N):
            if i == 0:
                previous_reward =  0
            else:
                previous_reward = external_rewards[i-1] + intrinsic_rewards[i-1]
            if i < N-1:
                current_reward = external_rewards[i] + intrinsic_rewards[i]
            else:
                current_reward = 0
            # Define the improvement based on the difference of rewards
            reward = current_reward - previous_reward

            start_index, end_index = self.actions[i]
            action_prob = self.action_probs[i]

            # Define Q value with help of logistic function, restrict it into the range [0,1]
            Q_value = gamma/(1 + np.exp(-beta*reward))

            eps = 1e-12
            action_prob[start_index] += Q_value
            action_prob[start_index] = action_prob[start_index]/(np.sum(action_prob[start_index]) + eps)
            action_prob[Nx*Ny + end_index] += Q_value
            action_prob[Nx*Ny + end_index] = action_prob[Nx*Ny + end_index]/(np.sum(action_prob[Nx*Ny + end_index]) + eps)

            action_probs.append(action_prob)

        return self.states, action_probs
