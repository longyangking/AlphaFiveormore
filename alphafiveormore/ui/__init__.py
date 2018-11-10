# The main class for User Interface

import numpy as np 
import threading
from . import nativeUI

import sys
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication

class UI(threading.Thread):
    def __init__(self, pressaction, board, sizeunit=50):
        threading.Thread.__init__(self)
        
        self.app = None
        self.board = board
        self.sizeunit = sizeunit
        self.pressaction = pressaction

        self.UI = None   

    def run(self):
        self.app = QApplication(sys.argv)
        self.UI = nativeUI.nativeUI(pressaction=self.pressaction,board=self.board,sizeunit=self.sizeunit)
        self.app.exec_()

    def setboard(self,board):
        while self.UI is None:
            pass
        self.UI.setboard(board=board)
    
    def gameend(self,score):
        while self.UI is None:
            pass
        self.UI.gameend(score=score)
