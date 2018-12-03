# The main class for User Interface

import numpy as np 
import threading
from . import nativeUI

import sys
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication

def init_ui():
    return QApplication(sys.argv)

class UI:
    def __init__(self, pressaction, board, sizeunit=50):
        self.board = board
        self.sizeunit = sizeunit
        self.pressaction = pressaction

        self.UI = nativeUI.nativeUI(pressaction=self.pressaction,board=self.board,sizeunit=self.sizeunit)  

    def setboard(self,board):
        while self.UI is None:
            pass
        self.UI.setboard(board=board)
    
    def gameend(self, score):
        while self.UI is None:
            pass
        self.UI.gameend(score=score)
