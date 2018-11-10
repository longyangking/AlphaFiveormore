# The native user interface powered by PyQt5

import numpy as np 
import sys
from PyQt5.QtWidgets import QWidget, QApplication,QDesktopWidget
from PyQt5.QtCore import * 
from PyQt5.QtGui import *

class nativeUI(QWidget):
    playsignal = pyqtSignal(tuple) 

    def __init__(self,pressaction,board,sizeunit=50,role="Human"):
        super(nativeUI,self).__init__(None)
        self.board = board
        self.board_shape = board.shape
        self.sizeunit = sizeunit
        self.R = 0.4*sizeunit

        self.mousex = 0
        self.mousey = 0

        self.chooseX = 0
        self.chooseY = 0

        self.start_position = 0
        self.end_position = 0

        self.has_picked = False
        self.isgameend = False

        self.pressaction = pressaction

        self.playsignal.connect(self.pressaction) 
        self.initUI()

    def getboard(self):
        return self.board

    def setboard(self,board):
        self.board = board
        self.update()

    def initUI(self):
        (Nx,Ny) = self.board.shape
        screen = QDesktopWidget().screenGeometry()
        size =  self.geometry()

        self.setGeometry((screen.width()-size.width())/2, 
                        (screen.height()-size.height())/2,
                        Nx*self.sizeunit, Ny*self.sizeunit)
        self.setWindowTitle("Five or More")
        self.setWindowIcon(QIcon('./ui/icon.png'))

        # set Background color
        palette =  QPalette()
        palette.setColor(self.backgroundRole(), QColor(255, 255, 255))
        self.setPalette(palette)

        self.setMouseTracking(True)
        self.show()

    def paintEvent(self, e):
        qp = QPainter()
        qp.begin(self)
        self.drawBoard(qp)
        self.drawChesses(qp)
        self.chooseChess(qp)
        if self.isgameend:
            self.drawgameend(qp)
        qp.end()

    def gameend(self,winner):
        self.isgameend = True
        self.winner = winner

    def drawgameend(self,qp):
        size =  self.geometry()
        qp.setPen(0)
        qp.setBrush(QColor(200, 200, 200, 180))
        width = size.width()/5*4
        height = size.height()/3
        qp.drawRect(size.width()/2-width/2, size.height()/2-height/2, width, height)

        qp.setPen(QColor(0,0,0))
        font = qp.font()
        font.setPixelSize(60)
        qp.setFont(font)
        qp.drawText(QRect(size.width()/2-width/2, size.height()/2-height/2, width, height),	0x0004|0x0080,str(self.winner + " Win"))

    def mouseMoveEvent(self,e):
        self.mousex = int(e.x()/self.sizeunit)
        self.mousey = int(e.y()/self.sizeunit)
        self.update() 
    
    def mousePressEvent(self,e):
        X = int(e.x()/self.sizeunit)
        Y = int(e.y()/self.sizeunit)
        if (self.board[X,Y] != 0) and not self.has_picked:
            self.chooseX = X
            self.chooseY = Y

            self.start_position = (X, Y)
            self.has_picked = True
            self.update()

        if  self.has_picked and (self.board[X,Y] == 0):
            self.end_position = (X, Y)

            (Nx,Ny) = self.board.shape
            start_index = self.start_position[0] + Nx*self.start_position[1]
            end_index = self.end_position[0] + Nx*self.end_position[1]
            self.playsignal.emit((start_index, end_index))
            self.has_picked = False
            self.update()

    def chooseChess(self,qp):
        #qp.setBrush(QColor(0, 0, 0))
        qp.setPen(QColor(0, 255, 255))
        qp.setBrush(0)
        qp.drawEllipse((self.mousex+0.5)*self.sizeunit-self.R,
                (self.mousey+0.5)*self.sizeunit-self.R, 
                2*self.R, 2*self.R)

        if self.has_picked:
            x, y = self.start_position
            qp.setPen(QColor(0, 0, 0))
            qp.setBrush(0)
            qp.drawEllipse((x+0.5)*self.sizeunit-self.R,
                    (y+0.5)*self.sizeunit-self.R, 
                    2*self.R, 2*self.R)

    def drawBoard(self,qp):
        (Nx,Ny) = self.board.shape
        qp.setPen(QColor(0, 0, 0))
        for i in range(Nx):
            qp.drawLine((i+0.5)*self.sizeunit, 0, (i+0.5)*self.sizeunit,Ny*self.sizeunit)   
        for j in range(Ny):
            qp.drawLine(0, (j+0.5)*self.sizeunit, Ny*self.sizeunit, (j+0.5)*self.sizeunit) 

    def drawChesses(self, qp):
        (Nx,Ny) = self.board.shape
        qp.setPen(0)
        for i in range(Nx):
            for j in range(Ny):
                if self.board[i,j] == 1:
                    qp.setBrush(QColor(255, 0, 0))
                elif self.board[i,j] == 2:
                    qp.setBrush(QColor(0, 0, 255))
                elif self.board[i,j] == 3:
                    qp.setBrush(QColor(0, 255, 0))
                elif self.board[i,j] == 4:
                    qp.setBrush(QColor(255, 0, 255))
                elif self.board[i,j] == 5:
                    qp.setBrush(QColor(255, 255, 0))

                if self.board[i,j] != 0:
                    qp.drawEllipse((i+0.5)*self.sizeunit-self.R, (j+0.5)*self.sizeunit-self.R, 2*self.R, 2*self.R)
