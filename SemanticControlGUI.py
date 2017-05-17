
# coding: utf-8

# In[1]:

## Imports and defines
# %pylab
import numpy as np
import sys

import cv2

import time
import glob
import datetime

import os

from PIL import Image
from PySide import QtCore, QtGui


import GraphWithValues as gwv
import VideoTexturesUtils as vtu
import SemanticsDefinitionTabGUI as sdt
import SemanticLoopingTabGUI as slt

app = QtGui.QApplication(sys.argv)

DICT_SEQUENCE_NAME = 'semantic_sequence_name'
DICT_BBOXES = 'bboxes'
DICT_FOOTPRINTS = 'footprints' ## same as bboxes but it indicates the footprint of the sprite on the ground plane
DICT_BBOX_ROTATIONS = 'bbox_rotations'
DICT_BBOX_CENTERS = 'bbox_centers'
DICT_FRAMES_LOCATIONS = 'frame_locs'
DICT_MASK_LOCATION = 'frame_masks_location'
DICT_ICON_TOP_LEFT = "icon_top_left"
DICT_ICON_FRAME_KEY = "icon_frame_key"
DICT_ICON_SIZE = "icon_size"
DICT_REPRESENTATIVE_COLOR = 'representative_color'
DICT_FRAME_SEMANTICS = "semantics_per_frame"
DICT_NUM_SEMANTICS = "number_of_semantic_classes"
DICT_PATCHES_LOCATION = "sequence_preloaded_patches_location"
DICT_TRANSITION_COSTS_LOCATION = "sequence_precomputed_transition_costs_location"
DICT_FRAME_COMPATIBILITY_LABELS = 'compatibiliy_labels_per_frame'
DICT_LABELLED_FRAMES = 'labelled_frames' ## includes the frames labelled for the semantic labels (the first [DICT_FRAME_SEMANTICS].shape[1])
DICT_NUM_EXTRA_FRAMES = 'num_extra_frames' ## same len as DICT_LABELLED_FRAMES
DICT_CONFLICTING_SEQUENCES = 'conflicting_sequences'
DICT_COMPATIBLE_SEQUENCES = 'compatible_sequences'
DICT_DISTANCE_MATRIX_LOCATION = 'sequence_precomputed_distance_matrix_location' ## for label propagation
DICT_SEQUENCE_LOCATION = "sequence_location"

DICT_USED_SEQUENCES = "used_semantic_sequences"
DICT_SEQUENCE_INSTANCES = "sequence_instances"
DICT_SEQUENCE_BG = "sequence_background_image"

DICT_SEQUENCE_FRAMES = 'sequence_frames'
DICT_SEQUENCE_IDX = 'semantic_sequence_idx' # index of the instantiated sem sequence in the list of all used sem sequences for a synthesised sequence
DICT_DESIRED_SEMANTICS = 'desired_semantics' # stores what the desired semantics are for a certain sprite 
#(I could index them by the frame when the toggle happened instead of using the below but maybe ordering is important and I would lose that using a dict)
DICT_FRAME_SEMANTIC_TOGGLE = 'frame_semantic_toggle'# stores the frame index in the generated sequence when the desired semantics have changed
DICT_OFFSET = "instance_offset"
DICT_SCALE = "instance_scale"

GRAPH_MAX_COST = 10000000.0

TL_IDX = 0
TR_IDX = 1
BR_IDX = 2
BL_IDX = 3

DO_SAVE_LOGS = False


# In[2]:

class FormattedKeystrokeLabel(QtGui.QLabel):
    def __init__(self, text="", parent=None):
        super(FormattedKeystrokeLabel, self).__init__(text, parent)
        
        self.extraSpace = 3
        
        
        self.wordsToRender = []
        self.wordsAreBold = []
        self.wordsWidths = []
        self.spaceWidth = QtGui.QFontMetrics(QtGui.QFont()).width(" ")
        self.wordHeight = QtGui.QFontMetrics(QtGui.QFont()).height()
        totalWidth = 0
        
        ## getting words from input
        for word in text.split(" ") :
            if "<b>" in word :
                self.wordsAreBold.append(True)
                self.wordsToRender.append("".join(("".join(word.split("<b>"))).split("</b>")))
            else :
                self.wordsAreBold.append(False)
                self.wordsToRender.append(word)
                
            font = QtGui.QFont()
            if self.wordsAreBold[-1] :
                font.setWeight(QtGui.QFont.Bold)
            else :
                font.setWeight(QtGui.QFont.Normal)
            
            self.wordsWidths.append(QtGui.QFontMetrics(font).width(self.wordsToRender[-1]))
            totalWidth += self.wordsWidths[-1]
            
        totalWidth += self.spaceWidth*(len(self.wordsToRender)-1)
        
        ## resize label
        self.setFixedSize(totalWidth+self.extraSpace*2, self.wordHeight+self.extraSpace*2)
        
    def paintEvent(self, event) :
        painter = QtGui.QPainter(self)
        padding = 1
        
        currentX = self.extraSpace
        for word, isBold, wordWidth in zip(self.wordsToRender, self.wordsAreBold, self.wordsWidths) :
            wordRect = QtCore.QRect(currentX, self.extraSpace, wordWidth, self.wordHeight)
            if word != "or" and word != "to" :
                ## draw rectangle
                painter.setBrush(QtGui.QBrush(QtGui.QColor.fromRgb(225, 225, 225)))
                painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(0, 0, 0, 0), 0, 
                                                  QtCore.Qt.SolidLine, QtCore.Qt.SquareCap, QtCore.Qt.MiterJoin))

                painter.drawRect(QtCore.QRect(wordRect.left()-padding, wordRect.top()-padding,
                                              wordRect.width()+padding*2, wordRect.height()+padding*2))
                
            ## draw text
            painter.setBrush(QtGui.QBrush(QtGui.QColor.fromRgb(0, 0, 0)))
            painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(0, 0, 0, 255), 3, 
                                              QtCore.Qt.SolidLine, QtCore.Qt.SquareCap, QtCore.Qt.MiterJoin))

            if isBold :
                font = QtGui.QFont()
                font.setWeight(QtGui.QFont.Bold)
                painter.setFont(font)
            else :
                painter.setFont(QtGui.QFont())

            painter.drawText(wordRect, word)
                
            currentX += (self.spaceWidth + wordWidth)
        painter.end()


# In[3]:

class HelpDialog(QtGui.QDialog):
    def __init__(self, parent=None, title=""):
        super(HelpDialog, self).__init__(parent)
        
        self.createGUI()
        
        self.setWindowTitle(title)
        
    def doneClicked(self):
        self.done(0)
    
    def createGUI(self):
        
        self.doneButton = QtGui.QPushButton("Done")
         
        ## SIGNALS ##
        
        self.doneButton.clicked.connect(self.doneClicked)
        
        ## LAYOUTS ##
        
        mainLayout = QtGui.QGridLayout()
        
        idx = 0
        mainLayout.addWidget(QtGui.QLabel("<b>Definition</b>"), idx, 1, 1, 1, QtCore.Qt.AlignLeft)
        mainLayout.addWidget(QtGui.QLabel("<b>Synthesis</b>"), idx, 4, 1, 1, QtCore.Qt.AlignLeft);idx+=1
        
        mainLayout.addWidget(FormattedKeystrokeLabel("Ctrl <b>Return</b>"), idx, 0, 1, 1, QtCore.Qt.AlignRight)
        mainLayout.addWidget(QtGui.QLabel("Track Forward"), idx, 1, 1, 1, QtCore.Qt.AlignLeft)
        mainLayout.addWidget(FormattedKeystrokeLabel("<b>u</b>"), idx, 3, 1, 1, QtCore.Qt.AlignRight)
        mainLayout.addWidget(QtGui.QLabel("Move selected instance up in the list"), idx, 4, 1, 1, QtCore.Qt.AlignLeft)
        idx+=1
        
        mainLayout.addWidget(FormattedKeystrokeLabel("Ctrl <b>Backspace</b>"), idx, 0, 1, 1, QtCore.Qt.AlignRight)
        mainLayout.addWidget(QtGui.QLabel("Track Backwards"), idx, 1, 1, 1, QtCore.Qt.AlignLeft)
        mainLayout.addWidget(FormattedKeystrokeLabel("<b>d</b>"), idx, 3, 1, 1, QtCore.Qt.AlignRight)
        mainLayout.addWidget(QtGui.QLabel("Move selected instance down in the list"), idx, 4, 1, 1, QtCore.Qt.AlignLeft)
        idx+=1
        
        mainLayout.addWidget(FormattedKeystrokeLabel("<b>Escape</b>"), idx, 0, 1, 1, QtCore.Qt.AlignRight)
        mainLayout.addWidget(QtGui.QLabel("Stop tracking or batch segmentation"), idx, 1, 1, 1, QtCore.Qt.AlignLeft)
        mainLayout.addWidget(FormattedKeystrokeLabel("<b>0</b> to <b>9</b>"), idx, 3, 1, 1, QtCore.Qt.AlignRight)
        mainLayout.addWidget(QtGui.QLabel("Request given semantics for selected instance"), idx, 4, 1, 1, QtCore.Qt.AlignLeft)
        idx+=1
        
        mainLayout.addWidget(FormattedKeystrokeLabel("<b>Delete</b>"), idx, 0, 1, 1, QtCore.Qt.AlignRight)
        mainLayout.addWidget(QtGui.QLabel("Delete current frame's bounding box"), idx, 1, 1, 1, QtCore.Qt.AlignLeft)
        mainLayout.addWidget(FormattedKeystrokeLabel("<b>r</b>"), idx, 3, 1, 1, QtCore.Qt.AlignRight)
        mainLayout.addWidget(QtGui.QLabel("Refine synthesised sequence"), idx, 4, 1, 1, QtCore.Qt.AlignLeft)
        idx+=1
        
        mainLayout.addWidget(FormattedKeystrokeLabel("<b>Enter</b>"), idx, 0, 1, 1, QtCore.Qt.AlignRight)
        mainLayout.addWidget(QtGui.QLabel("Set bounding box for current frame"), idx, 1, 1, 1, QtCore.Qt.AlignLeft)
        mainLayout.addWidget(FormattedKeystrokeLabel("<b>Space</b>"), idx, 3, 1, 1, QtCore.Qt.AlignRight)
        mainLayout.addWidget(QtGui.QLabel("Extend synthesised sequence from the <b>end</b>"), idx, 4, 1, 1, QtCore.Qt.AlignLeft)
        idx+=1
        
        mainLayout.addWidget(FormattedKeystrokeLabel("Ctrl <b>c</b>"), idx, 0, 1, 1, QtCore.Qt.AlignRight)
        mainLayout.addWidget(QtGui.QLabel("Copy current bounding box"), idx, 1, 1, 1, QtCore.Qt.AlignLeft)
        mainLayout.addWidget(FormattedKeystrokeLabel("Shift <b>Space</b>"), idx, 3, 1, 1, QtCore.Qt.AlignRight)
        mainLayout.addWidget(QtGui.QLabel("Extend synthesised sequence from the <b>current</b> frame"), idx, 4, 1, 1, QtCore.Qt.AlignLeft)
        idx+=1
        
        mainLayout.addWidget(FormattedKeystrokeLabel("Ctrl <b>v</b>"), idx, 0, 1, 1, QtCore.Qt.AlignRight)
        mainLayout.addWidget(QtGui.QLabel("Paste current bounding box"), idx, 1, 1, 1, QtCore.Qt.AlignLeft)
        mainLayout.addWidget(FormattedKeystrokeLabel("Ctrl <b>Space</b>"), idx, 3, 1, 1, QtCore.Qt.AlignRight)
        mainLayout.addWidget(QtGui.QLabel("Extend synthesised sequence of selected sequences from the <b>current</b> frame"), idx, 4, 1, 1, QtCore.Qt.AlignLeft)
        idx+=1
        
        mainLayout.addWidget(FormattedKeystrokeLabel("Ctrl <b>s</b>"), idx, 0, 1, 1, QtCore.Qt.AlignRight)
        mainLayout.addWidget(QtGui.QLabel("Save tracked sprites"), idx, 1, 1, 1, QtCore.Qt.AlignLeft)
        mainLayout.addWidget(FormattedKeystrokeLabel("(Shift) <b>t</b>"), idx, 3, 1, 1, QtCore.Qt.AlignRight)
        mainLayout.addWidget(QtGui.QLabel("Tag (compatibility) incompatiblity between <b>2</b> instances or frame of <b>1</b> selected instance"), idx, 4, 1, 1, QtCore.Qt.AlignLeft)
        idx+=1
        
        mainLayout.addWidget(FormattedKeystrokeLabel("<b>m</b>"), idx, 0, 1, 1, QtCore.Qt.AlignRight)
        mainLayout.addWidget(QtGui.QLabel("Switch mode (<font color=\"red\"><b>bbox</b></font> vs <font color=\"blue\"><b>scribble</b></font>)"), idx, 1, 1, 1, QtCore.Qt.AlignLeft)
        mainLayout.addWidget(FormattedKeystrokeLabel("<b>Delete</b>"), idx, 3, 1, 1, QtCore.Qt.AlignRight)
        mainLayout.addWidget(QtGui.QLabel("Delete currently selected instance"), idx, 4, 1, 1, QtCore.Qt.AlignLeft)
        idx+=1
        
        mainLayout.addWidget(FormattedKeystrokeLabel("<b>r</b>"), idx, 0, 1, 1, QtCore.Qt.AlignRight)
        mainLayout.addWidget(QtGui.QLabel("Rename Action"), idx, 1, 1, 1, QtCore.Qt.AlignLeft)
        mainLayout.addWidget(FormattedKeystrokeLabel("Ctrl <b>s</b>"), idx, 3, 1, 1, QtCore.Qt.AlignRight)
        mainLayout.addWidget(QtGui.QLabel("Save synthesised sequence"), idx, 4, 1, 1, QtCore.Qt.AlignLeft)
        idx+=1
        
        mainLayout.addWidget(FormattedKeystrokeLabel("<b>-</b>"), idx, 0, 1, 1, QtCore.Qt.AlignRight)
        mainLayout.addWidget(QtGui.QLabel("Delete example frame"), idx, 1, 1, 1, QtCore.Qt.AlignLeft)
        mainLayout.addWidget(FormattedKeystrokeLabel("<b>a</b>"), idx, 3, 1, 1, QtCore.Qt.AlignRight)
        mainLayout.addWidget(QtGui.QLabel("Add new instance"), idx, 4, 1, 1, QtCore.Qt.AlignLeft)
        idx+=1
        
        mainLayout.addWidget(FormattedKeystrokeLabel("<b>k</b>"), idx, 3, 1, 1, QtCore.Qt.AlignRight)
        mainLayout.addWidget(QtGui.QLabel("Change action command bindings for all actor sequences"), idx, 4, 1, 1, QtCore.Qt.AlignLeft)
        idx+=1
        
        verticalLine =  QtGui.QFrame()
        verticalLine.setFrameStyle(QtGui.QFrame.VLine)
        verticalLine.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        mainLayout.addWidget(verticalLine, 0, 2, idx, 1)
        
        horizontalLine =  QtGui.QFrame()
        horizontalLine.setFrameStyle(QtGui.QFrame.HLine)
        horizontalLine.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        mainLayout.addWidget(horizontalLine,idx, 0 , 1, 5)
        idx+=1
        
        mainLayout.addWidget(self.doneButton, idx, 0, 1, 5, QtCore.Qt.AlignCenter)
        idx+=1
        
        self.setLayout(mainLayout)

def showHelp(parent=None, title="Keyboard Shortcuts") :
    helpDialog = HelpDialog(parent, title)
    exitCode = helpDialog.exec_()
    
    return exitCode


# In[4]:

class Window(QtGui.QMainWindow):
    def __init__(self):
        super(Window, self).__init__()
        
        if DO_SAVE_LOGS :
            if not os.path.isdir("logFiles/") :
                os.mkdir("logFiles/")

            print "LOG:Starting", datetime.datetime.now()
            with open("logFiles/log-"+str(datetime.datetime.now()), "w+") as f:
                f.write("LOG:DEFINITION:Switch-&-" + str(datetime.datetime.now()) + "\n")
        
        if os.path.isfile("semantic_control_recent_loads.npy") :
            self.recentLoadedFiles = np.load("semantic_control_recent_loads.npy").item()
        else :
            self.recentLoadedFiles = {'raw_sequences':[], 'synthesised_sequences':[]}
        
        self.createGUI()
        
        self.showLoading(False)
        
        self.setWindowTitle("Action-based Video Synthesis")
        self.resize(1920, 950)
        
        self.readyForVT = False
        self.firstLoad = True
        self.dataLocation = ""
        self.semanticsDefinitionTab.setFocus()
    
    def openSequence(self) :
        return 
        
    def tabChanged(self, tabIdx) :
        if tabIdx == 0 :
            self.semanticsDefinitionTab.setFocus()
            
            ##
            if DO_SAVE_LOGS :
                with open(np.sort(glob.glob("logFiles/log-*"))[-1], "a+") as f :
                    f.write("LOG:DEFINITION:Switch-&-" + str(datetime.datetime.now()) +"\n")
                
        elif tabIdx == 1 :
            self.semanticLoopingTab.setFocus()
            
            ##
            if DO_SAVE_LOGS :
                with open(np.sort(glob.glob("logFiles/log-*"))[-1], "a+") as f :
                    f.write("LOG:LOOPING:Switch-&-" + str(datetime.datetime.now()) +"\n")

    def closeEvent(self, event) :
        self.semanticsDefinitionTab.cleanup()
        self.semanticLoopingTab.cleanup()
        
        
        ##
        if DO_SAVE_LOGS :
            with open(np.sort(glob.glob("logFiles/log-*"))[-1], "a+") as f :
                f.write("LOG:Closing-&-" + str(datetime.datetime.now()) +"\n")
            
    def showLoading(self, show) :
        if show :
            self.loadingLabel.setText("Loading... Please wait")
            self.loadingWidget.setVisible(True)
            self.infoLabel.setVisible(False)
        else :
            self.loadingWidget.setVisible(False)
            self.infoLabel.setVisible(True)

    def showHelpDialog(self) :
        showHelp(self)
        
    def loadRawSequencePressed(self, triggeredAction) :
        if triggeredAction.iconText() == "Find Location on Disk" :
            newLocation = self.semanticsDefinitionTab.loadFrameSequencePressed()
            if newLocation != "" :
                if len(self.recentLoadedFiles['raw_sequences']) > 9 :
                    del self.recentLoadedFiles['raw_sequences'][9]
                self.recentLoadedFiles['raw_sequences'].insert(0, newLocation)
                np.save("semantic_control_recent_loads.npy", self.recentLoadedFiles)
        else :
            savedLoc = triggeredAction.iconText()
            self.recentLoadedFiles['raw_sequences'] = [i for i in self.recentLoadedFiles['raw_sequences'] if i != savedLoc]
            self.recentLoadedFiles['raw_sequences'].insert(0, savedLoc)
            np.save("semantic_control_recent_loads.npy", self.recentLoadedFiles)
            
            self.semanticsDefinitionTab.loadFrameSequence(savedLoc)
        self.setRecentLoadedLists()
        
    def loadSynthesisedSequencePressed(self, triggeredAction) :
        if triggeredAction.iconText() == "Find Location on Disk" :
            newLocation = self.semanticLoopingTab.loadSynthesisedSequence()
            if newLocation != "" :
                if len(self.recentLoadedFiles['synthesised_sequences']) > 9 :
                    del self.recentLoadedFiles['synthesised_sequences'][9]
                self.recentLoadedFiles['synthesised_sequences'].insert(0, newLocation)
                np.save("semantic_control_recent_loads.npy", self.recentLoadedFiles)
        else :
            savedLoc = triggeredAction.iconText()
            self.recentLoadedFiles['synthesised_sequences'] = [i for i in self.recentLoadedFiles['synthesised_sequences'] if i != savedLoc]
            self.recentLoadedFiles['synthesised_sequences'].insert(0, savedLoc)
            np.save("semantic_control_recent_loads.npy", self.recentLoadedFiles)
            
            self.semanticLoopingTab.loadSynthesisedSequenceAtLocation(savedLoc)
        self.setRecentLoadedLists()
            
    def setRecentLoadedLists(self) :
        self.loadRawFrameSequenceMenu.clear()
        self.loadRawFrameSequenceMenu.addAction("Find Location on Disk")
        self.loadRawFrameSequenceMenu.addSeparator()
        for location in self.recentLoadedFiles['raw_sequences'] :
            self.loadRawFrameSequenceMenu.addAction(location)
            
        self.loadSynthesisedSequenceMenu.clear()
        self.loadSynthesisedSequenceMenu.addAction("Find Location on Disk")
        self.loadSynthesisedSequenceMenu.addSeparator()
        for location in self.recentLoadedFiles['synthesised_sequences'] :
            self.loadSynthesisedSequenceMenu.addAction(location)
        
        
    def createGUI(self) :
        
        ## WIDGETS ##

        self.infoLabel = QtGui.QLabel("No data loaded")
        self.infoLabel.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        self.infoLabel.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        
        self.openSequenceButton = QtGui.QPushButton("Open &Sequence")
        
        self.loadingLabel = QtGui.QLabel("Loading... Please wait!")
        self.loadingLabel.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        self.loadingLabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignTop)
        movie = QtGui.QMovie("loader.gif")
        self.loadingSpinner = QtGui.QLabel()
        self.loadingSpinner.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        self.loadingSpinner.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignBottom)
        self.loadingSpinner.setMovie(movie)
        movie.start()
        
        self.semanticsDefinitionTab = sdt.SemanticsDefinitionTab("", self)
        self.semanticLoopingTab = slt.SemanticLoopingTab(80, "", True, self)

        self.tabWidget = QtGui.QTabWidget()
        self.tabWidget.addTab(self.semanticsDefinitionTab, self.tr("Define Actor Sequences"))
        self.tabWidget.addTab(self.semanticLoopingTab, self.tr("Action-based Synthesis"))
        
        if True :
            self.tabWidget.setCurrentIndex(0)
            self.semanticsDefinitionTab.setFocus()
        else :
            self.tabWidget.setCurrentIndex(1)
            self.semanticLoopingTab.setFocus()
        
        ## SIGNALS ##
        
        self.openSequenceButton.clicked.connect(self.openSequence)
        
        self.tabWidget.currentChanged.connect(self.tabChanged)
        
        ## LAYOUTS ##
        
        self.mainBox = QtGui.QGroupBox("Main Controls")
        self.mainBox.setStyleSheet("QGroupBox { margin: 5px; border: 2px groove gray; border-radius: 3px; } QGroupBox::title {left: 15px; top: -7px; font: bold;}")
        mainBoxLayout = QtGui.QHBoxLayout()
        
        self.loadingWidget = QtGui.QWidget()
        loadingLayout = QtGui.QHBoxLayout()
        loadingLayout.addWidget(self.loadingSpinner)
        loadingLayout.addWidget(self.loadingLabel)
        self.loadingWidget.setLayout(loadingLayout)
        
        mainBoxLayout.addWidget(self.loadingWidget)
        mainBoxLayout.addWidget(self.infoLabel)
        mainBoxLayout.addStretch()
        
        buttonLayout = QtGui.QVBoxLayout()
        buttonLayout.addWidget(self.openSequenceButton)
        
        mainBoxLayout.addLayout(buttonLayout)
        self.mainBox.setLayout(mainBoxLayout)
        
#         mainLayout = QtGui.QVBoxLayout()
#         mainLayout.addWidget(self.tabWidget)
#         mainLayout.addWidget(mainBox)
        
        self.setCentralWidget(self.tabWidget)
        
        ## MENU ACTIONS ##
#         loadRawFrameSequenceAction = QtGui.QAction("Load &Raw Frame Sequence", self)
#         loadRawFrameSequenceAction.triggered.connect(self.loadRawSequencePressed)
        self.loadRawFrameSequenceMenu = QtGui.QMenu("Load &Raw Frame Sequence", self)
        self.loadRawFrameSequenceMenu.triggered.connect(self.loadRawSequencePressed)
        
        synthesiseNewSequenceAction = QtGui.QAction("Synthesise &New Sequence", self)
        synthesiseNewSequenceAction.triggered.connect(self.semanticLoopingTab.newSynthesisedSequence)
        self.loadSynthesisedSequenceMenu = QtGui.QMenu("Load &Synthesised Sequence", self)
        self.loadSynthesisedSequenceMenu.triggered.connect(self.loadSynthesisedSequencePressed)
        
        self.setRecentLoadedLists()
        
        
        loadInputSequenceAction = QtGui.QAction("Load &Actor Sequence", self)
        loadInputSequenceAction.triggered.connect(self.semanticLoopingTab.loadSemanticSequence)
        setBackgroundImageAction = QtGui.QAction("Set &Background Image", self)
        setBackgroundImageAction.triggered.connect(self.semanticLoopingTab.setBgImage)
        
        helpAction = QtGui.QAction("&Help", self)
        helpAction.setShortcut('Ctrl+H')
        helpAction.triggered.connect(self.showHelpDialog)
    
        ## MENU BAR ##
        fileMenu = self.menuBar().addMenu("&File")
        fileMenu.addMenu(self.loadRawFrameSequenceMenu)
        fileMenu.addSeparator()
        fileMenu.addAction(synthesiseNewSequenceAction)
        fileMenu.addMenu(self.loadSynthesisedSequenceMenu)
        fileMenu.addAction(loadInputSequenceAction)
        fileMenu.addAction(setBackgroundImageAction)
        
        
        aboutMenu = self.menuBar().addMenu("&About")
        aboutMenu.addAction(helpAction)
        


# In[8]:

# %%capture
def main():
    window = Window()
    window.show()
    app.exec_()
    del window

if __name__ == "__main__":
    main()


# In[12]:

# logLocation = np.sort(glob.glob("logFiles/log-*"))[-1]
# # logLocation = "/media/ilisescu/UUI/Semantic Control/logFiles/log-2016-04-05 14_06_44.540685"
# # logLocation = "/home/ilisescu/PhD/data/synthesisedSequences/USER STUDIES SEQUENCES/aron/task_log"
# # logLocation = "/home/ilisescu/PhD/iPy/logFiles/havana_bus_sequence_white_bus2_distmatcompute_log"

# with open(logLocation) as f :
#     allLines = f.readlines()
    
#     timeSpentInTabs = [[], []]
#     listOfSpritesInDefinition = {}
#     isDoingTracking = True
#     currentSprite = ""
    
#     for line in allLines :
#         if "\n" in line :
#             line = line[:-2]
            
#         action, timestamp = line.split("-&-")
#         timeOfAction = np.array(timestamp.split(" ")[-1].split(":"), float)
#         print action, timestamp.split(" ")[-1]
#         if "DEFINITION:Switch" in action :
#             isDefinitionTab = True
#             timeSpentInTabs[0].append([timeOfAction, timeOfAction])
            
#             if len(timeSpentInTabs[1]) > 0 :
#                 timeSpentInTabs[1][-1][-1] = timeOfAction
                
#             if currentSprite != "" and currentSprite in listOfSpritesInDefinition.keys() :
#                 if isDoingTracking :
#                     listOfSpritesInDefinition[currentSprite]["tracking"].append([timeOfAction, timeOfAction])
#                 else :
#                     listOfSpritesInDefinition[currentSprite]["segmenting"].append([timeOfAction, timeOfAction])
                
#         elif "LOOPING:Switch" in action :
#             isDefinitionTab = False
#             timeSpentInTabs[1].append([timeOfAction, timeOfAction])
            
#             if len(timeSpentInTabs[0]) > 0 :
#                 timeSpentInTabs[0][-1][-1] = timeOfAction
                
#             if currentSprite != "" and currentSprite in listOfSpritesInDefinition.keys() :
#                 if isDoingTracking :
#                     listOfSpritesInDefinition[currentSprite]["tracking"][-1][-1] = timeOfAction
#                 else :
#                     listOfSpritesInDefinition[currentSprite]["segmenting"][-1][-1] = timeOfAction
                
            
#         if isDefinitionTab :
#             if "Selecting" in action :
#                 if currentSprite != "" and currentSprite in listOfSpritesInDefinition.keys() :
#                     if isDoingTracking :
#                         listOfSpritesInDefinition[currentSprite]["tracking"][-1][-1] = timeOfAction
#                     else :
#                         listOfSpritesInDefinition[currentSprite]["segmenting"][-1][-1] = timeOfAction
                        
#                 currentSprite = action.split(":")[-1].split(" ")[-1]
                
#             if currentSprite != "" :
#                 if currentSprite not in listOfSpritesInDefinition :
#                     listOfSpritesInDefinition[currentSprite] = {}
                
#                 if "Selecting" in action :
#                     if isDoingTracking :
#                         if "tracking" not in listOfSpritesInDefinition[currentSprite] :
#                             listOfSpritesInDefinition[currentSprite]["tracking"] = []
                            
#                         listOfSpritesInDefinition[currentSprite]["tracking"].append([timeOfAction, timeOfAction])
#                     else :
#                         if "segmenting" not in listOfSpritesInDefinition[currentSprite] :
#                             listOfSpritesInDefinition[currentSprite]["segmenting"] = []
                            
#                         listOfSpritesInDefinition[currentSprite]["segmenting"].append([timeOfAction, timeOfAction])
                        
#                 if "Start Segmenting" in action :
#                     if "tracking" in listOfSpritesInDefinition[currentSprite].keys() :
#                         listOfSpritesInDefinition[currentSprite]["tracking"][-1][-1] = timeOfAction
#                     isDoingTracking = False
                    
#                     if "segmenting" not in listOfSpritesInDefinition[currentSprite] :
#                         listOfSpritesInDefinition[currentSprite]["segmenting"] = []
                            
#                     listOfSpritesInDefinition[currentSprite]["segmenting"].append([timeOfAction, timeOfAction])
                    
#                 if "Start Tracking" in action :
#                     if "segmenting" in listOfSpritesInDefinition[currentSprite].keys() :
#                         listOfSpritesInDefinition[currentSprite]["segmenting"][-1][-1] = timeOfAction
#                     isDoingTracking = False
                    
#                     if "tracking" not in listOfSpritesInDefinition[currentSprite] :
#                         listOfSpritesInDefinition[currentSprite]["tracking"] = []
                    
#                     listOfSpritesInDefinition[currentSprite]["tracking"].append([timeOfAction, timeOfAction])
                
#         else :
#             print "nothing to do"
            
#         if "Closing" in action :
#             if isDefinitionTab :
#                 if currentSprite != "" :
#                     if isDoingTracking :
#                         listOfSpritesInDefinition[currentSprite]["tracking"][-1][-1] = timeOfAction
#                     else :
#                         listOfSpritesInDefinition[currentSprite]["segmenting"][-1][-1] = timeOfAction
                
#                 if len(timeSpentInTabs[0]) > 0 :
#                     timeSpentInTabs[0][-1][-1] = timeOfAction
#             else :
#                 if len(timeSpentInTabs[1]) > 0 :
#                     timeSpentInTabs[1][-1][-1] = timeOfAction
        
            
# print
# print "---------------------- STATISTICS ----------------------"
# for spriteKey in listOfSpritesInDefinition.keys() :
#     print "SPRITE:", spriteKey
#     if "tracking" in listOfSpritesInDefinition[spriteKey].keys() :
#         totalTime = np.zeros(3)
#         for instance in listOfSpritesInDefinition[spriteKey]["tracking"] :
#             tmp = instance[1]-instance[0]
#             if tmp[1] < 0.0 :
#                 tmp[1] += 60.0
#                 tmp[0] -= 1.0
#             if tmp[2] < 0.0 :
#                 tmp[2] += 60.0
#                 tmp[1] -= 1.0
#             totalTime += tmp
#             if totalTime[1] >= 60.0 :
#                 totalTime[1] -= 60.0
#                 totalTime[0] += 1.0
#             if totalTime[2] >= 60.0 :
#                 totalTime[2] -= 60.0
#                 totalTime[1] += 1.0
# #             print instance#, instance[1]-instance[0], tmp
#         print "TRACKING TIME: {0} hours, {1} minutes, {2} seconds".format(totalTime[0], totalTime[1], totalTime[2])
#     else :
#         print "NO TRACKING"
        
#     if "segmenting" in listOfSpritesInDefinition[spriteKey].keys() :
#         totalTime = np.zeros(3)
#         for instance in listOfSpritesInDefinition[spriteKey]["segmenting"] :
#             tmp = instance[1]-instance[0]
#             if tmp[1] < 0.0 :
#                 tmp[1] += 60.0
#                 tmp[0] -= 1.0
#             if tmp[2] < 0.0 :
#                 tmp[2] += 60.0
#                 tmp[1] -= 1.0
#             totalTime += tmp
#             if totalTime[1] >= 60.0 :
#                 totalTime[1] -= 60.0
#                 totalTime[0] += 1.0
#             if totalTime[2] >= 60.0 :
#                 totalTime[2] -= 60.0
#                 totalTime[1] += 1.0
# #             print instance#, instance[1]-instance[0], tmp
#         print "SEGMENTATION TIME: {0} hours, {1} minutes, {2} seconds".format(totalTime[0], totalTime[1], totalTime[2])
#     else :
#         print "NO SEGMENTATION"
#     print
        
# totalTime = np.zeros(3)
# for instance in timeSpentInTabs[0] :
#     tmp = instance[1]-instance[0]
#     if tmp[1] < 0.0 :
#         tmp[1] += 60.0
#         tmp[0] -= 1.0
#     if tmp[2] < 0.0 :
#         tmp[2] += 60.0
#         tmp[1] -= 1.0
#     totalTime += tmp
#     if totalTime[1] >= 60.0 :
#         totalTime[1] -= 60.0
#         totalTime[0] += 1.0
#     if totalTime[2] >= 60.0 :
#         totalTime[2] -= 60.0
#         totalTime[1] += 1.0
# #     print instance#, instance[1]-instance[0], tmp
# print "TOTAL DEFINITION TIME: {0} hours, {1} minutes, {2} seconds".format(totalTime[0], totalTime[1], totalTime[2])

# totalTime = np.zeros(3)
# for instance in timeSpentInTabs[1] :
#     tmp = instance[1]-instance[0]
#     if tmp[1] < 0.0 :
#         tmp[1] += 60.0
#         tmp[0] -= 1.0
#     elif tmp[1] >= 60.0 :
#         tmp[1] -= 60.0
#         tmp[0] += 1.0
#     if tmp[2] < 0.0 :
#         tmp[2] += 60.0
#         tmp[1] -= 1.0
#     elif tmp[2] >= 60.0 :
#         tmp[2] -= 60.0
#         tmp[1] += 1.0
#     totalTime += tmp
# #     print instance#, instance[1]-instance[0], tmp
# print "TOTAL LOOPING TIME: {0} hours, {1} minutes, {2} seconds".format(totalTime[0], totalTime[1], totalTime[2])

