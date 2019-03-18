# ------------------------------------------------------------
# Training script example for Keras implementation
# 
# Kaisar Kushibar (2019)
# kaisar.kushibar@udg.edu
# ------------------------------------------------------------
import os
import sys
import numpy as np
from functools import partial
from tkinter import filedialog
from tkinter import *
from tkinter.ttk import *

import Queue

import ConfigParser
import nibabel as nib
from cnn_cort.load_options import *
from keras.utils import np_utils 

CURRENT_PATH = os.getcwd()
user_config = ConfigParser.RawConfigParser()
user_config.read(os.path.join(CURRENT_PATH, 'configuration.cfg'))
options = load_options(user_config)

from cnn_cort.base import load_data, generate_training_set, testing
from cnn_cort.keras_net import get_callbacks, get_model

from train_test_task import TestTask, TrainTask


class StdoutRedirector(object):
    def __init__(self, tLog):
        self.log = tLog

    def write(self, msg):
        self.log.insert(END, msg)
        self.log.see(END)

    def flush(self):
        self.log.see(END)


class Application(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        self.pack()
        self.show_main_window()
        self.isProcessing = False
        self.queue = Queue.Queue()


    def get_label_entry_button(self, parent, row, labelTitle, buttonTitle):
        label = Label(parent, text=labelTitle)
        entry = Entry(parent)
        button = Button(parent, text=buttonTitle)
        label.grid(row=row, column=0, sticky=W+E+N+S, padx=5, pady=5)
        entry.grid(row=row, column=1, sticky=W+E+N+S, padx=5, pady=5)
        button.grid(row=row, column=2, sticky=W+E+N+S, padx=5, pady=5)
        parent.grid_columnconfigure(1, weight=2)

        return entry, button


    def get_label_entry(self, parent, row, labelTitle, entryText):
        label = Label(parent, text=labelTitle)
        entry = Entry(parent)
        entry.insert(0, entryText)
        label.grid(row=row, column=0, sticky=W+E+N+S, ipadx=5, ipady=5, padx=5, pady=5)
        entry.grid(row=row, column=1, sticky=W+E+N+S, ipadx=5, ipady=5, padx=5, pady=5)
        parent.grid_columnconfigure(1, weight=2)

        return entry


    def tab_train_test(self, parent):
        # create widgets
        pWindow = PanedWindow(parent, orient='vertical')
        
        # Train frame
        fTrain = LabelFrame(pWindow, text='Train')

        eTrainFolder, bTrainFolder = self.get_label_entry_button(fTrain, 0, 'Train path:', 'Browse')
        bTrainFolder.config(command=partial(self.select_path, eTrainFolder, 'DIR'))

        eModelFolder, bModelFolder = self.get_label_entry_button(fTrain, 1, 'Pre-trained model:', 'Browse')
        bModelFolder.config(command=partial(self.select_path, eModelFolder, 'FILE'))

        eSaveFolder, bSaveFolder = self.get_label_entry_button(fTrain, 2, 'Save path:', 'Browse')
        bSaveFolder.config(command=partial(self.select_path, eSaveFolder, 'DIR'))

        eModelName = self.get_label_entry(fTrain, 3, 'Save model name:', '')

        bStartTrain = Button(fTrain, text='Start training')
        bStartTrain.grid(row=4, columnspan=3, sticky=N+S, padx=5, pady=5)
        
        # Log frame
        fLog = LabelFrame(pWindow, text='Log')
        sLog = Scrollbar(fLog)
        tLog = Text(fLog, wrap=WORD, yscrollcommand=sLog.set, bg="#000000", fg='#42f450', borderwidth=0, highlightthickness=0)
        sLog.config(command=tLog.yview)
        sLog.pack(side=RIGHT, fill=Y)
        tLog.pack(side=LEFT, fill=BOTH, expand=True)
        
        # Redirect all standard outputs to the tLog
        sys.stdout = StdoutRedirector(tLog)

        # Test frame, appears before Log frame, but need the log screen declared first
        fTest = LabelFrame(pWindow, text='Test')
        eTestFolder, bTestFolder = self.get_label_entry_button(fTest, 0, 'Test path:', 'Browse')
        bTestFolder.config(command=partial(self.select_path, eTestFolder, 'DIR'))
        eTestModel, bTestModel = self.get_label_entry_button(fTest, 1, 'Model:', 'Browse')
        bTestModel.config(command=partial(self.select_path, eTestModel, 'FILE'))
        bStartTest = Button(fTest, text='Start testing', command=partial(self.start_testing, eTestModel, eTestFolder))
        bStartTest.grid(row=2, columnspan=3, sticky=N+S, padx=5, pady=5)
        
        bStartTrain.config(command=partial(self.start_training, eTrainFolder, eSaveFolder, eModelName, eModelFolder))

        # set window title
        self.winfo_toplevel().title('Brain sub-cortical structure segmentation tool')
        # set geometry
        pWindow.add(fTrain, weight=33)
        pWindow.add(fTest, weight=33)
        pWindow.add(fLog, weight=33)
        pWindow.pack(fill=BOTH, expand=True)
        return pWindow

    
    def start_training(self, trainPath, savePath, modelName, savedModel):
        if self.queue.empty():
            TrainTask(self.queue, trainPath, savePath, modelName, savedModel, [self.eT1Name, self.eGTName, self.eOutputName]).start()
        else:
            print 'WAITING FOR A PROCESS TO FINISH...'

    def start_testing(self, modelPath, testPath):
        if self.queue.empty():
            TestTask(self.queue, modelPath, testPath, [self.eT1Name, self.eGTName, self.eOutputName]).start()
        else:
            print 'WAITING FOR A PROCESS TO FINISH...'


    def tab_config(self, parent):
        pWindow = PanedWindow(parent, orient='vertical')
        frame = Frame(pWindow)
        self.eT1Name = self.get_label_entry(frame, 0, 'T1 name', 'T1.nii.gz')
        self.eGTName = self.get_label_entry(frame, 1, 'Ground truth name', 'ground_truth.nii.gz')
        self.eOutputName = self.get_label_entry(frame, 2, 'Output file name', 'seg_out.nii.gz')
        pWindow.add(frame)
        pWindow.pack(side=TOP, fill=BOTH, expand=True)
        return pWindow


    def show_main_window(self):
        # tabs
        nMainWindow = Notebook(self.master, width=700, height=650)
        nMainWindow.add(self.tab_train_test(nMainWindow), text='Train and test')
        nMainWindow.add(self.tab_config(nMainWindow), text='Configurations')
        nMainWindow.pack(fill=BOTH, expand=True, ipadx=10, ipady=10)


    def select_path(self, entry, mode):
        name = ''
        if mode == 'DIR':
            name = filedialog.askdirectory(title='Select path')
        if mode == 'FILE':
            name = filedialog.askopenfilename(title='Select pre-trained model', filetypes=(('HDF5 files', '*.h5'), ("all files","*.*")))
        if mode == 'SAVE':
            name = filedialog.asksaveasfilename(title='Save trained model', filetypes=(('HDF5 files', '*.h5'), ("all files","*.*")))
        entry.delete(0, END)
        entry.insert(0, name)


root = Tk()
app = Application(master=root)
app.mainloop()



