import os, sys, ConfigParser
import threading
import nibabel as nib
from cnn_cort.load_options import *
from keras.utils import np_utils 
from cnn_cort.base import load_data, generate_training_set, testing
from cnn_cort.keras_net import get_callbacks, get_model
from tkinter import *
from tkinter.ttk import *
import tensorflow as tf


CURRENT_PATH = os.getcwd()
user_config = ConfigParser.RawConfigParser()

user_config.read(os.path.join(CURRENT_PATH, 'configuration.cfg'))
options = load_options(user_config)

weights_save_path = os.path.join(CURRENT_PATH, 'nets', options['experiment'])

TRANSFER_LEARNING = True
FEATURE_WEIGHTS_FILE = '/home/kaisar/Documents/PhD/source_codes/sub-cortical_segmentation/nets/weights_for_init/model.h5'

graph = tf.get_default_graph()


class TrainTask(threading.Thread):

    def __init__(self, queue, trainPath, savePath, modelName, savedModel, configs):
        threading.Thread.__init__(self)
        self.queue = queue
        self.trainPath = trainPath
        self.savePath = savePath
        self.modelName = modelName
        self.savedModel = savedModel
        self.transfer_learning = False
        self.feature_weights_file = None
        if len(self.savedModel.get()) > 0:
            self.transfer_learning = True
            self.feature_weights_file = self.savedModel.get()
        self.eT1Name = configs[0]
        self.eGTName = configs[1]
        self.eOutputName = configs[2]

    def run(self):
        self.queue.put("Training", block=True)
        print 'TRAINING MODE'
        options['train_folder'] = self.trainPath.get()
        options['experiment'] = self.modelName.get()
        options['net_verbose'] = 2  # have to set the verbose to one-line-per-epoch to avoid long log history in GUI
        # get data patches from all orthogonal views 
        x_axial, x_cor, x_sag, y, x_atlas, names = load_data(options)

        # build the training dataset
        x_train_axial, x_train_cor, x_train_sag, x_train_atlas, y_train = generate_training_set(x_axial,
                                                                                                x_cor,
                                                                                                x_sag,
                                                                                                x_atlas,
                                                                                                y,
                                                                                                options)

        saveFolder = os.path.join(self.savePath.get(), self.modelName.get())
        if not os.path.exists(saveFolder):
            os.makedirs(saveFolder)

        with graph.as_default():
            model = get_model(dropout_rate=0.3, transfer_learning=self.transfer_learning, feature_weights_file=self.feature_weights_file)
            model.compile(optimizer='adam', loss=['categorical_crossentropy'], metrics=['accuracy'])
            model.summary()  # print_fn=lambda x: self.log.insert(END, x + '\n'))
            model.fit([x_train_axial, x_train_cor, x_train_sag, x_train_atlas],
                      [np_utils.to_categorical(y_train, num_classes=15)],
                      validation_split=options['train_split'],
                      epochs=options['max_epochs'],
                      batch_size=options['batch_size'],
                      verbose=options['net_verbose'],
                      shuffle=True,
                      callbacks=get_callbacks(options, saveFolder))
            model.save(os.path.join(saveFolder, 'model.h5'))

        print 'Model saved in {}'.format(saveFolder)
        print 'FINISHED'
        self.queue.get("Training")


class TestTask(threading.Thread):

    def __init__(self, queue, modelPath, testPath, configs):
        threading.Thread.__init__(self)
        self.queue = queue
        self.modelPath = modelPath
        self.testPath = testPath
        self.eT1Name = configs[0]
        self.eGTName = configs[1]
        self.eOutputName = configs[2]

    def run(self):
        self.queue.put("Testing", block=True)
        print 'TESTING MODE'
        with graph.as_default():
            model = get_model(0.0, False, self.modelPath.get())
            model.summary()  # print_fn=lambda x: self.log.insert(END, x + '\n'))
            options['t1_name'] = self.eT1Name.get()
            options['roi_name'] = self.eGTName.get()
            options['out_name'] = self.eOutputName.get()
            options['test_folder'] = self.testPath.get()
            testing(options, model)
            print 'FINISHED'
            self.queue.get("Testing")


