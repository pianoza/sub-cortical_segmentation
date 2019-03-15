import os, sys, ConfigParser
import threading
import nibabel as nib
from cnn_cort.load_options import *
from keras.utils import np_utils 
from cnn_cort.base import load_data, generate_training_set, testing
from cnn_cort.keras_net import get_callbacks, get_model
from tkinter import *
from tkinter.ttk import *


CURRENT_PATH = os.getcwd()
user_config = ConfigParser.RawConfigParser()

user_config.read(os.path.join(CURRENT_PATH, 'configuration.cfg'))
options = load_options(user_config)

weights_save_path = os.path.join(CURRENT_PATH, 'nets', options['experiment'])

TRANSFER_LEARNING = True
FEATURE_WEIGHTS_FILE = '/home/kaisar/Documents/PhD/source_codes/sub-cortical_segmentation/nets/weights_for_init/model.h5'


class TrainTask(threading.Thread):

    def __init__(self, queue, log):
        threading.Thread.__init__(self)
        self.queue = queue
        self.log = log

    def run(self):
        if not os.path.exists(os.path.join(weights_save_path, 'model.h5')):

            # get data patches from all orthogonal views 
            x_axial, x_cor, x_sag, y, x_atlas, names = load_data(options)

            # build the training dataset
            x_train_axial, x_train_cor, x_train_sag, x_train_atlas, y_train = generate_training_set(x_axial,
                                                                                                    x_cor,
                                                                                                    x_sag,
                                                                                                    x_atlas,
                                                                                                    y,
                                                                                                    options)

            model = get_model(dropout_rate=0.3, transfer_learning=TRANSFER_LEARNING, feature_weights_file=FEATURE_WEIGHTS_FILE)
            model.compile(optimizer='adam', loss=['categorical_crossentropy'], metrics=['accuracy'])
            model.summary()
            os.makedirs(weights_save_path)
            model.fit([x_train_axial, x_train_cor, x_train_sag, x_train_atlas],
                      [np_utils.to_categorical(y_train, num_classes=15)],
                      validation_split=options['train_split'],
                      epochs=options['max_epochs'],
                      batch_size=options['batch_size'],
                      verbose=options['net_verbose'],
                      shuffle=True,
                      callbacks=get_callbacks(options, weights_save_path))
            # model.save_weights(os.path.join(weights_save_path, 'model.h5'))
            model.save(os.path.join(weights_save_path, 'model.h5'))
            print 'Model saved in {}'.format(weights_save_path)


class TestTask(threading.Thread):

    def __init__(self, queue, modelPath, testPath, log, configs):
        threading.Thread.__init__(self)
        self.queue = queue
        self.modelPath = modelPath
        self.testPath = testPath
        self.log = log
        self.eT1Name = configs[0]
        self.eGTName = configs[1]
        self.eOutputName = configs[2]

    def run(self):
        self.log.insert(END, 'TESTING MODE\n')
        print self.modelPath.get()
        #TODO: THREAD WORKAROUND WITH GRAPH.AS_DEFAULT()
        model = get_model(0.0, False, self.modelPath.get())
        model.summary(print_fn=lambda x: self.log.insert(END, x + '\n'))
        options['t1_name'] = self.eT1Name.get()
        options['roi_name'] = self.eGTName.get()
        options['out_name'] = self.eOutputName.get()
        options['test_folder'] = self.testPath.get()
        testing(options, model)





