# ------------------------------------------------------------
# Training script example for Keras implementation
# 
# Kaisar Kushibar (2018)
# kaisar.kushibar@udg.edu
# ------------------------------------------------------------
import os, sys, ConfigParser
import nibabel as nib
from cnn_cort.load_options import *
from keras.utils import np_utils 


CURRENT_PATH = os.getcwd()

# --------------------------------------------------
# 1. load options from config file. Options are set
#    the configuration.cfg file 
# --------------------------------------------------

user_config = ConfigParser.RawConfigParser()

user_config.read(os.path.join(CURRENT_PATH, 'configuration.cfg'))
options = load_options(user_config)

from cnn_cort.base import load_data, generate_training_set, testing
from cnn_cort.keras_net import get_callbacks, get_model

# --------------------------------------------------
# set experiment file. If exists, testing goes on
# --------------------------------------------------
weights_save_path = os.path.join(CURRENT_PATH, 'nets', options['experiment'])

TRANSFER_LEARNING = True
FEATURE_WEIGHTS_FILE = '/home/kaisar/Documents/PhD/source_codes/sub-cortical_segmentation/nets/weights_for_init/model.h5'

# --------------------------------------------------
# train the net
# --------------------------------------------------
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
else:
# --------------------------------------------------
# test the model (for each scan)
# --------------------------------------------------
    print '\nTESTING MODE\n'
    model = get_model(0.0, False, os.path.join(weights_save_path, 'model.h5'))
    model.summary()
    testing(options, model)




