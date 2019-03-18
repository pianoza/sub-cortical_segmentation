import os
import numpy as np
import keras
from keras.models import Model, load_model
from keras.layers.core import Lambda
from keras.layers import Input, MaxPool2D, BatchNormalization, Dropout, Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.merge import Concatenate, Add, Multiply
from keras.layers.advanced_activations import PReLU
import keras.backend as K
from keras.optimizers import Adagrad, Adadelta, Adam, SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint

from tkinter import *
from tkinter.ttk import *

## extra imports to set GPU options
import tensorflow as tf
###################################
# TensorFlow wizardry
config = tf.ConfigProto()
# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True
# Create a session with the above options specified.
K.tensorflow_backend.set_session(tf.Session(config=config))
###################################
K.set_image_dim_ordering("th")


def get_callbacks(options, save_path):
    checkpoint = ModelCheckpoint(filepath=os.path.join(save_path, 'checkpoint.h5'))
    early_stopping = EarlyStopping(monitor='val_loss', patience=options['patience'])
    return [checkpoint, early_stopping]


def get_branch(prefix, dropout_rate=0.5):
    # input layer
    x_input = Input(shape=(1, 32, 32), dtype='float32', name=prefix+'_input')
    # shallow layers
    x = Conv2D(20, (3, 3), name=prefix+'_conv_1')(x_input)
    x = BatchNormalization(name=prefix+'_batch_norm_1')(x)
    x = PReLU(name=prefix+'_prelu_1')(x)
    x = Conv2D(20, (3, 3), name=prefix+'_conv_2')(x)
    x = BatchNormalization(name=prefix+'_batch_norm_2')(x)
    x = PReLU(name=prefix+'_prelu_2')(x)
    x = MaxPool2D((2, 2), name=prefix+'_max_pool_1')(x)
    x = Conv2D(40, (3, 3), name=prefix+'_conv_3')(x)
    x = BatchNormalization(name=prefix+'_batch_norm_3')(x)
    x = PReLU(name=prefix+'_prelu_3')(x)
    # deep layers
    x = Conv2D(40, (3, 3), name=prefix+'_conv_4')(x)
    x = BatchNormalization(name=prefix+'_batch_norm_4')(x)
    x = PReLU(name=prefix+'_prelu_4')(x)
    x = MaxPool2D((2, 2), name=prefix+'_max_pool_2')(x)
    x = Conv2D(60, (3, 3), name=prefix+'_conv_5')(x)
    x = BatchNormalization(name=prefix+'_batch_norm_5')(x)
    x = PReLU(name=prefix+'_prelu_5')(x)
    x = Dropout(dropout_rate, name=prefix+'_dropout_1')(x)
    x = Flatten(name=prefix+'_flatten')(x)
    x = Dense(180, name=prefix+'_dense_1')(x)
    x_output = PReLU(name=prefix+'_prelu_6')(x)
    return Model(name=prefix+'_branch', inputs=[x_input], outputs=[x_output])


def get_feature_model(dropout_rate=0.5):
    # all inputs
    x_input = Input(shape=(1, 32, 32), dtype='float32', name='feature_x_input')
    y_input = Input(shape=(1, 32, 32), dtype='float32', name='feature_y_input')
    z_input = Input(shape=(1, 32, 32), dtype='float32', name='feature_z_input')
    atlas_input = Input(shape=(15,), dtype='float32', name='feature_atlas_input')

    # axial
    x_branch = get_branch('axial', dropout_rate)
    y_branch = get_branch('sagittal', dropout_rate)
    z_branch = get_branch('coronal', dropout_rate)

    x = x_branch(x_input)
    y = y_branch(y_input)
    z = z_branch(z_input)

    # FC layer 540
    fc = Concatenate(name='concat_xyz')([x, y, z])
    fc = Dropout(dropout_rate, name='fc_drop_1')(fc)
    fc = Dense(540, name='fc_dense_1')(fc)
    fc = PReLU(name='fc_prelu_1')(fc)

    # concatenate channels 540 + 15
    fc = Dropout(dropout_rate, name='fc_drop_2')(fc)
    fc = Concatenate(name='concat_xyza')([fc, atlas_input])

    # FC layer 270
    fc = Dense(270, name='fc_dense_2')(fc)
    output = PReLU(name='fc_prelu_2')(fc)
    
    # Feature model
    model = Model(name='feature_model', inputs=[x_input, y_input, z_input, atlas_input], outputs=[output])

    return model


def print_nested_model(model, level):
    for i, l in enumerate(model.layers):
        if isinstance(l, Model):
            print_nested_model(l, level+1)
        else:
            print ' '*level*4, i, l.name, 'trainable' if l.trainable else 'not trainable'


def freeze_convs(model, level):
    for i, l in enumerate(model.layers):
        if isinstance(l, Model):
            freeze_convs(l, level+1)
        else:
            if 'conv' in l.name:
                l.trainable = False


def get_model(dropout_rate, transfer_learning, feature_weights_file):
    x_input= Input(shape=(1, 32, 32), dtype='float32', name='model_x_input')
    y_input= Input(shape=(1, 32, 32), dtype='float32', name='model_y_input')
    z_input= Input(shape=(1, 32, 32), dtype='float32', name='model_z_input')
    atlas_input= Input(shape=(15,), dtype='float32', name='model_atlas_input')

    feature_model = get_feature_model(dropout_rate)
    features = feature_model([x_input, y_input, z_input, atlas_input])

    classifier = Dense(15, name='model_softmax', activation='softmax')
    output = classifier(features)
    model = Model([x_input, y_input, z_input, atlas_input], [output], name='full_model')

    if feature_weights_file is not None:
        print 'Loading an existing model weights'
        # model.load_weights(feature_weights_file, by_name=True)
        model = load_model(feature_weights_file)

    if transfer_learning:
        print 'TRANSFER LEARNING'
        freeze_convs(model, 0)

    # print_nested_model(model, 0)
    return model


