import tensorflow as tf
from tensorflow import keras
import numpy as np
import pdb
import pickle

class PointNet(keras.Model):
    def __init__(self):
        super(PointNet, self).__init__()
        self.fc1 = keras.layers.Dense(3,activation='relu');
        self.fc2 = keras.layers.Dense(64,activation='relu');
        # self.fc3 = keras.layers.Dense(64,activation='relu');
        self.fc_feature = keras.layers.Dense(512,activation='relu');
        self.maxpooling = keras.layers.GlobalMaxPool1D()
        self.fc4 = keras.layers.Dense(128,activation='relu');
        self.fc5 = keras.layers.Dense(256,activation='relu');
        self.drop1 = keras.layers.Dropout(0.1);
        self.drop3 = keras.layers.Dropout(0.3);
        self.logist = keras.layers.Dense(3,activation='linear');
        #self.logist = keras.layers.Dense(4,activation='linear'); # pressure, velocity

    def call(self, inputs):
        mdata,pvdata = inputs
        ## proc mdata
        mout = self.fc1(mdata)
        mout = self.fc2(mout)
        # mout = self.fc3(mout)
        mout = self.fc_feature(mout)
        mout = tf.expand_dims(mout, 0)
        mout = self.maxpooling(mout)

        ## proc pvdata
        pvout = self.fc1(pvdata)
        pvout = self.fc2(pvout)
        # pvout = self.fc3(pvout)
        pvout = self.fc4(pvout)
        ##
        mout = tf.broadcast_to(mout,[pvout.shape[0],mout.shape[1]])
        out = tf.concat([pvout,mout],axis=-1)
        # out = pvout+mout
        # out = self.drop3(out)
        out = self.fc5(out)
        # out = self.drop1(out)
        out = self.logist(out)

        return out
