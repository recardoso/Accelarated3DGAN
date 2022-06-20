from __future__ import print_function
import os

import glob

from collections import defaultdict
try:
    import cPickle as pickle
except ImportError:
    import pickle
    
import argparse
import sys
import h5py 
import numpy as np
import time
import math
import tensorflow as tf

import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adadelta, Adam, RMSprop
from tensorflow.keras.utils import Progbar

from tensorflow.compat.v1.keras.layers import BatchNormalization
from tensorflow.keras.layers import (Input, Dense, Reshape, Flatten, Lambda,Dropout, Activation, Embedding)
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import (UpSampling3D, Conv3D, ZeroPadding3D, AveragePooling3D)
from tensorflow.keras.models import Model, Sequential
import math

import json

#Divide files in train and test lists     
def DivideFiles(FileSearch="/data/LCD/*/*.h5", Fractions=[.9,.1],datasetnames=["ECAL","HCAL"],Particles=[],MaxFiles=-1):
    print ("Searching in :",FileSearch)
    Files =sorted( glob.glob(FileSearch))
    print ("Found {} files. ".format(len(Files)))
    FileCount=0
    Samples={}
    for F in Files:
        FileCount+=1
        basename=os.path.basename(F)
        ParticleName=basename.split("_")[0].replace("Escan","")
        if ParticleName in Particles:
            try:
                Samples[ParticleName].append(F)
            except:
                Samples[ParticleName]=[(F)]
        if MaxFiles>0:
            if FileCount>MaxFiles:
                break
    out=[]
    for j in range(len(Fractions)):
        out.append([])
    SampleI=len(Samples.keys())*[int(0)]
    for i,SampleName in enumerate(Samples):
        Sample=Samples[SampleName]
        NFiles=len(Sample)
        for j,Frac in enumerate(Fractions):
            EndI=int(SampleI[i]+ round(NFiles*Frac))
            out[j]+=Sample[SampleI[i]:EndI]
            SampleI[i]=EndI
    return out




def GetDataAngleParallel(dataset, xscale =1, xpower=1, yscale = 100, angscale=1, angtype='theta', thresh=1e-4, daxis=-1): 
    X=np.array(dataset.get('ECAL'))* xscale
    Y=np.array(dataset.get('energy'))/yscale
    X[X < thresh] = 0
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    ecal = np.sum(X, axis=(1, 2, 3))
    indexes = np.where(ecal > 10.0)
    X=X[indexes]
    Y=Y[indexes]
    if angtype in dataset:
        ang = np.array(dataset.get(angtype))[indexes]
    #else:  
      #ang = gan.measPython(X)
    X = np.expand_dims(X, axis=daxis)
    ecal=ecal[indexes]
    ecal=np.expand_dims(ecal, axis=daxis)
    if xpower !=1.:
        X = np.power(X, xpower)

    Y = [[el] for el in Y]
    ang = [[el] for el in ang]
    ecal = [[el] for el in ecal]

    final_dataset = {'X': X,'Y': Y, 'ang': ang, 'ecal': ecal}

    return final_dataset

def RetrieveTFRecord(recorddatapaths):
    recorddata = tf.data.TFRecordDataset(recorddatapaths)

    #print(type(recorddata))

    
    retrieveddata = {
        'ECAL': tf.io.FixedLenSequenceFeature((), dtype=tf.float32, allow_missing=True), #float32
        'ecalsize': tf.io.FixedLenSequenceFeature((), dtype=tf.int64, allow_missing=True), #needs size of ecal so it can reconstruct the narray
        #'bins': tf.io.FixedLenSequenceFeature((), dtype=tf.int64, allow_missing=True),
        'energy': tf.io.FixedLenSequenceFeature((), dtype=tf.float32, allow_missing=True), #float32
        'eta': tf.io.FixedLenSequenceFeature((), dtype=tf.float32, allow_missing=True),
        'mtheta': tf.io.FixedLenSequenceFeature((), dtype=tf.float32, allow_missing=True),
        'sum': tf.io.FixedLenSequenceFeature((), dtype=tf.float32, allow_missing=True),
        'theta': tf.io.FixedLenSequenceFeature((), dtype=tf.float32, allow_missing=True),
    }

    def _parse_function(example_proto):
        # Parse the input `tf.Example` proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, retrieveddata)

    parsed_dataset = recorddata.map(_parse_function)

    #return parsed_dataset

    #print(type(parsed_dataset))

    for parsed_record in parsed_dataset:
        dataset = parsed_record

    dataset['ECAL'] = tf.reshape(dataset['ECAL'], dataset['ecalsize'])

    dataset.pop('ecalsize')

    return dataset

def RetrieveTFRecordpreprocessing(recorddatapaths, batch_size):
    recorddata = tf.data.TFRecordDataset(recorddatapaths, num_parallel_reads=tf.data.experimental.AUTOTUNE)
    #recorddata = tf.data.TFRecordDataset(recorddatapaths)
    
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA


    retrieveddata = {
        'X': tf.io.FixedLenSequenceFeature((), dtype=tf.float32, allow_missing=True), #float32
        #'ecalsize': tf.io.FixedLenSequencFeature((), dtype=tf.int64, allow_missing=True), #needs size of ecal so it can reconstruct the narray
        'Y': tf.io.FixedLenFeature((), dtype=tf.float32, default_value=0.0), #float32
        'ang': tf.io.FixedLenFeature((), dtype=tf.float32, default_value=0.0),
        'ecal': tf.io.FixedLenFeature((), dtype=tf.float32, default_value=0.0),
    }

    def _parse_function(example_proto):
        # Parse the input `tf.Example` proto using the dictionary above.
        data = tf.io.parse_single_example(example_proto, retrieveddata)
        data['X'] = tf.reshape(data['X'],[1,51,51,25])
        #print(tf.shape(data['Y']))
        data['Y'] = tf.reshape(data['Y'],[1])
        data['ang'] = tf.reshape(data['ang'],[1])
        data['ecal'] = tf.reshape(data['ecal'],[1])
        #print(tf.shape(data['Y']))
        return data


    parsed_dataset = recorddata.map(_parse_function).repeat().batch(batch_size, drop_remainder=True).with_options(options)



    return parsed_dataset
    #return parsed_dataset, ds_size