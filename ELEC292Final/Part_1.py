import pandas as pd
import numpy as np
import h5py
import pylab as pl
import matplotlib.pyplot as plt
from scipy.stats import mode, kurtosis, skew, t
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import random

# this file reads the data, segments it and splits it testing/training


def segmentation(signal, window_size=5, overlap=0):
    segments = []
    for start in range(0, len(signal), window_size-overlap):
        end = start + window_size
        segment = signal[start:end]
        if len(segment) == window_size:
            segments.append(segment)
    return segments


dj_walking = "dj_WalkingRawData.csv"
dj_jumping = "dj_JumpingRawData.csv"

isabel_walking = "Isabel_Walking_RawData2.csv"
isabel_jumping = "Isabel_Jumping_RawData4.csv"

lizzy_walking = "lizzy_NewWalkingData.csv"
lizzy_jumping = "lizzy_NewJumping.csv"

# segment into 5 second intervals

dj_walking_segment = segmentation(dj_walking)
dj_jumping_segment = segmentation(dj_jumping)
isabel_walking_segment = segmentation(isabel_walking)
isabel_jumping_segment = segmentation(isabel_jumping)
lizzy_walking_segment = segmentation(lizzy_walking)
lizzy_jumping_segment = segmentation(lizzy_jumping)

# shuffle everyone's data
random.shuffle(dj_walking_segment)
random.shuffle(dj_jumping_segment)
random.shuffle(isabel_walking_segment)
random.shuffle(isabel_jumping_segment)
random.shuffle(lizzy_walking_segment)
random.shuffle(lizzy_jumping_segment)

# everyone walking and everyone jumping
everyone_walking = []
everyone_walking.extend(dj_walking_segment)
everyone_walking.extend(isabel_walking_segment)
everyone_walking.extend(lizzy_walking_segment)

everyone_jumping = []
everyone_jumping.extend(dj_jumping_segment)
everyone_jumping.extend(isabel_jumping_segment)
everyone_jumping.extend(lizzy_jumping_segment)

walking_training, walking_testing = train_test_split(everyone_walking, test_size=0.1, shuffle=True, random_state=5)
jumping_training, jumping_testing = train_test_split(everyone_jumping, test_size=0.1, shuffle=True, random_state=5)


