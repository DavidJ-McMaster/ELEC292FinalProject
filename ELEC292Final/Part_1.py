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

dj_data_walking = pd.read_csv(dj_walking)
dj_data_jumping = pd.read_csv(dj_jumping)
isabel_data_walking = pd.read_csv(isabel_walking)
isabel_data_jumping = pd.read_csv(isabel_jumping)
lizzy_data_walking = pd.read_csv(lizzy_walking)
lizzy_data_jumping = pd.read_csv(lizzy_jumping)

dj_walking_segment = segmentation(dj_data_walking)
dj_jumping_segment = segmentation(dj_data_jumping)
isabel_walking_segment = segmentation(isabel_data_walking)
isabel_jumping_segment = segmentation(isabel_data_jumping)
lizzy_walking_segment = segmentation(lizzy_data_walking)
lizzy_jumping_segment = segmentation(lizzy_data_jumping)

# shuffle everyone's data
random.shuffle(dj_walking_segment)
random.shuffle(dj_jumping_segment)
random.shuffle(isabel_walking_segment)
random.shuffle(isabel_jumping_segment)
random.shuffle(lizzy_walking_segment)
random.shuffle(lizzy_jumping_segment)

dj_walking_df = pd.concat([pd.DataFrame(segment) for segment in dj_walking_segment], ignore_index=True)
isabel_walking_df = pd.concat([pd.DataFrame(segment) for segment in isabel_walking_segment], ignore_index=True)
lizzy_walking_df = pd.concat([pd.DataFrame(segment) for segment in lizzy_walking_segment], ignore_index=True)


dj_jumping_df = pd.concat([pd.DataFrame(segment) for segment in dj_jumping_segment], ignore_index=True)
isabel_jumping_df = pd.concat([pd.DataFrame(segment) for segment in isabel_jumping_segment], ignore_index=True)
lizzy_jumping_df = pd.concat([pd.DataFrame(segment) for segment in lizzy_jumping_segment], ignore_index=True)

#dj_jumping_df = pd.DataFrame(dj_jumping_segment)
#isabel_jumping_df = pd.DataFrame(isabel_jumping_segment)
#lizzy_jumping_df = pd.DataFrame(lizzy_jumping_segment)


# everyone walking and everyone jumping
everyone_walking = pd.concat([dj_walking_df, isabel_walking_df, lizzy_walking_df], ignore_index=True)
everyone_jumping = pd.concat([dj_jumping_df, isabel_jumping_df, lizzy_jumping_df], ignore_index=True)

everyone_walking.to_csv("all_walking.csv")
everyone_jumping.to_csv("all_jumping.csv")

walking_training, walking_testing = train_test_split(everyone_walking, test_size=0.1, shuffle=True, random_state=5)
jumping_training, jumping_testing = train_test_split(everyone_jumping, test_size=0.1, shuffle=True, random_state=5)

# now need to store in HDF5 file
# need to verify that this is right and works
# for right now whatever

with h5py.File("data.h5", 'w') as hdf_file:

    Member1 = hdf_file.create_group("Member1") # dj
    Member2 = hdf_file.create_group("Member2") # isabel
    Member3 = hdf_file.create_group("Member3") # lizzy

    Member1.create_dataset("walking", data=dj_walking)
    Member1.create_dataset("jumping", data=dj_jumping)

    Member2.create_dataset("walking", data=isabel_walking)
    Member2.create_dataset("jumping", data=isabel_jumping)

    Member3.create_dataset("walking", data=lizzy_walking)
    Member3.create_dataset("jumping", data=lizzy_jumping)


