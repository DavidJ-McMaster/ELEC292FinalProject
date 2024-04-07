import pandas as pd
import numpy as np
import h5py
import pylab as pl
import matplotlib.pyplot as plt
from scipy.stats import mode, kurtosis, skew, t
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, roc_auc_score, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import random
from sklearn.pipeline import make_pipeline

from Part_1 import *
from Part_2 import *
from Part_3 import *

# classifier
'''
walking_training, walking_testing = train_test_split(everyone_walking, test_size=0.1, shuffle=True, random_state=5)
jumping_training, jumping_testing = train_test_split(everyone_jumping, test_size=0.1, shuffle=True, random_state=5)

scaler = StandardScaler()
l_reg = LogisticRegression(max_iter=10000)
clf = make_pipeline(StandardScaler(), l_reg)
clf.fit(walking_training, jumping_training)
jumping_pred = clf.predict(walking_testing)
jumping_clf_prob = clf.predict_proba(walking_testing)
acc = accuracy_score(jumping_testing, jumping_pred)
print('accuracy is: ', acc)
recall = recall_score(jumping_testing, jumping_pred)
print('recall is: ', recall)
cm = confusion_matrix(jumping_testing, jumping_pred)
cm_display = ConfusionMatrixDisplay(cm).plot()
plt.show()

'''

from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Assuming 'Label' column indicates the activity (walking or jumping)
walking_data = everyone_walking.drop(columns=['Label'])
walking_labels = everyone_walking['Label']
jumping_data = everyone_jumping.drop(columns=['Label'])
jumping_labels = everyone_jumping['Label']

# Train-test split
walking_training, walking_testing, walking_labels_training, walking_labels_testing = train_test_split(walking_data, walking_labels, test_size=0.1, shuffle=True, random_state=5)
jumping_training, jumping_testing, jumping_labels_training, jumping_labels_testing = train_test_split(jumping_data, jumping_labels, test_size=0.1, shuffle=True, random_state=5)

# Classifier pipeline
scaler = StandardScaler()
l_reg = LogisticRegression(max_iter=10000)
clf = make_pipeline(StandardScaler(), l_reg)

# Fit the classifier
clf.fit(walking_training, walking_labels_training)

# Predictions
jumping_pred = clf.predict(walking_testing)

# Model evaluation
acc = accuracy_score(jumping_labels_testing, jumping_pred)
print('Accuracy:', acc)

recall = recall_score(jumping_labels_testing, jumping_pred)
print('Recall:', recall)

cm = confusion_matrix(jumping_labels_testing, jumping_pred)
cm_display = ConfusionMatrixDisplay(cm).plot()
plt.show()


