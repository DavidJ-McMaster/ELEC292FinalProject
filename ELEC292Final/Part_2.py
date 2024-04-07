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
from Part_1 import *

#visualization

# Create separate plots for walking and jumping along the x, y, and z axes
# Create separate plots for walking and jumping along the x, y, and z axes
plt.figure(figsize=(16, 10))

# DJ walking plots
plt.subplot(2, 3, 1)
plt.plot(dj_walking_sorted['Time (s)'], dj_walking_sorted['Linear Acceleration x (m/s^2)'])
plt.title('Walking Linear Acceleration x - DJ')
plt.xlabel('Time (s)')
plt.ylabel('Linear Acceleration x (m/s^2)')

plt.subplot(2, 3, 2)
plt.plot(dj_walking_sorted['Time (s)'], dj_walking_sorted['Linear Acceleration y (m/s^2)'])
plt.title('Walking Linear Acceleration y - DJ')
plt.xlabel('Time (s)')
plt.ylabel('Linear Acceleration y (m/s^2)')

plt.subplot(2, 3, 3)
plt.plot(dj_walking_sorted['Time (s)'], dj_walking_sorted['Linear Acceleration z (m/s^2)'])
plt.title('Walking Linear Acceleration z - DJ')
plt.xlabel('Time (s)')
plt.ylabel('Linear Acceleration z (m/s^2)')

# DJ jumping plots
plt.subplot(2, 3, 4)
plt.plot(dj_jumping_sorted['Time (s)'], dj_jumping_sorted['Linear Acceleration x (m/s^2)'])
plt.title('Jumping Linear Acceleration x - DJ')
plt.xlabel('Time (s)')
plt.ylabel('Linear Acceleration x (m/s^2)')

plt.subplot(2, 3, 5)
plt.plot(dj_jumping_sorted['Time (s)'], dj_jumping_sorted['Linear Acceleration y (m/s^2)'])
plt.title('Jumping Linear Acceleration y - DJ')
plt.xlabel('Time (s)')
plt.ylabel('Linear Acceleration y (m/s^2)')

plt.subplot(2, 3, 6)
plt.plot(dj_jumping_sorted['Time (s)'], dj_jumping_sorted['Linear Acceleration z (m/s^2)'])
plt.title('Jumping Linear Acceleration z - DJ')
plt.xlabel('Time (s)')
plt.ylabel('Linear Acceleration z (m/s^2)')

plt.tight_layout()
plt.show()

# isabel

plt.figure(figsize=(16, 10))

# Isabel walking plots
plt.subplot(2, 3, 1)
plt.plot(isabel_walking_sorted['Time (s)'], isabel_walking_sorted['Linear Acceleration x (m/s^2)'])
plt.title('Walking Linear Acceleration x - Isabel')
plt.xlabel('Time (s)')
plt.ylabel('Linear Acceleration x (m/s^2)')

plt.subplot(2, 3, 2)
plt.plot(isabel_walking_sorted['Time (s)'], isabel_walking_sorted['Linear Acceleration y (m/s^2)'])
plt.title('Walking Linear Acceleration y - Isabel')
plt.xlabel('Time (s)')
plt.ylabel('Linear Acceleration y (m/s^2)')

plt.subplot(2, 3, 3)
plt.plot(isabel_walking_sorted['Time (s)'], isabel_walking_sorted['Linear Acceleration z (m/s^2)'])
plt.title('Walking Linear Acceleration z - Isabel')
plt.xlabel('Time (s)')
plt.ylabel('Linear Acceleration z (m/s^2)')

# DJ jumping plots
plt.subplot(2, 3, 4)
plt.plot(isabel_jumping_sorted['Time (s)'], isabel_jumping_sorted['Linear Acceleration x (m/s^2)'])
plt.title('Jumping Linear Acceleration x - Isabel')
plt.xlabel('Time (s)')
plt.ylabel('Linear Acceleration x (m/s^2)')

plt.subplot(2, 3, 5)
plt.plot(isabel_jumping_sorted['Time (s)'], isabel_jumping_sorted['Linear Acceleration y (m/s^2)'])
plt.title('Jumping Linear Acceleration y - Isabel')
plt.xlabel('Time (s)')
plt.ylabel('Linear Acceleration y (m/s^2)')

plt.subplot(2, 3, 6)
plt.plot(isabel_jumping_sorted['Time (s)'], isabel_jumping_sorted['Linear Acceleration z (m/s^2)'])
plt.title('Jumping Linear Acceleration z - Isabel')
plt.xlabel('Time (s)')
plt.ylabel('Linear Acceleration z (m/s^2)')

plt.tight_layout()
plt.show()


# lizzy
plt.figure(figsize=(16, 10))

# DJ walking plots
plt.subplot(2, 3, 1)
plt.plot(lizzy_walking_sorted['Time (s)'], lizzy_walking_sorted['Linear Acceleration x (m/s^2)'])
plt.title('Walking Linear Acceleration x - DJ')
plt.xlabel('Time (s)')
plt.ylabel('Linear Acceleration x (m/s^2)')

plt.subplot(2, 3, 2)
plt.plot(lizzy_walking_sorted['Time (s)'], lizzy_walking_sorted['Linear Acceleration y (m/s^2)'])
plt.title('Walking Linear Acceleration y - DJ')
plt.xlabel('Time (s)')
plt.ylabel('Linear Acceleration y (m/s^2)')

plt.subplot(2, 3, 3)
plt.plot(lizzy_walking_sorted['Time (s)'], lizzy_walking_sorted['Linear Acceleration z (m/s^2)'])
plt.title('Walking Linear Acceleration z - DJ')
plt.xlabel('Time (s)')
plt.ylabel('Linear Acceleration z (m/s^2)')

# DJ jumping plots
plt.subplot(2, 3, 4)
plt.plot(lizzy_jumping_sorted['Time (s)'], lizzy_jumping_sorted['Linear Acceleration x (m/s^2)'])
plt.title('Jumping Linear Acceleration x - DJ')
plt.xlabel('Time (s)')
plt.ylabel('Linear Acceleration x (m/s^2)')

plt.subplot(2, 3, 5)
plt.plot(lizzy_jumping_sorted['Time (s)'], lizzy_jumping_sorted['Linear Acceleration y (m/s^2)'])
plt.title('Jumping Linear Acceleration y - DJ')
plt.xlabel('Time (s)')
plt.ylabel('Linear Acceleration y (m/s^2)')

plt.subplot(2, 3, 6)
plt.plot(lizzy_jumping_sorted['Time (s)'], lizzy_jumping_sorted['Linear Acceleration z (m/s^2)'])
plt.title('Jumping Linear Acceleration z - DJ')
plt.xlabel('Time (s)')
plt.ylabel('Linear Acceleration z (m/s^2)')

plt.tight_layout()
plt.show()
'''
#WALKING X PLOTS
#dj data walking x plot
plt.figure(figsize = (10, 8))
#by default plot draws lines between the pointss/connects the data
dj_walking_df.plot(x='Time (s)', y='Linear Acceleration x (m/s^2)')
plt.title('DJ Acceleration in x')
#adding labels to the axis
plt.xlabel("Time (s)")
plt.ylabel("Linear Acceleration x (m/s^2)")
#rotatess the labels 90 degrees to make it look better/no overlap
pl.xticks(rotation = 90)
#adding a legend
plt.legend(["X acceleration"])
#formts the plot (fixes distance between elements and labels)
plt.tight_layout()
#function to show the plot
plt.show()

#lizzy data walking x plot
plt.figure(figsize = (10, 8))
#by default plot draws lines between the pointss/connects the data
lizzy_walking_df.plot(x='Time (s)', y='Linear Acceleration x (m/s^2)')
plt.title('Lizzy Acceleration in x')
#adding labels to the axis
plt.xlabel("Time (s)")
plt.ylabel("Linear Acceleration x (m/s^2)")
#rotatess the labels 90 degrees to make it look better/no overlap
pl.xticks(rotation = 90)
#adding a legend
plt.legend(["X acceleration"])
#formts the plot (fixes distance between elements and labels)
plt.tight_layout()
#function to show the plot
plt.show()

#isabel data walking x plot
plt.figure(figsize = (10, 8))
#by default plot draws lines between the pointss/connects the data
isabel_walking_df.plot(x='Time (s)', y='Linear Acceleration x (m/s^2)')
plt.title('Isabel Acceleration in x')
#adding labels to the axis
plt.xlabel("Time (s)")
plt.ylabel("Linear Acceleration x (m/s^2)")
#rotatess the labels 90 degrees to make it look better/no overlap
pl.xticks(rotation = 90)
#adding a legend
plt.legend(["X acceleration"])
#formts the plot (fixes distance between elements and labels)
plt.tight_layout()
#function to show the plot
plt.show()

#WALKING Y PLOTS
#dj data walking y plot
plt.figure(figsize = (10, 8))
#by default plot draws lines between the pointss/connects the data
dj_walking_df.plot(x='Time (s)', y='Linear Acceleration y (m/s^2)')
plt.title('DJ Acceleration in y')
#adding labels to the axis
plt.xlabel("Time (s)")
plt.ylabel("Linear Acceleration y (m/s^2)")
#rotatess the labels 90 degrees to make it look better/no overlap
pl.xticks(rotation = 90)
#adding a legend
plt.legend(["Y acceleration"])
#formts the plot (fixes distance between elements and labels)
plt.tight_layout()
#function to show the plot
plt.show()

#lizzy data walking y plot
plt.figure(figsize = (10, 8))
#by default plot draws lines between the pointss/connects the data
lizzy_walking_df.plot(x='Time (s)', y='Linear Acceleration y (m/s^2)')
plt.title('Lizzy Acceleration in y')
#adding labels to the axis
plt.xlabel("Time (s)")
plt.ylabel("Linear Acceleration y (m/s^2)")
#rotatess the labels 90 degrees to make it look better/no overlap
pl.xticks(rotation = 90)
#adding a legend
plt.legend(["Y acceleration"])
#formts the plot (fixes distance between elements and labels)
plt.tight_layout()
#function to show the plot
plt.show()

#isabel data walking y plot
plt.figure(figsize = (10, 8))
#by default plot draws lines between the pointss/connects the data
isabel_walking_df.plot(x='Time (s)', y='Linear Acceleration y (m/s^2)')
plt.title('Isabel Acceleration in y')
#adding labels to the axis
plt.xlabel("Time (s)")
plt.ylabel("Linear Acceleration y (m/s^2)")
#rotatess the labels 90 degrees to make it look better/no overlap
pl.xticks(rotation = 90)
#adding a legend
plt.legend(["Y acceleration"])
#formts the plot (fixes distance between elements and labels)
plt.tight_layout()
#function to show the plot
plt.show()

#WALKING Z PLOTS
#dj data walking z plot
plt.figure(figsize = (10, 8))
#by default plot draws lines between the pointss/connects the data
dj_walking_df.plot(x='Time (s)', y='Linear Acceleration z (m/s^2)')
plt.title('DJ Acceleration in z')
#adding labels to the axis
plt.xlabel("Time (s)")
plt.ylabel("Linear Acceleration z (m/s^2)")
#rotatess the labels 90 degrees to make it look better/no overlap
pl.xticks(rotation = 90)
#adding a legend
plt.legend(["Z acceleration"])
#formts the plot (fixes distance between elements and labels)
plt.tight_layout()
#function to show the plot
plt.show()

#lizzy data walking z plot
plt.figure(figsize = (10, 8))
#by default plot draws lines between the pointss/connects the data
lizzy_walking_df.plot(x='Time (s)', y='Linear Acceleration z (m/s^2)')
plt.title('Lizzy Acceleration in z')
#adding labels to the axis
plt.xlabel("Time (s)")
plt.ylabel("Linear Acceleration z (m/s^2)")
#rotatess the labels 90 degrees to make it look better/no overlap
pl.xticks(rotation = 90)
#adding a legend
plt.legend(["Z acceleration"])
#formts the plot (fixes distance between elements and labels)
plt.tight_layout()
#function to show the plot
plt.show()

#isabel data walking z plot
plt.figure(figsize = (10, 8))
#by default plot draws lines between the pointss/connects the data
isabel_walking_df.plot(x='Time (s)', y='Linear Acceleration y (m/s^2)')
plt.title('Isabel Acceleration in z')
#adding labels to the axis
plt.xlabel("Time (s)")
plt.ylabel("Linear Acceleration z (m/s^2)")
#rotatess the labels 90 degrees to make it look better/no overlap
pl.xticks(rotation = 90)
#adding a legend
plt.legend(["Z acceleration"])
#formts the plot (fixes distance between elements and labels)
plt.tight_layout()
#function to show the plot
plt.show()

#JUMPING PLOTS
#JUMPING X PLOTS
#dj data jumping x plot
plt.figure(figsize = (10, 8))
#by default plot draws lines between the pointss/connects the data
dj_jumping_df.plot(x='Time (s)', y='Linear Acceleration x (m/s^2)')
plt.title('DJ jumping Acceleration in x')
#adding labels to the axis
plt.xlabel("Time (s)")
plt.ylabel("Linear Acceleration x (m/s^2)")
#rotatess the labels 90 degrees to make it look better/no overlap
pl.xticks(rotation = 90)
#adding a legend
plt.legend(["X acceleration"])
#formts the plot (fixes distance between elements and labels)
plt.tight_layout()
#function to show the plot
plt.show()

#lizzy data jumping x plot
plt.figure(figsize = (10, 8))
#by default plot draws lines between the pointss/connects the data
lizzy_jumping_df.plot(x='Time (s)', y='Linear Acceleration x (m/s^2)')
plt.title('Lizzy jumping Acceleration in x')
#adding labels to the axis
plt.xlabel("Time (s)")
plt.ylabel("Linear Acceleration x (m/s^2)")
#rotatess the labels 90 degrees to make it look better/no overlap
pl.xticks(rotation = 90)
#adding a legend
plt.legend(["X acceleration"])
#formts the plot (fixes distance between elements and labels)
plt.tight_layout()
#function to show the plot
plt.show()

#isabel data jumping x plot
plt.figure(figsize = (10, 8))
#by default plot draws lines between the pointss/connects the data
isabel_jumping_df.plot(x='Time (s)', y='Linear Acceleration x (m/s^2)')
plt.title('Isabel jumping Acceleration in x')
#adding labels to the axis
plt.xlabel("Time (s)")
plt.ylabel("Linear Acceleration x (m/s^2)")
#rotatess the labels 90 degrees to make it look better/no overlap
pl.xticks(rotation = 90)
#adding a legend
plt.legend(["X acceleration"])
#formts the plot (fixes distance between elements and labels)
plt.tight_layout()
#function to show the plot
plt.show()

#JUMPING Y PLOTS
#dj data jumping y plot
plt.figure(figsize = (10, 8))
#by default plot draws lines between the pointss/connects the data
dj_jumping_df.plot(x='Time (s)', y='Linear Acceleration y (m/s^2)')
plt.title('DJ jumping Acceleration in y')
#adding labels to the axis
plt.xlabel("Time (s)")
plt.ylabel("Linear Acceleration y (m/s^2)")
#rotatess the labels 90 degrees to make it look better/no overlap
pl.xticks(rotation = 90)
#adding a legend
plt.legend(["Y acceleration"])
#formts the plot (fixes distance between elements and labels)
plt.tight_layout()
#function to show the plot
plt.show()

#lizzy data jumping y plot
plt.figure(figsize = (10, 8))
#by default plot draws lines between the pointss/connects the data
lizzy_jumping_df.plot(x='Time (s)', y='Linear Acceleration y (m/s^2)')
plt.title('Lizzy jumping Acceleration in y')
#adding labels to the axis
plt.xlabel("Time (s)")
plt.ylabel("Linear Acceleration y (m/s^2)")
#rotatess the labels 90 degrees to make it look better/no overlap
pl.xticks(rotation = 90)
#adding a legend
plt.legend(["Y acceleration"])
#formts the plot (fixes distance between elements and labels)
plt.tight_layout()
#function to show the plot
plt.show()

#isabel data jumping y plot
plt.figure(figsize = (10, 8))
#by default plot draws lines between the pointss/connects the data
isabel_jumping_df.plot(x='Time (s)', y='Linear Acceleration y (m/s^2)')
plt.title('Isabel Acceleration in y')
#adding labels to the axis
plt.xlabel("Time (s)")
plt.ylabel("Linear Acceleration y (m/s^2)")
#rotatess the labels 90 degrees to make it look better/no overlap
pl.xticks(rotation = 90)
#adding a legend
plt.legend(["Y acceleration"])
#formts the plot (fixes distance between elements and labels)
plt.tight_layout()
#function to show the plot
plt.show()

#JUMPING Z PLOTS
#dj data jumping z plot
plt.figure(figsize = (10, 8))
#by default plot draws lines between the pointss/connects the data
dj_jumping_df.plot(x='Time (s)', y='Linear Acceleration z (m/s^2)')
plt.title('DJ jumping Acceleration in z')
#adding labels to the axis
plt.xlabel("Time (s)")
plt.ylabel("Linear Acceleration z (m/s^2)")
#rotatess the labels 90 degrees to make it look better/no overlap
pl.xticks(rotation = 90)
#adding a legend
plt.legend(["Z acceleration"])
#formts the plot (fixes distance between elements and labels)
plt.tight_layout()
#function to show the plot
plt.show()

#lizzy data jumping z plot
plt.figure(figsize = (10, 8))
#by default plot draws lines between the pointss/connects the data
lizzy_jumping_df.plot(x='Time (s)', y='Linear Acceleration z (m/s^2)')
plt.title('Lizzy jumping Acceleration in z')
#adding labels to the axis
plt.xlabel("Time (s)")
plt.ylabel("Linear Acceleration z (m/s^2)")
#rotatess the labels 90 degrees to make it look better/no overlap
pl.xticks(rotation = 90)
#adding a legend
plt.legend(["Z acceleration"])
#formts the plot (fixes distance between elements and labels)
plt.tight_layout()
#function to show the plot
plt.show()

#isabel data jumping z plot
plt.figure(figsize = (10, 8))
#by default plot draws lines between the pointss/connects the data
isabel_walking_df.plot(x='Time (s)', y='Linear Acceleration y (m/s^2)')
plt.title('Isabel jumping Acceleration in z')
#adding labels to the axis
plt.xlabel("Time (s)")
plt.ylabel("Linear Acceleration z (m/s^2)")
#rotatess the labels 90 degrees to make it look better/no overlap
pl.xticks(rotation = 90)
#adding a legend
plt.legend(["Z acceleration"])
#formts the plot (fixes distance between elements and labels)
plt.tight_layout()
#function to show the plot
plt.show()
'''


