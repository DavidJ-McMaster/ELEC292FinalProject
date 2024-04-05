import tkinter as tk
from tkinter import filedialog
import pandas as pd
#used to clean up the axis of the ployts that have lots of time values
import pylab as pl
import matplotlib.pyplot as plt


def choose_file():
    # needs to be a csv file
    # the filedialog.askopenfilename is how we can actually allow the user to choose a file
    filename = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])

    # this checks to see if the user actually selected a file to open
    if filename:
        entry.delete(0, tk.END)
        entry.insert(0, filename)


def process_file():
    input_file = entry.get()
    if input_file:
        df = pd.read_csv(input_file)

        # code that was done in the project
        # x-axis???

        fig, plots = plt.subplots(figsize=(10,8))
    
        df.plot(x='Time (s)', y= ['Linear Acceleration x (m/s^2)', 'Linear Acceleration y (m/s^2)','Linear Acceleration z (m/s^2)'], ax=plots)

        plots.set_title('Acceleration in XYZ axes')
        plots.set_xlabel("Time (s)")
        plots.set_ylabel("Linear Acceleration (m/s^2)")

        plt.xticks(rotation=90)

        plots.legend(["X acceleration", "Y Acceleration", "Z accleration"])
        plt.tight_layout()
        plt.show()


        plt.figure(figsize=(10, 8))
        # by default plot draws lines between the pointss/connects the data
        df.plot(x='Time (s)', y='Linear Acceleration x (m/s^2)')
        plt.title('Acceleration in x')
        # adding labels to the axis
        plt.xlabel("Time (s)")
        plt.ylabel("Linear Acceleration x (m/s^2)")
        # rotates the labels 90 degrees to make it look better/no overlap
        pl.xticks(rotation=90)
        # adding a legend
        plt.legend(["X acceleration"])
        # formts the plot (fixes distance between elements and labels)
        plt.tight_layout()
        # function to show the plot
        plt.show()

        # data  y plot
        plt.figure(figsize=(10, 8))
        # by default plot draws lines between the points/connects the data
        df.plot(x='Time (s)', y='Linear Acceleration y (m/s^2)')
        plt.title('Acceleration in y')
        # adding labels to the axis
        plt.xlabel("Time (s)")
        plt.ylabel("Linear Acceleration y (m/s^2)")
        # rotatess the labels 90 degrees to make it look better/no overlap
        pl.xticks(rotation=90)
        # adding a legend
        plt.legend(["Y acceleration"])
        # formts the plot (fixes distance between elements and labels)
        plt.tight_layout()
        # function to show the plot
        plt.show()

        # data z-axis
        plt.figure(figsize=(10, 8))
        # by default plot draws lines between the pointss/connects the data
        df.plot(x='Time (s)', y='Linear Acceleration z (m/s^2)')
        plt.title('Acceleration in z')
        # adding labels to the axis
        plt.xlabel("Time (s)")
        plt.ylabel("Linear Acceleration z (m/s^2)")
        # rotatess the labels 90 degrees to make it look better/no overlap
        pl.xticks(rotation=90)
        # adding a legend
        plt.legend(["Z acceleration"])
        # formts the plot (fixes distance between elements and labels)
        plt.tight_layout()
        # function to show the plot
        plt.show()


root = tk.Tk()
root.title("CSV File Input")

label = tk.Label(root, text="Select a CSV file:")
label.pack()

entry = tk.Entry(root, width=100)
entry.pack()

browse_button = tk.Button(root, text="Browse", command=choose_file, width=20, height=5, bg="white", fg="blue")
browse_button.pack()

process_button = tk.Button(root, text="Process", command=process_file, width=20, height=5, bg="white", fg="blue")
process_button.pack()

root.mainloop()
