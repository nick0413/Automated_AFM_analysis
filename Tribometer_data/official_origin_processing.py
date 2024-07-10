import originpro as op
import os
import numpy as np
import pandas as pd


def user_interface():
    print("Welcome to the Origin data analysis tool using python for the friction coefficient data.")
    print("Please select where to process data:")
    print("1. OriginPro")
    print("2. Exit")
    choice = input("Enter your choice: ")
    if choice == '1':
        data = origin_data_input()
    elif choice == '2':
        exit()
    else:
        print("Invalid choice")
        user_interface()
    return data

# current_work_dir = os.getcwd()


def origin_data_input():
    data_file_path = input("Enter the path of the data file: ")
    # Open the workbook
    op.new()
    wks = op.new_sheet()
    wks.from_file(data_file_path)
    return wks

def read_data_into_pandas():
    data = pd.read_excel('path') #TODO: Change the path to the data file
    return data

class Processed_TriboFile: # a class to store the previously processed data
    # class defines properties that an object can have
    # object is an instantiated or specific version of a class
    def __init__(self, worksheet, data, file_folder, file_name):
        """Initializes the Processed_TriboFile object with the data, file folder, and file name."""
        self.worksheet = worksheet
        self.data = data # as a .xlsx (microsoft excel) file
        self.file_folder = file_folder # directory where the file is stored
        self.file_name = file_name # name of the file

    def smooth_data(self): # methods that modify the processed tribofile <-- seems wayyy harder
        """ Loads excel data to origin and applies Savitzky-Golay smoothing at polynomal 2 and window size 100 to the data
        Saves the smoothed data to the same excel file.
        """
        pass

    def process_data(self): # much easier to use scipy
        """Uses Scipy.signal? to smoothen out data Pandas to takes average CoF values for each cycle pertaining to the number of repetition of tests done per sample that have the same combinations of parameters
          and notes down the STDEV in the column adjacent to it."""
        pass

    def plot_graphs(self, save_location=None, plot_cof_cycle=False, plot_avg_cof=False, plot_error_bars=False): # hard to do but pays off in long run <-- not super important
        # maybe first just automating making the graphs in OriginPro without any errors or lines <-- maybe way easier to do in plotly or matplotlib
        """ Plot graphs. Ideally including the following: % reduction arrows pointing to ref lines from non reference bars, error bars, reference lines, no overlapping, etc.
        """
        pass


    
    pass

"""
class FrictCoF_vs_Cycles:
    def __init__(self, wks, data):
        self.wks = wks
        self.data = data
    # Assuming 'wks' is your worksheet variable and it's already populated with data
    # Plot the data if not already plotted
    graph = op.new_graph(template='line')  # Use an appropriate template
    dataplot = graph[0].add_plot(wks, 0)  # Assuming data is in the first column

    # Apply SG Smoothing
    # You need to specify the window size (odd number) and polynomial order
    window_size = 100  # Example window size
    poly_order = 2  # Example polynomial order

    # Access the Data Processor
    dp = op.DataProcessor(dataplot)

    # Apply SG Smoothing
    dp.smooth_sg(window_size, poly_order)

    # Note: Adjust 'window_size' and 'poly_order' according to your data and smoothing needs
    pass

def main():
    pass

if __name__ == '__main__':
    main()

"""
"""
import originpro as op

# Start an Origin session
app = op.Application()
app.Visible = app.MAINWND_SHOW

# Load data from a CSV file
fname = 'path_to_your_data_file.csv'
wks = op.new_sheet()
wks.from_file(fname)

# Create a line plot
graph = op.new_graph(template='line')
layer = graph[0]
layer.add_plot(wks, 0, 1)  # Assuming your X data is in column 0 and Y data in column 1

# Rescale the graph
layer.rescale()

# Save the project (optional)
app.save('path_to_save_your_project.opj')

# End the Origin session
app.exit()

"""
    