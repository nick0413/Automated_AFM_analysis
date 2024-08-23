import pandas as pd
import matplotlib.pyplot as plt
from scipy import odr, integrate
# import plotly.graph_objects as go
import os

# create function, test one or random test per graph
# create function to remove quotations from user input
# create function to plot all tests of one sample on one graph

def display_initial_message():
    """Displays the initial message to the user."""
    print('Welcome to the Nanoindentation Processing Program to calculate the total energy of the nanoindentation test.' \
          '\nThis program will calculate the total energy (Ut), plastic energy (Ur) and elastic energy (Ue) of the nanoindentation test using different modes.' )

def get_user_input(prompt: str, valid_options: list) -> str:
    """Gets validated user input based on the provided prompt and valid options.
        If the user's input is invalid, the function will call itself again.
        Arguments: prompt: string, valid_options: list.
        Return: user_input: string."""
    
    user_input = input(prompt) # prompt the user for input
    while user_input not in valid_options: # validate the user's input
        print(f'Invalid input. Please enter one of the following: {", ".join(valid_options)}.') # display an error message
        user_input = input(prompt) # prompt the user for input again
    return user_input # return the user's input

def raw_or_excel():
    """Asks the user if they want to process raw data or data from an excel file.
    The user input is validated against the valid options 'raw' and 'excel'.
    This user input is the first input the user will provide."""

    first_input = get_user_input('Do you want to process raw data or data from an excel file? (raw/excel): ', ['raw', 'excel']) # calls get user input function
    if first_input == 'raw': # if the user chooses to process raw data
        #---------------------------MARK: Data Files--------------------
        raw_files_folder_path = input('Please enter the path to the folder containing the raw data files: ') # prompt the user for the path to the folder containing the raw data files
        # raw_files_folder_path = 'C:\\Users\\alber\\Summer Research Code\\Automated_AFM_analysis\\Tests' # can also modify to make it so that the user can code the path in without interfacing with the terminal
        raw_data_processing(raw_files_folder_path) # call the raw data processing function
    elif first_input == 'excel': # if the user chooses to process data from an excel file
        excel_path = input('Please enter the path to the excel file: ') # prompt the user for the path to the excel file
        excel_data_processing(excel_path, average_or_multiple()) # TODO make a function called excel processing # call the excel data processing function
        pass

def average_or_multiple():
    """
    Prompts the user to input whether they will be using an excel file containing average data for multiple samples.
    Returns:
        str: The user's input, either 'y' or 'n'.
    """
    
    return get_user_input('Will you be inputting an excel file containing data for multiple samples? (y/n): ', ['y', 'n'])

def raw_data_processing(folder_path: str) -> pd.DataFrame:
    """Processes raw data (.txt files) from a folder containing nanoindentation data.
    Arguments: folder_path: string.
    Return: None.
    Within the folder path, the function will iterate through the files and calculate the total energy (Ut), plastic energy (Ur) and elastic energy (Ue) of the nanoindentation test.
    The function will also create an excel file containing the average and standard deviation of the Ut, Ur, and Ue values for each sample.
    The excel file will be saved in the same folder path as the raw data.
    Additionally, the subfolder's name within the folder path should be the name of the sample that is being nano-indented."""

    files_in_folder = [] # list to store the files in the folder
    avg_std_sample_dict = {} # used for exporting each file's avg_std_sample into an excel file as a 'report'
    avg_std_output = pd.DataFrame(columns=['Sample', 'Ut', 'Ut STD', 'Ur', 'Ur STD', 'Ue', 'Ue STD']) # used for exporting the average and std of the Ut, Ur, Ue values
    excel_report_name = os.path.basename(folder_path) # used for naming the excel report
    sample = '' # used for storing the sample name
    # print(excel_report_name)

    for root, dirs, files in os.walk(folder_path): # iterates through the folder path
        if not files or dirs: # if there are no files in the folder or if there are subdirectories
            continue # continue to the next iteration
        avg_std_sample = pd.DataFrame(columns=['File Name', 'Ut', 'Ur', 'Ue']) # used for storing the Ut, Ur, Ue values of each file
        # print(f'This is the file list: {files}')
        plt.figure(figsize=(15,9)) # create a new figure for each plot

        # fig = go.Figure() # create a plotly figure
        for count, file in enumerate(files): # iterates through the files in the folder
            files_in_folder.append(file) # appends the file to the list <-- to be used for counting the number of files in the folder
            file_path = os.path.join(root, file) # joins the root and file to obtain the file's specific path
            # print(f'Processing {file_path}')
            path_components = file_path.split(os.path.sep) # splits the file path by the backslash (works on windows computers)
            if len(path_components) < 2: # if there are not enough components for a second-to-last folder
                print('There has been an error with the path. Please ensure that the sample\'s name is the name of a subfolder.') # print an error message
                exit()  # Exits the program to prevent further errors
            sample = path_components[-2] # the second-to-last path component is the sample name as the file is within that sub-directory
            # Assuming the files are TXTs, you can read them into DataFrames
            if file.endswith('.txt'):
                #usecols=['Depth (nm)', 'Load (µN)'],
                if os.path.getsize(file_path) < 4000: # if the file is empty
                    continue
                df = pd.read_csv(file_path, sep='\t', header=3, usecols=[0,1], encoding='utf-8') # Assuming the data is tab-separated, read the file into a dataframe
                df.loc[-1] = df.columns # adding the column names to the -1st row
                df.index = df.index + 1 # shifting index 
                df = df.sort_index() # sorting by index
                df.columns = ['Depth (nm)', 'Load (µN)'] # renaming the columns, assumes that the columns are in the same order
                # TODO add a check to see if the columns in the file are in the correct order, so that it is not hardcoded
                # print(df.columns)
                df = df.astype(float) # converting the columns to float (decimal numbers) types
                # fig.add_trace(go.Scatter(x=df['Depth (nm)'], y=df['Load (µN)'], mode='lines', name=file))
                plt.plot(df['Depth (nm)'], df['Load (µN)'], label=f'{sample} {file}') # plot the depth vs load curve
                plt.legend(loc='best') # show the legend

                avg_std_sample.loc[count] = [file, *discrete_integration(df)] # appends the Ut, Ur, Ue values (obtained from trapezoidal integration) to the avg_std_sample dataframe
        
        # Customize layout
        # fig.update_layout(
        #     title=sample,
        #     xaxis_title='Depth (nm)',
        #     yaxis_title='Load (µN)')

        # # Show the figure
        # fig.show()
        # fig.data = []
        os.makedirs(os.path.join(folder_path, 'Output Plots'), exist_ok=True) # create a folder called 'Output Plots' if it does not exist
        output_folder_images = os.path.join(folder_path, 'Output Plots') # create a path to the 'Output Plots' folder
        plt.xlabel('Depth (nm)') # label the x-axis
        plt.ylabel('Load (µN)') # label the y-axis
        plt.savefig(os.path.join(output_folder_images, f'{sample}.png')) # save the plot as a png file
        plt.close() # close the plot
        # plt.show() # show the plot

        #TODO fix this part
        
        avg_std_sample_dict[sample] = avg_std_sample # appends the avg_std_sample dataframe to the avg_std_sample_dict dictionary
        print(f'Processing {sample}') # prints the sample name
        avg_std_output.loc[len(avg_std_output)] = [sample, avg_std_sample['Ut'].mean(), avg_std_sample['Ut'].std(), avg_std_sample['Ur'].mean(), avg_std_sample['Ur'].std(), avg_std_sample['Ue'].mean(), avg_std_sample['Ue'].std()] # appends the average and std of the Ut, Ur, Ue values to the avg_std_output dataframe
    # print(avg_std_sample_dict) # prints the avg_std_sample_dict dictionary
    print(avg_std_output.to_string()) # prints the avg_std_output dataframe
    print(f'Found {len(files_in_folder)} files in the folder')
    excel_output(avg_std_sample_dict, avg_std_output, folder_path) # calls the excel_output function



def excel_data_processing(excel_file_path: str, average_or_multiple: str) -> None:
    max_data = pd.DataFrame(columns=['Sample', 'Depth @ Max Load (nm)', 'Max Load (µN)', 'Max Indent Depth (nm)', 'Load @ Max Indent Depth (µN)']) # include in final excel file
    discrete_energy = pd.DataFrame(columns=['Sample', 'Ut', 'Ur', 'Ue']) # Shows the Ut, Ur, Ue values for each sample
    excel_folder_path = os.path.dirname(excel_file_path) # get the folder path of the excel file

    if average_or_multiple == 'y': # if the user chooses to process an excel file containing average data for multiple samples    
        complete_data = avg_std_data_excel(excel_file_path) # Dictionary storing all the sheet names as keys and dataframes as values
        avg_std_output = pd.DataFrame(columns=['Sample', 'Ut', 'Ut STD', 'Ur', 'Ur STD', 'Ue', 'Ue STD'])
        avg_std_sample_dict = {}
        for (sample, dataframe) in complete_data.items():
            test_numbers = list(set(dataframe.columns.get_level_values(0)))
            avg_std_sample = pd.DataFrame(columns=['Test No.', 'Ut', 'Ur', 'Ue'])
            for count, test in enumerate(test_numbers):                
                if test == 'Test 1':
                    # plt.plot(dataframe[test]['Depth (nm)'], dataframe[test]['Load (µN)'], label=f'{sample} Test {test}')
                    # plt.plot(dataframe[test]['Depth (nm)'], dataframe[test]['Load (µN)'], label=f'{sample} Test {test} Negs')
                    avg_std_sample.loc[count] = [test, *discrete_integration(dataframe[test], True, sample)]
                else:
                    avg_std_sample.loc[count] = [test, *discrete_integration(dataframe[test])]
                

            avg_std_sample['Test No. Num'] = avg_std_sample['Test No.'].str.extract('(\d+)').astype(int)
            avg_std_sample = avg_std_sample.sort_values(by='Test No. Num').drop(columns='Test No. Num')
            avg_std_sample_dict[sample] = avg_std_sample
            avg_std_output.loc[len(avg_std_output)] = [sample, avg_std_sample['Ut'].mean(), avg_std_sample['Ut'].std(), avg_std_sample['Ur'].mean(), avg_std_sample['Ur'].std(), avg_std_sample['Ue'].mean(), avg_std_sample['Ue'].std()]
        excel_output(avg_std_sample_dict, avg_std_output, excel_folder_path, 'Avg Energy Values')
    
    elif average_or_multiple == 'n': # if the user chooses to process an excel file containing data for averages
        excel_df = load_avg_test_data(excel_file_path)
        super_columns_list = list(set(excel_df.columns.get_level_values(0)))

        for count, sample in enumerate(super_columns_list): #TODO: make this into a function!!!
            excel_df[sample] = excel_df[sample].map(lambda x: x if x >= 0 else None)
            load_data = excel_df[sample]['Load (µN)']
            depth_data = excel_df[sample]['Depth (nm)']
            max_data.loc[count] = [sample, depth_data[load_data.idxmax()], load_data.max(), depth_data.max(), load_data[depth_data.idxmax()]]
            discrete_energy.loc[count] = [sample, *discrete_integration(excel_df[sample])]
        discrete_energy = discrete_energy.sort_values(by='Sample')
        #print(discrete_energy.to_string()) # optional print statement in terminal which outputs the dataframe
        excel_output({}, discrete_energy, excel_folder_path, 'Average Energy Values')

def excel_output(avg_std_sample_dict: dict, avg_std_output: pd.DataFrame, folder_path: str, specific_name=None) -> None:
    """Exports the dataframes within avg_std_sample_dict and the avg_std_output dataframe to an excel file.
    Arguments: avg_std_sample_dict: dictionary, avg_std_output: dataframe, folder_path: string.
    Return: None.
    The function will create an excel file containing the average and standard deviation of the Ut, Ur, and Ue values for each sample.
    Also includes the Ut, Ur, and Ue values for each file within the sample folder.
    The excel file will be saved in the same folder path as the raw data."""

    excel_report_name = os.path.basename(folder_path) # used for naming the excel report
    excel_report_name += ' Refactored' # add 'Refactored' to the excel report name
    if specific_name: # if a specific name is provided, implicit boolean check
        excel_report_name = specific_name # set the excel report name to the specific name
        excel_report_name += ' Refactored' # add 'Refactored' to the excel report name
    full_path = os.path.join(folder_path, f'{excel_report_name}.xlsx') # create the full path to the excel file
    with pd.ExcelWriter(full_path) as writer: # creates an excel writer object
        if avg_std_sample_dict: # if the avg_std_sample_dict is not empty
            for (sample, dataframe) in avg_std_sample_dict.items(): # iterates through the avg_std_sample_dict dictionary
                dataframe.to_excel(writer, sheet_name=sample, index=False) # exports the dataframe within the dictionary to an excel sheet
        avg_std_output.to_excel(writer, sheet_name='Average Ut Ur Ue', index=False) # exports the avg_std_output dataframe to an excel sheet

def load_avg_test_data(excel_file:str) -> pd.DataFrame:
    """
    Load and process average test data from an Excel file.
    Parameters:
        excel_file (str): The path to the Excel file containing the data.
    Returns:
        pd.DataFrame: The processed data as a pandas DataFrame.
    """

    excel_data = pd.read_excel(excel_file, header=[0,1], sheet_name='All') # read the excel file into a dataframe
    excel_data = excel_data.dropna() # drop rows with NaN values
    return excel_data # return the processed data

#------------------------------------------------------------#
def avg_std_data_excel(excel_data:pd.DataFrame) -> pd.DataFrame:
    """
    Reads an Excel file containing multiple sheets and returns a dictionary of dataframes.
    Parameters:
        excel_data (pd.DataFrame): The Excel file to be processed.
    Returns:
        dict: A dictionary where the keys are the sheet names and the values are the corresponding dataframes.
    """

    dataframe_dict = {} # initialize the dictionary to store the dataframes
    excel_file = pd.ExcelFile(excel_data) # read the excel file
    # excel_file = excel_file.map(lambda x: x if x >= 0 else None)
    sheet_names = excel_file.sheet_names # get the sheet names of the excel file
    for sheet in sheet_names: # iterate through the sheet names
        dataframe_dict[sheet] = pd.read_excel(excel_file, sheet_name=sheet, header=[0,1]) # read the sheet into a dataframe and store it in the dictionary
    return dataframe_dict # return the dictionary containing the dataframes
#-----------------------------------------------

def discrete_integration(sub_dataframe:pd.DataFrame, test_one=False, sample=None) -> float:
    """
    Calculates the total energy, Ut, Ur, and Ue values for a given sub_dataframe.
    Args:
        sub_dataframe (pd.DataFrame): The input dataframe containing the depth and load columns.
        test_one (bool, optional): Flag indicating if it is the first test. Defaults to False.
        sample (Any, optional): The sample identifier. Defaults to None.
    Returns:
        List[float]: A list containing the Ut, Ur, and Ue values.
    Raises:
        None
    """

    # sub_dataframe = sub_dataframe.map(lambda x: x if x >= 0 else None)
    # print(f'Before size {sub_dataframe.size}')
    sub_dataframe = sub_dataframe[sub_dataframe['Depth (nm)'] >= 0] # remove negative values from depth column
    sub_dataframe = sub_dataframe[sub_dataframe['Load (µN)'] >= 0] # remove negative values from load column
    # print(f'Middle size {sub_dataframe.size}')
    sub_dataframe = sub_dataframe.dropna().reset_index(drop=True) # drop rows with NaN values
    
    # print(f'Final size {sub_dataframe.size}')

    Ur = integrate.trapezoid(sub_dataframe['Load (µN)'], sub_dataframe['Depth (nm)']) # total energy
    hmax_index = sub_dataframe['Depth (nm)'].idxmax() # index of max depth
    hmax = sub_dataframe['Depth (nm)'][hmax_index] # max depth
    if test_one: # if it is the first test
        print(f'{sample} Test 1: {hmax}') # print the max depth
    
    # print(hmax_index)
    filtered_df_Ut = sub_dataframe.iloc[:(hmax_index+1)] # loading curve data points filtered
    Ut = integrate.trapezoid(filtered_df_Ut['Load (µN)'], filtered_df_Ut['Depth (nm)']) # total energy
    # print(sub_dataframe.iloc[hmax_index:].to_string())
    filtered_df_Ue = sub_dataframe.iloc[hmax_index:].iloc[::-1] # unloading curve data points filtered
    if test_one: # if it is the first test <-- used mainly for testing purposes
        plt.clf() # clear the plot
        plt.plot(filtered_df_Ut['Depth (nm)'],filtered_df_Ut['Load (µN)'], label=f'{sample} Test 1 Loading Curve') # plot the loading curve
        plt.plot( filtered_df_Ue['Depth (nm)'],filtered_df_Ue['Load (µN)'], label=f'{sample} Test 1 Unloading Curve') # plot the unloading curve
        plt.legend() # show the legend
        plt.show() # show the plot
    # print(filtered_df_Ue.to_string())
    ue2 = integrate.trapezoid(filtered_df_Ue['Load (µN)'], filtered_df_Ue['Depth (nm)']) # integrate to find the area under the unloading curve
    Ue = Ut-Ur # calculate the elastic energy from ut and ur
    ue2 = round(ue2, 4) # round the ue2 value to 4 decimal places to ensure accuracy with minimal errors 
    Ue = round(Ue, 4) # round the Ue value to 4 decimal places to ensure accuracy with minimal errors
    # print(Ue == ue2)
    if Ue != ue2: # ensure that both methods for calculating the elastic energy are the same
        print(f'Ut = {Ut}\nUr = {Ur}\nUe = {Ue}\nue2 = {ue2}') # print the values of Ut, Ur, Ue, and ue2
        print('Ue calculation error') # print an error message if the methods for calculating elastic energy do not match
        exit() # exit the program to prevent further errors
    Ut = round(Ut, 2) # round the Ut value to 2 decimal places for displaying in excel or other outputs
    Ur = round(Ur, 2) # round the Ur value to 2 decimal places for displaying in excel or other outputs
    Ue = round(Ue, 2) # round the Ue value to 2 decimal places for displaying in excel or other outputs
    return [Ut, Ur, Ue] # return the Ut, Ur, and Ue values when the function is called with the discrete_integration function

#------------------------------------------------------------#
# Functions for Fitting data (not used yet maybe in the future)
def fit_data(function_name, data:pd.DataFrame, hmax) -> list:
    """
    Fits the given data to a specified function using Orthogonal Distance Regression (ODR).
    Parameters:
    - function_name (callable): The function to fit the data to.
    - data (pd.DataFrame): The data to be fitted, with columns 'Depth (nm)' and 'Load (µN)'.
    - hmax: The maximum value of h.
    Returns:
    - list: The estimated parameters of the fitted function.
    """

    model = odr.Model(function_name) # create a model for the ODR
    data = odr.Data(x=data['Depth (nm)'], y=data['Load (µN)']) # create a data object for the ODR
    odr_instance = odr.ODR(data, model, beta0=[1.0, 1.0]) # create an ODR instance
    output = odr_instance.run() # run the ODR
    return output.beta # return the estimated parameters of the fitted function
    
def total_energy(loading_params, unloading_params, hmax) -> float:
    """
    Calculate the total energy under curve during nanoindentation.
    Parameters:
    loading_params (list): List of loading parameters [a, b].
    unloading_params (list): List of unloading parameters [a, b, d].
    hmax (float): Maximum indentation depth.
    Returns:
    float: The total energy under curve during nanoindentation.
    """

    loading_data = pd.DataFrame(columns=['Sample', 'a', 'b', 'Ut']) # include in final excel file
    unloading_data = pd.DataFrame(columns=['Sample', 'a', 'b', 'd', 'Ur']) # include in final excel file
    Ut = (loading_params[0]*hmax**(loading_params[1]+1))/(loading_params[1]+1)
    Ur = (unloading_params[0]*(hmax-unloading_params[2])**(unloading_params[1]+1))/(unloading_params[1]+1)
    return Ur

def loading_function(params, h) -> float:
    """
    Calculate the loading function value based on the given parameters and indentation depth.
    Parameters:
    params (tuple): A tuple containing two parameters (a, b).
    h (float): The indentation depth.
    Returns:
    float: The calculated loading function value.
    """

    a, b = params
    return a*h**b

def unloading_function(params, h) -> float:
    """
    Calculates the unloading force in a nanoindentation process.
    Parameters:
    - params (list): A tuple containing the parameters a, d, and b.
    - h (float): The indentation depth.
    Returns:
    - float: The calculated unloading force.
    """

    a, d, b= params
    return a*(h-d)**b

#------------------------------------------------------------#

#---------------------------MARK: MAIN FUNCTION---------------------------------
def main():
    display_initial_message() # display the initial message to the user
    raw_or_excel() # call the raw_or_excel function to process the raw data or data from an excel file

if __name__ == '__main__':
    main()