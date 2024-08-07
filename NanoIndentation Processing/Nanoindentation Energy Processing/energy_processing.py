# Processing Kasi's nanoindentation data

"""For complete data with standard deviations, take the max index with no Nan values

remove all data rows with NaN values

for each sheet, get the standard deviations of the rows

Loading curve: P = ah^b
Ut (total energy) = (a*hmax^(b+1))/(b+1)

Unloading curve: P = a(h-d)^b
Ur = (a*(hmax-d)^(b+1))/(b+1)

U = Ut - Ur

loading curve starts at 

# therefore need to split data into loading and unloading datapoints

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import odr, integrate
import os

def average_or_multiple():
    print('Welcome to the Nanoindentation Processing Program to calculate the total energy of the nanoindentation test.' \
          '\nThis program will calculate the total energy (Ut), plastic energy (Ur) and elastic energy (Ue) of the nanoindentation test using different modes.' )
    raw_data_processed_status = input('Do you want to process raw data or data from an excel file? (raw/excel): ')
    code_functionality = input('\nWill you be inputting an excel file containing average data for multiple samples? (y/n): ')
    if code_functionality != 'y' and code_functionality != 'n':
        print('Invalid input. Please enter either y or n.')
        average_or_multiple()
    else:
        return code_functionality

def raw_data_processing(folder_path: str) -> pd.DataFrame:
    files_in_folder = []
    avg_std_sample_dict = {} # used for exporting each file's avg_std_sample into an excel file as a 'report'#TODO this line was used before
    avg_std_output = pd.DataFrame(columns=['Sample', 'Ut', 'Ut STD', 'Ur', 'Ur STD', 'Ue', 'Ue STD']) #TODO this line was used before
    excel_report_name = os.path.basename(folder_path)
    sample = ''
    # print(excel_report_name)

    for root, dirs, files in os.walk(folder_path):
        if not files or dirs:
            continue
        avg_std_sample = pd.DataFrame(columns=['File Name', 'Ut', 'Ur', 'Ue']) #TODO this line was used before
        # print(f'This is the file list: {files}')
        for count, file in enumerate(files):
            files_in_folder.append(file)
            file_path = os.path.join(root, file)
            # print(f'Processing {file_path}')
            path_components = file_path.split('\\')
            if len(path_components) < 2:
                print('There has been an error with the path')
                exit()  # Not enough components for a second-to-last folder
            sample = path_components[-2]
            # Assuming the files are TXTs, you can read them into DataFrames
            if file.endswith('.txt'):
                #usecols=['Depth (nm)', 'Load (µN)'],
                df = pd.read_csv(file_path, sep='\t', header=3, usecols=[0,1], encoding='utf-8') # Assuming the data is tab-separated
                df.loc[-1] = df.columns
                df.index = df.index + 1
                df = df.sort_index()
                df.columns = ['Depth (nm)', 'Load (µN)']

                # print(df.columns)
                df = df.astype(float)
                avg_std_sample.loc[count] = [file, *discrete_integration(df)]
        
        avg_std_sample_dict[sample] = avg_std_sample
        print(f'Processing {sample}')
        avg_std_output.loc[len(avg_std_output)] = [sample, avg_std_sample['Ut'].mean(), avg_std_sample['Ut'].std(), avg_std_sample['Ur'].mean(), avg_std_sample['Ur'].std(), avg_std_sample['Ue'].mean(), avg_std_sample['Ue'].std()]
    print(avg_std_sample_dict)
    print(avg_std_output.to_string())

    with pd.ExcelWriter(f'{folder_path}\\{excel_report_name}.xlsx') as writer:
        for (sample, dataframe) in avg_std_sample_dict.items():
            dataframe.to_excel(writer, sheet_name=sample, index=False)
        avg_std_output.to_excel(writer, sheet_name='Average Ut Ur Ue ', index=False)

    print(f'Found {len(files_in_folder)} files in the folder')


def main():
    max_data = pd.DataFrame(columns=['Sample', 'Depth @ Max Load (nm)', 'Max Load (µN)', 'Max Indent Depth (nm)', 'Load @ Max Indent Depth (µN)']) # include in final excel file
    loading_data = pd.DataFrame(columns=['Sample', 'a', 'b', 'Ut']) # include in final excel file
    unloading_data = pd.DataFrame(columns=['Sample', 'a', 'b', 'd', 'Ur']) # include in final excel file
    discrete_energy = pd.DataFrame(columns=['Sample', 'Ut', 'Ur', 'Ue']) # include in final excel file

    if average_or_multiple() == 'y': # TODO FINISH THIS
        # need to integrate all the test columns and then take average and std of the Ut, Ur, Ue values
        complete_data = avg_std_data_excel('NanoIndentation Processing\\Nanoindentation Energy Processing\\Energy Ph Curve.xlsx') # Dictionary storing all the sheet names as keys and dataframes as values
        # plt.figure()
        avg_std_output = pd.DataFrame(columns=['Sample', 'Ut', 'Ut STD', 'Ur', 'Ur STD', 'Ue', 'Ue STD'])
        avg_std_sample_dict = {}
        for (sample, dataframe) in complete_data.items():
            test_numbers = list(set(dataframe.columns.get_level_values(0)))
            avg_std_sample = pd.DataFrame(columns=['Test No.', 'Ut', 'Ur', 'Ue'])
            for count, test in enumerate(test_numbers):
                # new_df = dataframe[test].map(lambda x: x if x >= 0 else None) #TODO: FIND OUT WHAT IS GOING WRONG HERE
                
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
        # plt.legend()
        # plt.show()
        with pd.ExcelWriter('NanoIndentation Processing\\Nanoindentation Energy Processing\\Average Ut Ur Ue.xlsx') as writer:
            for (sample, dataframe) in avg_std_sample_dict.items():
                dataframe.to_excel(writer, sheet_name=sample, index=False)
            avg_std_output.to_excel(writer, sheet_name='Average Ut Ur Ue', index=False)    
        print(avg_std_output.to_string())
        exit() #Todo: to be implemented

    excel_df = load_avg_test_data('NanoIndentation Processing\\Nanoindentation Energy Processing\\Average Ph Curve.xlsx')
    super_columns_list = list(set(excel_df.columns.get_level_values(0)))

    for count, sample in enumerate(super_columns_list): #TODO: make this into a function!!!
        excel_df[sample] = excel_df[sample].map(lambda x: x if x >= 0 else None)
        load_data = excel_df[sample]['Load (µN)']
        depth_data = excel_df[sample]['Depth (nm)']
        max_data.loc[count] = [sample, depth_data[load_data.idxmax()], load_data.max(), depth_data.max(), load_data[depth_data.idxmax()]]
        discrete_energy.loc[count] = [sample, *discrete_integration(excel_df[sample])]
        #depth_data.max() is the max depth 
    # print(max_data.to_string())
    discrete_energy = discrete_energy.sort_values(by='Sample')
    print(discrete_energy.to_string())
        # loading_data.loc[count] = [sample, *fit_data(loading_function, excel_df[sample], max_data.iloc[count][3])] #TODO: Add the final Ut value
        # unloading_data.loc[count] = [sample, *fit_data(unloading_function, excel_df[sample], max_data.iloc[count][3])] #TODO: Add the final Ur value

def total_energy(loading_params, unloading_params, hmax) -> float:
    Ut = (loading_params[0]*hmax**(loading_params[1]+1))/(loading_params[1]+1)
    Ur = (unloading_params[0]*(hmax-unloading_params[2])**(unloading_params[1]+1))/(unloading_params[1]+1)
    return Ur

def loading_function(params, h) -> float:
    a, b = params
    return a*h**b

def unloading_function(params, h) -> float:
    a, d, b= params
    return a*(h-d)**b

def load_avg_test_data(excel_file:str) -> pd.DataFrame:
    excel_data = pd.read_excel(excel_file, header=[0,1], sheet_name='All')
    excel_data = excel_data.dropna()
    return excel_data
#------------------------------------------------------------#
def avg_std_data_excel(excel_data:pd.DataFrame) -> pd.DataFrame:
    dataframe_dict = {}
    excel_file = pd.ExcelFile(excel_data)
    # excel_file = excel_file.map(lambda x: x if x >= 0 else None)
    sheet_names = excel_file.sheet_names
    for sheet in sheet_names:
        dataframe_dict[sheet] = pd.read_excel(excel_file, sheet_name=sheet, header=[0,1])
    return dataframe_dict
#---------------------------MARK: HELLO--------------------
def avg_std_calculation(dataframe:pd.DataFrame) -> pd.DataFrame: # TODO: Implement this
    
    pass

#------------------------------------------------------------#

def discrete_integration(sub_dataframe:pd.DataFrame, test_one=False, sample=None) -> float:
    # sub_dataframe = sub_dataframe.map(lambda x: x if x >= 0 else None)
    # print(f'Before size {sub_dataframe.size}')
    sub_dataframe = sub_dataframe[sub_dataframe['Depth (nm)'] >= 0]
    sub_dataframe = sub_dataframe[sub_dataframe['Load (µN)'] >= 0]
    # print(f'Middle size {sub_dataframe.size}')
    sub_dataframe = sub_dataframe.dropna().reset_index(drop=True)
    
    # print(f'Final size {sub_dataframe.size}')

    Ur = integrate.trapezoid(sub_dataframe['Load (µN)'], sub_dataframe['Depth (nm)'])
    hmax_index = sub_dataframe['Depth (nm)'].idxmax()
    hmax = sub_dataframe['Depth (nm)'][hmax_index]
    if test_one:
        print(f'{sample} Test 1: {hmax}')
    
    # print(hmax_index)
    filtered_df_Ut = sub_dataframe.iloc[:(hmax_index+1)] # loading curve
    Ut = integrate.trapezoid(filtered_df_Ut['Load (µN)'], filtered_df_Ut['Depth (nm)'])
    # print(sub_dataframe.iloc[hmax_index:].to_string())
    filtered_df_Ue = sub_dataframe.iloc[hmax_index:].iloc[::-1] # unloading curve
    plt.clf()
    if test_one:
        plt.plot(filtered_df_Ut['Depth (nm)'],filtered_df_Ut['Load (µN)'], label=f'{sample} Test 1 Loading Curve')
        plt.plot( filtered_df_Ue['Depth (nm)'],filtered_df_Ue['Load (µN)'], label=f'{sample} Test 1 Unloading Curve')
        plt.legend()
        plt.show()
    # print(filtered_df_Ue.to_string())
    ue2 = integrate.trapezoid(filtered_df_Ue['Load (µN)'], filtered_df_Ue['Depth (nm)'])
    Ue = Ut-Ur
    ue2 = round(ue2, 4)
    Ue = round(Ue, 4)
    # print(Ue == ue2)
    if Ue != ue2:
        print(f'Ut = {Ut}\nUr = {Ur}\nUe = {Ue}\nue2 = {ue2}')
        print('Ue calculation error')
        exit()
    Ut = round(Ut, 2)
    Ur = round(Ur, 2)
    Ue = round(Ue, 2)
    return [Ut, Ur, Ue]

def fit_data(function_name, data:pd.DataFrame, hmax) -> list:
    model = odr.Model(function_name)
    data = odr.Data(x=data['Depth (nm)'], y=data['Load (µN)'])
    odr_instance = odr.ODR(data, model, beta0=[1.0, 1.0])
    output = odr_instance.run()
    return output.beta
    




if __name__ == '__main__':
    raw_data_processing('C:\\Users\\alber\\OneDrive - University of Calgary\\2024Tribometer\\NanoIndentation\\Vinay NanoIndentation Data')