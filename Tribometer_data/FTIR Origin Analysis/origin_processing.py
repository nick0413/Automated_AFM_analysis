import originpro as op
import pandas as pd
import os
from functools import reduce # processing for originpro

# import matplotlib.pyplot as plt

# practice plots for FTIR data

class FTIR_file:
    def __init__(self, file_name, folder):
        self.folder = folder
        self.file_name = file_name
        self.full_path = f'{folder}\\{file_name}'
        self.data = pd.read_csv(f'{self.full_path}')
        self.columns = self.data.columns

def main():
    folder = 'Tribometer_data\\FTIR Origin Analysis\\vinay'
    # Read data from a file
    files_in_folder = os.listdir(folder)
    print(files_in_folder)
    FTIR_files = [FTIR_file(file, folder) for file in files_in_folder if file.endswith('.csv')]
    print(f'Found {len(FTIR_files)} csv files in the folder: {folder}.')
    print(check_data(FTIR_files).to_string())


def check_data(object_list:list): #TODO fix object list instances
    print(f'Checking if columns names in files are the same...')
    set_checker = [tuple(dataframe.columns) for dataframe in object_list]
    set_checker = list(set(set_checker))
    if len(set_checker) == 1:
        print(f'All files have the same columns: {set_checker[0]}')
        column_name_length = len(set_checker[0])
        column_names = ''
        for count, column in enumerate(set_checker[0]):
            column_names += f'{count+1}: {column} \n'
        
        reference_column_name = set_checker[0][ref_col_input(column_names, column_name_length)-1]
    else:
        raise ValueError(f'Columns in files are not the same: {set_checker}')

    print(f'Checking if x-axis column values in files are the same or if one is a subset of the other...')
    first_df = object_list[0].data

    for df in object_list[1:]:
        if not first_df.equals(df.data):
            print(f'Files do not have the same x-axis values')
        else:
            print(f'Files have the same x-axis values')
    if status_input():
        print('Proceeding with analysis...')
        intersecting_dataframe = reduce(lambda left, right: pd.merge(left, right, on=list(left.columns), how='inner'), [object.data for object in object_list])
        return intersecting_dataframe
    else:
        print('Exiting program...')
        exit()



def ref_col_input(column_names:str, col_names_len:int):
    reference_column = input(f'Which column is the x-axis column? Input integer value please: \n {column_names}')
    if reference_column.isnumeric() and int(reference_column) <= col_names_len:
        return int(reference_column)
    else:
        print(f'Invalid input. Please enter a number between 1 and {col_names_len}')
        return ref_col_input(column_names, col_names_len)

def status_input():
    status = input('Would you like to proceed with the analysis? (y/n): \n')
    if status.lower() == 'y':
        return True
    elif status.lower() == 'n':
        return False
    else:
        print('Invalid input. Please enter y or n')
        return status_input()


def origin_line():
    # Open a new Origin instance   
    app = op.Application()

    # Open a new workbook
    wb = app.new_book()

    # Import data from a file
    wb.from_file('data.txt')

    # Perform some data processing
    wb.active_layer.do_something()

    # Save the workbook
    wb.save('processed_data.opj')


if __name__ == '__main__':
    main()