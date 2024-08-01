# Processing Kasi's nanoindentation data

"""For complete data with standard deviations, take the max index with no Nan values

remove all data rows with NaN values

for each sheet, get the standard deviations of the rows

"""

"""
Loading curve: P = ah^b
Ut (total energy) = (a*hmax^(b+1))/(b+1)

Unloading curve: P = a(h-d)^b
Uu = (a*(hmax-d)^(b+1))/(b+1)

U = Ut - Uu

loading curve starts at 

# therefore need to split data into loading and unloading datapoints

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import odr

def main():
    max_data = pd.DataFrame(columns=['Sample', 'Depth @ Max Load (nm)', 'Max Load (µN)', 'Max Indent Depth (nm)', 'Load @ Max Indent Depth (µN)']) # include in final excel file
    loading_data = pd.DataFrame(columns=['Sample', 'a', 'b', 'Ut']) # include in final excel file
    unloading_data = pd.DataFrame(columns=['Sample', 'a', 'b', 'd', 'Uu']) # include in final excel file
    excel_df = load_avg_test_data('NanoIndentation Processing\\Nanoindentation Energy Processing\\Average Ph Curve.xlsx')
    super_columns_list = list(set(excel_df.columns.get_level_values(0)))
    for count, sample in enumerate(super_columns_list):
        load_data = excel_df[sample]['Load (µN)']
        depth_data = excel_df[sample]['Depth (nm)']
        max_data.loc[count] = [sample, depth_data[load_data.idxmax()], load_data.max(), depth_data.max(), load_data[depth_data.idxmax()]]
        #depth_data.max() is the max depth 
    print(max_data.to_string())
        # loading_data.loc[count] = [sample, *fit_data(loading_function, excel_df[sample], max_data.iloc[count][3])] #TODO: Add the final Ut value
        # unloading_data.loc[count] = [sample, *fit_data(unloading_function, excel_df[sample], max_data.iloc[count][3])] #TODO: Add the final Uu value

def total_energy(loading_params, unloading_params, hmax) -> float:
    Ut = (loading_params[0]*hmax**(loading_params[1]+1))/(loading_params[1]+1)
    Uu = (unloading_params[0]*(hmax-unloading_params[2])**(unloading_params[1]+1))/(unloading_params[1]+1)
    return Uu

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

def fit_data(function_name, data:pd.DataFrame, hmax) -> list:
    model = odr.Model(function_name)
    data = odr.Data(x=data['Depth (nm)'], y=data['Load (µN)'])
    odr_instance = odr.ODR(data, model, beta0=[1.0, 1.0])
    output = odr_instance.run()
    return output.beta
    




if __name__ == '__main__':
    main()