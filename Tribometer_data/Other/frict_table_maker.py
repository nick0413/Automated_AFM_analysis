import pandas as pd
import re
# from natsort import natsorted
def main():
    with open('AverageFrictionCoefficientDataJune10.csv', 'r') as opened_csv:
        excel_output(opened_csv)
        
# def natural_sort_key(s):
#     return [int(text) if text.isdigit() else text.upper() for text in re.split('([0-9]+)', s)]


def excel_output(csv_file:str):
    columns_list = ['10N', '20N']
    speed_list = ['10mms', '20mms', '100mms']
    discriminators = {'Oils':['OA-0', 'OA-1', 'OA-10', 'OA-20'], 'Greases':['TOCN', 'C20A']} # else then just C20A for grease discriminator
    patterns = {}
    df_graphs_dict = {}

    for (key, value) in discriminators.items():
        patterns[key] = '|'.join(value)

    initial_average_cof = pd.read_csv(csv_file, header=None).transpose().rename(columns={0:'Name', 1:'CoF'}).dropna(how='any')
    initial_average_cof['CoF'] = pd.to_numeric(initial_average_cof['CoF']).astype('float64') # Changed every string CoF value to a float
    initial_average_cof['Name'] = initial_average_cof['Name'].str.replace('mms.*', 'mms', regex=True)
    initial_average_cof['Name'] = initial_average_cof['Name'].str.replace('TOCN10_C20A1_OA7.5', 'TOCN', regex=True)
    initial_average_cof['Name'] = initial_average_cof['Name'].str.replace('2wtOA', 'OA-2', regex=True)
    initial_average_cof['Name'] = initial_average_cof['Name'].str.replace(r'20wt_?', '', regex=True)

    # print(initial_average_cof.to_string())

    
    for (key, pattern) in patterns.items():
        for speed in speed_list:
            combined_pattern = f"({pattern}).*({speed})"
            df_graphs_dict[f'{key} {speed}'] = initial_average_cof[initial_average_cof['Name'].str.contains(combined_pattern)].drop_duplicates().reset_index(drop=True)

    for key in list(df_graphs_dict.keys()):
        if df_graphs_dict[key].empty == True:
            del df_graphs_dict[key]

    # with pd.ExcelWriter('FrictCoFData.xlsx') as writer:
    #     for (key, value) in df_graphs_dict.items():
    #         print(f'\n Here is the df for {key}\n{value}')
    #         value.to_excel(writer, sheet_name=key, index=False)

    # TODO Change CoF column to two columns 10N and 20N and remove all force and speed params from name
    # Take mean and std where name contains 20N and 10 then combine two dataframes

    final_df_list = []
    final_df_dict = {}
    merge_right = ''
    new_value = None

    #-------------------------------------------------------------------
    # Test 1 <-- checking to see if all the data is within all the dataframes
    data_length = 0
    for (key, value) in df_graphs_dict.items():
        data_length += len(value)
        new_value = value.groupby('Name')['CoF'].agg(['mean', 'std']).reset_index()
        # final_df_dict[key] = final_df
        # print(f'\n{key} \n {value.to_string()}\n')
    
        for column in columns_list:
            final_df_list.append(new_value[new_value['Name'].str.contains(column)].drop_duplicates().reset_index(drop=True))
            final_df_list[-1].columns = ['Name', f'{column} Mean', f'{column} STDEV'] #////////

            # if columns_list.index(column) == (len(columns_list) -1):
            #     merge_right = f'{column} Mean'
        
        for column in columns_list:
            for i in range(len(final_df_list)):
                final_df_list[i]['Name'] = final_df_list[i]['Name'].str.replace(f'_{column}.*', '', regex=True)
                # final_df_list[i].sort_values(by='Name', key=lambda col: col.map(natural_sort_key))
                # final_df_list[i].assign(f'{column} % Reduction' =lambda x:  )

        final_df_dict[key] = final_df_list[0].merge(final_df_list[1], on='Name', how='outer')
        final_df_list.clear()
    

    # with pd.ExcelWriter('ProcessedFrictCoFData.xlsx') as writer:
    for (key, value) in final_df_dict.items():
        # value.to_excel(writer, sheet_name=key, index=False)
        print(f'\n{key} \n {value.to_string()}\n')

        

    print(f'{data_length} = Num Data Columns Processed')

    # Test 1 ends
    #-------------------------------------------------------------------

    # df_oils = initial_average_cof[initial_average_cof['Name'].str.contains(oil_pattern)]
    # df_greases = initial_average_cof[initial_average_cof['Name'].str.contains(grease_pattern)]
    # df_greases_additional = initial_average_cof[initial_average_cof['Name'].str.contains('C20A')]
    # df_greases = pd.concat([df_greases, df_greases_additional]).drop_duplicates().reset_index(drop=True)

    # df_dict = {'Oils': df_oils, 'Greases': df_greases}

    # for (key, value) in df_dict:
    #     df_10N = initial_average_cof[initial_average_cof['Name'].str.contains('10N')] # Boolean indexing to find if 10N in string
    #     df_20N = initial_average_cof[initial_average_cof['Name'].str.contains('20N')] # Boolean indexing to find if 20N in string

if __name__ == "__main__":
    main()


