import os
import re
import pandas as pd

def main():
    # Directory containing the files
    directory = 'C:\\Users\\alber\\Summer Research Code\\Automated_AFM_analysis\\Tribometer_data\\OA_Project_Oils'
    
    # Get all files in the directory
    files_in_folders = get_files(directory)
    new_files_list = []
    original_files = []

    for root, files in files_in_folders.items():
        original_files += files
        for file in files:
            original_name = file[:]
            file_base, file_extension = os.path.splitext(original_name)  # Preserve file extension
            new_name = convert_naming_convention(file_base) + file_extension
            while new_name in new_files_list:
                new_name = new_name + 'copy'
            new_files_list.append(new_name)
            new_name = convert_naming_convention(original_name)
            # print(os.path.exists(os.path.join(root, original_name)))
            # print(os.path.exists('C:\\Users\\alber\\Summer Research Code\\Automated_AFM_analysis\\Tribometer_data\\OA_Project_Oils\\Sample 13\\PAO4_OA-20_10N_100mms_test4_May9'))
            os.rename(os.path.join(root, original_name), os.path.join(root, new_name))

    # print(new_files_list)
    files_dict = {'Old Name': original_files, 'New Name': new_files_list}
    df = pd.DataFrame(files_dict)
    export_excel(df, directory)
    print(df.to_string())
    # Rename all files in the directory
        

def export_excel(df, directory):
    # Export the dataframe to an Excel file
    excel_path = os.path.join(directory, 'Renamed_files.xlsx')
    df.to_excel(excel_path, index=False)

def get_files(directory):
    # Get all files in the directory
    files_in_folder = {}
    for root, dirs, files in os.walk(directory):
            if not files:
                continue
            for file in files:
                if file.endswith('.xlsx'):
                    files.remove(file)
            files_in_folder[root] = files
    return files_in_folder

def convert_naming_convention(original_name):
    # Extract the sample (PAO4+OA), percent (-20), load (10N), speed (20mms), test number, and date
    sample = re.search(r'^([A-Z]+\d+)_(\w+)', original_name)
    percent = re.search(r'-(\d+)', original_name)
    load = re.search(r'(\d+N)', original_name)
    speed = re.search(r'(\d+mms)', original_name)
    test_number = re.search(r'(test\d+)', original_name)
    date = re.search(r'(\w+\d+)$', original_name)
    
    # Combine the sample components
    combined_sample = f"{sample.group(1)}+{sample.group(2)}"
    
    # Construct the new name
    new_name = f"{percent.group(1)}-{combined_sample}_{load.group(1)}_{speed.group(1)}_{test_number.group(1)}"
    return new_name

# # Example usage
# original_name = 'PAO4_OA-20_10N_20mms_test2_May8'
# new_name = convert_naming_convention(original_name)
# print(new_name)  # Output: 20-PAO4+OA_10N_20mms.May8.test2

if __name__ == '__main__':
    main()