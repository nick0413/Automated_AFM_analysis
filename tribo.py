import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import re
import scipy.signal as signal
import os

# Refactored code finished

'''
	Tribo is a work in progress module that is designed to process tribology data from a tribometer.

	Authored by: 	Nicolas Cordoba 	|| cordobagarzonnicolas@gmail.com
					Albert Kalayil 		|| 79732912+AlbertKalayil@users.noreply.github.com
	# Refactored				
'''

def split_string(s):
    return re.split('[-_]', s)

labels = {
    "upward_sections": "Forward Movement CoF",
    "downward_sections": "Downward Movement CoF",
    "forward_friction": "Forward Friction",
    "backward_friction": "Backward Friction",
    # Add more labels as needed
}

def find_first_zero(xpos: np.ndarray)->int:
	"""
	Finds the index of the first zero in an array of x positions.

	Parameters:
	- xpos (np.ndarray): An array of x positions.

	Returns:
	- int: The index of the first zero in the array.
	"""
	return np.where(xpos == 0)[0][0]

def find_last_zero(xpos: np.ndarray)->int:
	"""
	Finds the index of the last zero in an array of x positions.

	Parameters:
	- xpos (np.ndarray): An array of x positions.

	Returns:
	- int: The index of the last zero in the array.
	"""
	return np.where(xpos == 0)[0][-1]

def get_data(file_path: str, num_rows: int=500, graph=False)->pd.DataFrame:
	"""
	Reads a tab-separated values (TSV) file and filters the data.

	Parameters:
		file_path (str): The path to the TSV file to be read. The file should have a header at the first line, and the data is expected to start from the second line.
		num_rows (int): The number of rows to read from the file
	Returns:
		pd.DataFrame: A pandas DataFrame containing the first 500 rows of data where 'x-Position' is greater than or equal to 0.
	"""
	
	try:
		data = pd.read_csv(file_path, sep='\t', header=1)
		
		
		if num_rows=='all' or num_rows==0:
			data = data[data['x-Position'] >= 0]
			
		elif num_rows>0:
			data = data[data['x-Position'] >= 0].head(num_rows)	

		xpos= data['x-Position'].to_numpy()
		first_zero_index = find_first_zero(xpos)
		last_zero_index = find_last_zero(xpos)
		data = data.iloc[first_zero_index:last_zero_index+10]

		xpos = data['x-Position'].to_numpy()
		Fx= data['Fx'].to_numpy()
		Fz= data['Fz'].to_numpy()

		if graph:
			plt.plot(xpos)
			plt.xlim(0, 1000)
			plt.show()

		return data, xpos, Fx, Fz
	
	except Exception as e:
		print(f"Error reading {file_path}: {e}")
		return pd.DataFrame() 

def find_x_segments(xpos: np.ndarray):
	"""
	Identifies the peaks and valleys in a given array of x positions.

	Parameters:
	- xpos (np.ndarray): An array of x positions.

	Returns:
	- Tuple[np.ndarray, np.ndarray]: Two arrays containing the indices of the peaks and valleys, respectively.
	"""
	xpos_peaks, _ = signal.find_peaks(xpos)
	xpos_valleys, _ = signal.find_peaks(-xpos) 
	
	# make sure the first point is a valley
	if xpos_valleys[0] > xpos_peaks[0]:
		xpos_valleys = np.insert(xpos_valleys, 0, 0)
	if xpos_valleys[-1] < xpos_peaks[-1]:
		xpos_peaks = np.append(xpos_peaks, len(xpos) - 1)

	return xpos_peaks, xpos_valleys

def plot_CoF(upward_sections, forward_friction, downward_sections, backward_friction,cutoff=None):
	"""
	Plots the Coefficient of Friction (CoF) for upward and downward sections of a tribology test.

	Parameters:
	- upward_sections (List[np.ndarray]): List of arrays representing the upward sections of the test.
	- forward_friction (List[np.ndarray]): List of arrays representing the friction in the forward direction for each upward section.
	- downward_sections (List[np.ndarray]): List of arrays representing the downward sections of the test.
	- backward_friction (List[np.ndarray]): List of arrays representing the friction in the backward direction for each downward section.
	"""

	all_x_values = np.concatenate(upward_sections + downward_sections)
	min_x, max_x = np.min(all_x_values), np.max(all_x_values)

	section_number = len(upward_sections)
	if section_number>100:
		print("section number",section_number)
		sparce=int(section_number/50)
	else: 
		sparce=1
	

	for ii in range(section_number):
		
		if ii%sparce==0:
			plt.plot(upward_sections[ii], forward_friction[ii], c='b',label=labels["upward_sections"] if ii == 0 else "")
			plt.plot(downward_sections[ii], backward_friction[ii],c='r',label=labels["downward_sections"] if ii == 0 else "")
		else:
			pass
		# plt.plot(upward_sections[ii], forward_friction[ii], c='b',label=labels["upward_sections"] if ii == 0 else "")
		
		# plt.plot(downward_sections[ii], backward_friction[ii],c='r',label=labels["downward_sections"] if ii == 0 else "")
		

	x_value_at_percentage = min_x + (max_x - min_x) * (cutoff )
	x2_value_at_percentage = min_x + (max_x - min_x) * (1-cutoff )
	plt.axvline(x=x_value_at_percentage, color='g', linestyle='--', label=f'{cutoff*100}% of x-axis')
	plt.axvline(x=x2_value_at_percentage, color='g', linestyle='--')
	plt.legend()
	plt.show()

def segment_data(xpos: np.ndarray, xpos_peaks: np.ndarray, xpos_valleys: np.ndarray, Fx: np.ndarray, Fz: np.ndarray):
	"""
	Segments the data into upward and downward sections based on peaks and valleys, and calculates the forward and backward friction.

	Parameters:
	- xpos (np.ndarray): An array of x positions.
	- xpos_peaks (np.ndarray): An array containing the indices of the peaks in xpos.
	- xpos_valleys (np.ndarray): An array containing the indices of the valleys in xpos.
	- Fx (np.ndarray): An array of force in the x direction.
	- Fz (np.ndarray): An array of force in the z direction.

	Returns:
	- List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]: Four lists containing the upward sections, forward friction, downward sections, and backward friction, respectively.
	"""
	upward_sections = []
	forward_friction= []
	downward_sections = []
	backward_friction = []
	

	for ii in range(len(xpos_valleys)-1):
		
		upward_sections.append(xpos[xpos_valleys[ii]:xpos_peaks[ii]])
		forward_friction.append(Fx[xpos_valleys[ii]:xpos_peaks[ii]]/Fz[xpos_valleys[ii]:xpos_peaks[ii]])

		downward_sections.append(xpos[xpos_peaks[ii]:xpos_valleys[ii+1]])
		backward_friction.append(Fx[xpos_peaks[ii]:xpos_valleys[ii+1]]/Fz[xpos_peaks[ii]:xpos_valleys[ii+1]])

		
	return [upward_sections, forward_friction, downward_sections, backward_friction]

def calculate_CoF(upward_sections, forward_friction, downward_sections, backward_friction, cutoff=0.1):
	# print("section lenght",len(upward_sections), len(downward_sections))
	
	forward_friction_avg=np.zeros(len(forward_friction))
	backward_friction_avg=np.zeros(len(backward_friction))
	CoF_avg=np.zeros(len(forward_friction))

	for ii in range(len(forward_friction)):
		forward_cutoff =int(forward_friction[ii].shape[0]*cutoff)
		backwad_cutoff =int(backward_friction[ii].shape[0]*cutoff)

		forward_friction_avg[ii]=np.mean(forward_friction[ii][forward_cutoff:-forward_cutoff])
		backward_friction_avg[ii]=np.mean(backward_friction[ii][backwad_cutoff:-backwad_cutoff])

		CoF_avg[ii]=(abs(forward_friction_avg[ii])+abs(backward_friction_avg[ii]))*0.5


	# print("forward friction",forward_friction_avg, "\nbackward friction",backward_friction_avg,"\nCoF",CoF_avg)


	return CoF_avg

class Tribo_file:
	def __init__(self,file_folder, file_name, outlier=False):
		self.file_name = file_name
		self.file_folder = file_folder
		self.percent,self.name,self.force,self.speed,self.test = split_string(self.file_name)
		self.outlier=outlier
		
		self.load_data()

	def load_data(self):
		self.data,self.xpos,self.Fx,self.Fz = get_data(self.file_folder+"\\"+self.file_name,"all")
		

	def process_data(self,cutoff=0.1):

		self.xpos_peaks, self.xpos_valleys = find_x_segments(self.xpos)

		segments=segment_data(self.xpos, self.xpos_peaks, self.xpos_valleys,self.Fx,self.Fz)

		self.upward_sections, self.forward_friction, self.downward_sections, self.backward_friction=segments

		self.CoF_array = calculate_CoF(self.upward_sections, self.forward_friction, self.downward_sections, self.backward_friction,cutoff)

		self.CoF_avg=np.average(self.CoF_array)
		self.CoF_std=np.std(self.CoF_array, ddof=1)

		pad_length = 3000 - len(self.CoF_array)
		if pad_length > 0:
			padded_array = np.pad(self.CoF_array, (0, pad_length), 'constant', constant_values=self.CoF_avg)
		else:
			padded_array = self.CoF_array

		self.CoF_array=padded_array

def load_files(files_in_folder: list,folder: str,outlier_tests: dict={}):
	Tribo_files = []
	complete_CoF_df = pd.DataFrame()
	for file_name in files_in_folder:
		if  outlier_tests:
			if file_name in outlier_tests[folder]:
			
				file_n=Tribo_file(folder, file_name, outlier=True)
			
			else:
				file_n=Tribo_file(folder, file_name)
		else:
			file_n=Tribo_file(folder, file_name)

		file_n.process_data(0.2)
		Tribo_files.append(file_n)
		
		complete_CoF_df[file_name]=file_n.CoF_array	

	return Tribo_files, complete_CoF_df

def smoothing_df(data_array: pd.DataFrame):
	sg_smoothing = pd.DataFrame()
	for column in data_array.columns:
		sg_smoothing[column]=savgol_filter(data_array[column], 100, 2)

	return sg_smoothing

def get_speeds_and_names_in_folder(files_in_folder: list, Tribo_files_list: list):
	speeds=[]
	names=[]

	for file in Tribo_files_list:
		if file.speed not in speeds:
			speeds.append(file.speed)

		if file.name not in names:
			names.append(file.name)

	return speeds, names

def remove_outliers(files: list, excluded_speeds: list =[], verbose: bool = False):
	rows=[]
	test_list = [] 
	count1 = 0
	for file in files:
		if file.outlier:
			count1 += 1
		elif file.speed in excluded_speeds:
			continue 
		else:
			data_to_append = [ file.name,  file.CoF_avg,  file.CoF_std,file.speed, file.percent, file.force,file.test]
			rows.append(data_to_append)

		if verbose: 
			print('Outliers num: ', count1)
			print('Outlier Files found: ', test_list)
	
	return rows

def sort_dfs_by_speed(speeds: list, df: pd.DataFrame):
	'''
	This returns a list of dataframes sorted by speed
	'''
	
	speed_sheets=[]
	for speed in speeds:
		speed_sheet=df[df['Speed']==speed]
		print("speed sheet\n",speed_sheet)
		speed_sheets.append(speed_sheet)

	return speed_sheets

def export_excel_results(speed_sheets_df,speeds,sg_smoothing_df,folder,complete_CoF,verbose: bool = False):
	with pd.ExcelWriter(f'{folder}.xlsx') as writer: 
		total_dfs=[]
		for df,speed  in zip(speed_sheets_df,speeds): 
			rows_total=[]
			

			for name in df['Name'].unique():	
				for force in df['Force'].unique(): 
					
					if verbose: print(f"----------{name}\t{force}------------")

					df2=df[(df['Name']==name) & (df['Force']==force)]
					# Here we want to combine all of the tests for the same name and force

					if df2.empty: 
						continue #it is possible that there are no tests for a given combination of name and force

					CoF_total_avg=np.average(df2['CoF_avg'])
					CoF_total_std=np.sqrt(np.sum(df2['CoF_std'].to_numpy()**2))/len(df2['CoF_std'])
					

					row=[name,CoF_total_avg,CoF_total_std,speed,df2.get("Force").iloc[0],df2.get("Percent").iloc[0]]
					rows_total.append(row)
			df=pd.DataFrame(rows_total,columns=['Name','CoF_avg','CoF_std','Speed', 'Force','Percent'])
			print(speed)
			total_dfs.append(df)
			df.to_excel(writer, sheet_name=speed, index=False)


		complete_df=pd.concat(total_dfs)
		complete_df.to_excel(writer, sheet_name='Total', index=False)
		complete_CoF.to_excel(writer, sheet_name='CoF', index=True)
		sg_smoothing_df.to_excel(writer, sheet_name='Smoothed CoF', index=True)


def cof_overlays(smoothed_df):
	"""
	Plotting Coefficient of Friction (CoF) vs Cycle number for all the tests in the dataframe.
	Args:
		smoothed_df (pandas.DataFrame): Dataframe with columns containing the smoothed CoF values for all the tests.
	Returns:
		None: CoF vs Cycle number overlays in one plot which contains all the tests in the dataframe.
	"""

	for column in smoothed_df.columns: # iterates over all the column names in the list of column names
		plt.plot(smoothed_df[column],label=column) # plots the datapoints within the column on the graph
		plt.legend() # adds a legend to the graph
	plt.show() # displays the graph


def plot_combinations_of_params(smoothed_df, folder):
	
	"""
	A function that plots a Coefficient of Friction vs Cycles graph for each combination of parameters of a sample in the dataframe.
	:param smoothed_df: A pandas DataFrame containing the data for the tests.
	:param folder: A string representing the folder path where the generated images will be saved.
	:return: None
    :Ouput Graphs (plt.plot)

    The graph will contain all the tests that have the same combination of parameters.

    The purpose of this function is to identify the outlier tests that may be contributing to the high standard deviation in the Average CoF values.
    Additionally, this function will allow one to see the general trend of the CoF values for each combination of parameters for each sample.
    
    :param 
    """

	pattern = r'(_test|_Test).*' # Regex pattern to remove the test number from the column name
	unique_names = {}  # Dictionary to store the unique names of the tests

	for column in smoothed_df.columns: # Iterate over all the column names in the list of column names
		if '10mms' in column: # Skip the column if it contains '10mms' in the name <-- we had considered it an invalid test
			continue
		if '20N' or '10N' in column:
		# if '20N' in column:

			unique_name = re.sub(pattern, '', column) # Apply the regex pattern to get the unique name
			if unique_name in unique_names: # Check if the unique name is already a key in the dictionary
				unique_names[unique_name].append(column) # Append the original column name to the list associated with the unique name
			else:
				unique_names[unique_name] = [column] # Create a new entry with the unique name as the key and a list containing the original column name

	os.makedirs(f'{folder}_Images', exist_ok=True) # make a new directory to store the images if it doesn't already exist

	for (key, value) in unique_names.items(): # keys are the unique names which include sample name along with the specific combination of parameters
		# values are the list of tests (repetitions) done on the same combination of parameters
		for spec_val in value: # iterates through tests within the list of tests done on the same combination of parameters
			plt.plot(smoothed_df[spec_val],label=spec_val) # plot the CoF values vs cycle number with the legend being the specific file name of the test
		# plt.title(key)
		plt.legend() # includes legend in the graph
		plt.savefig(os.path.join(f'{folder}_Images', f'{key}.png')) # saves the figure as the unique name in the previously created folder
		plt.show() # displays the graph
	# print(key, unique_names[key])


def output_good_tests(avg_CoF_tests, folder, sg_smoothing_df):
	"""
	Outputs the good tests from the given average coefficient of friction (CoF) tests.
	Parameters:
	- avg_CoF_tests (dict): A dictionary containing the good tests from each of the combinations of parameters.
	- folder (str): The folder name for which the good tests need to be extracted.
	- sg_smoothing_df (pandas.DataFrame): The DataFrame containing the smoothed data.
	Returns:
	- None
	Prints the good tests found in the specified folder and saves them to an Excel file.
	Example usage:
	output_good_tests(avg_CoF_tests, 'Folder1', sg_smoothing_df)
	"""

	print(avg_CoF_tests[folder], f'\n Length: {len(avg_CoF_tests[folder])}')
	count = 0
	good_test_in_folder_list = []
	for ii in range(len(sg_smoothing_df.columns)):
		for jj in range(len(avg_CoF_tests[folder])):
			# print(  avg_CoF_tests[folder][jj]==sg_smoothing.columns[ii])
			if  avg_CoF_tests[folder][jj]==sg_smoothing_df.columns[ii]:
				count += 1
				print(avg_CoF_tests[folder][jj])
				good_test_in_folder_list.append(avg_CoF_tests[folder][jj])
	print(count)

	good_test_df = sg_smoothing_df.loc[:, good_test_in_folder_list]

	# for good_test in avg_CoF_tests[folder]:
	#     good_test_df = pd.concat([good_test_df, sg_smoothing[good_test]], ignore_index=True)
	print(good_test_df)

	with pd.ExcelWriter(f'{folder}.xlsx', engine='openpyxl', mode='a') as writer: 
		good_test_df.to_excel(writer, sheet_name='Good_Tests', index=False)
		# good_test_df.to_excel(writer, sheet_name='Good Tests', index=False)


#-------------------MARK: friction table making code------------------
# Refactored code for the previous code block

# Constants
FORCE_OPTION = 'f'
SPEED_OPTION = 's'
YES_OPTION = 'y'
NO_OPTION = 'n'
PERCENT_COLUMN = 'Percent'
NAME_COLUMN = 'Name'
COF_AVG_COLUMN = 'CoF_avg'
COF_STD_COLUMN = 'CoF_std'
TOTAL_SHEET_NAME = 'Total'

def display_initial_message():
	"""
	Displays the initial message to the user.
	This function prints a message that explains the purpose of the code block. It states that the code will reorganize the "total" sheet into a friction table based on the user input, which includes percent reduction for ease of integration into origin.
	"""
	print('This block of code will reorganize the "total" sheet into a friction table based on the user input which will \n' \
          'include percent reduction for ease of integration into origin.')

def read_total_dataframe(folder: str) -> pd.DataFrame:
	"""
	Reads the 'Total' sheet from the specified Excel file.
	Parameters:
	folder (str): The path to the folder containing the Excel file.
	Returns:
	pd.DataFrame: The data from the 'Total' sheet as a pandas DataFrame.
	"""
	return pd.read_excel(f'{folder}.xlsx', sheet_name=TOTAL_SHEET_NAME)

def get_user_input(prompt: str, valid_options: list) -> str:
	"""
	Gets validated user input based on the provided prompt and valid options.
	Args:
		prompt (str): The prompt message displayed to the user.
		valid_options (list): A list of valid options that the user input should match.
	Returns:
		str: The user input that matches one of the valid options.
	"""

	user_input = input(prompt)
	while user_input not in valid_options:
		print(f'Invalid input. Please enter one of the following: {", ".join(valid_options)}.')
		user_input = input(prompt)
	return user_input

def get_table_title() -> str:
	"""
	Gets the table title from the user.
	Returns:
		str: The table title entered by the user.
	"""

	return get_user_input('Would you like to create a friction table based on force or speed? (only type f for force or s for speed): ', [FORCE_OPTION, SPEED_OPTION])

def get_include_percent() -> str:
    """
	Gets the user input for including the percentages in the friction table.
	Returns:
	str: The user input for including the percentages in the friction table.
	"""
    return get_user_input('Would you like to include the percentages in the names in the friction table? (y/n): ', [YES_OPTION, NO_OPTION])

def get_reference_sample(total_dataframe: pd.DataFrame) -> str:
	
    """
	Gets the reference sample from the user.
	This function prompts the user to enter the reference sample for the friction table. The reference sample is the sample that the user is measuring the percent reduction with respect to. The function checks if the entered reference sample is valid by comparing it with the values in the 'NAME_COLUMN' of the 'total_dataframe'. If the entered reference sample is not valid, the function displays an error message and prompts the user to enter a valid sample name.
	Parameters:
		total_dataframe (pd.DataFrame): The dataframe containing the data in the total sheet in the excel file.
	Returns:
		str: The reference sample entered by the user.
	"""
    ref_sample = input('Please enter the reference sample for the friction table: (sample that you are measuring the percent reduction with respect to)\n'
                       'If you included the percentage in the names, please include the percentage in the reference name as well: ')
	# just input the sample name (ie. C20A not 10-C20A_20N_100mms_test1)
    if ref_sample not in total_dataframe[NAME_COLUMN].values:
        print('Invalid input. Please enter a valid sample name')
        return get_reference_sample(total_dataframe)
    return ref_sample

def process_dataframe(total_dataframe: pd.DataFrame, table_title: str, include_percent: str, reference_sample: str) -> dict:
	"""
	Processes the total dataframe and returns a dictionary of organized dataframes.
	Args:
		total_dataframe (pd.DataFrame): The total dataframe to be processed.
		table_title (str): The title of the table to be organized.
		include_percent (str): Option to include percent reduction in the organized table.
		reference_sample (str): The reference sample used for comparison.
	Returns:
		dict: A dictionary containing organized dataframes.
	Raises:
		None
	Examples:
		>>> df = pd.DataFrame(...)
		>>> result = process_dataframe(df, 'Force', 'Yes', 'Sample A')
		>>> print(result)
		{'Force Organized Table': pd.DataFrame(...), ...}
	
	"""
	if include_percent == YES_OPTION:
		total_dataframe[NAME_COLUMN] = total_dataframe[PERCENT_COLUMN].astype(str) + '%' + ' ' + total_dataframe[NAME_COLUMN]

	possible_params = {FORCE_OPTION: 'Force', SPEED_OPTION: 'Speed'}
	dataframe_dict = {}
	other_unique_params = {}

	unique_param_titles = total_dataframe[possible_params[table_title]].unique()

	for key, value in possible_params.items():
		if key != table_title:
			other_unique_params[value] = total_dataframe[value].unique()

	for param in unique_param_titles:
		param_df = total_dataframe[total_dataframe[possible_params[table_title]] == param].drop(columns=[possible_params[table_title]])
		final_df = pd.DataFrame()

		for other_param, unique_values in other_unique_params.items():
			for param2 in unique_values:
				temp_df = param_df[param_df[other_param] == param2].drop(columns=[other_param])
				if include_percent == NO_OPTION:
					temp_df = temp_df.drop(columns=[PERCENT_COLUMN])

				reference_value = temp_df[temp_df[NAME_COLUMN] == reference_sample][COF_AVG_COLUMN].values[0]
				percent_reduction_df = temp_df[COF_AVG_COLUMN].apply(lambda x: 'REF' if x == reference_value else ((reference_value - x) / reference_value) * 100)

				temp_df = temp_df.rename(columns={COF_AVG_COLUMN: f'{param2}', COF_STD_COLUMN: f'{param2} STDEV'})
				temp_df.insert(2, f'% Percent Reduction {param2}', percent_reduction_df)

				reference_row = temp_df[temp_df[NAME_COLUMN] == reference_sample]
				temp_df = temp_df[temp_df[NAME_COLUMN] != reference_sample]
				temp_df = pd.concat([reference_row, temp_df], ignore_index=True)

				if final_df.empty:
					final_df = temp_df
				else:
					final_df = pd.merge(final_df, temp_df, how='left', on=NAME_COLUMN)

		dataframe_dict[f'{param} Organized Table'] = final_df

	return dataframe_dict

def save_to_excel(dataframe_dict: dict, folder: str):
    """Saves the organized dataframes to an Excel file."""
    # refactored_sheet_name = folder+'_refactored'
    with pd.ExcelWriter(f'{folder}.xlsx', engine='openpyxl', mode='a') as writer:
        for key, value in dataframe_dict.items():
            value.to_excel(writer, sheet_name=key, index=False)
    
    print(f'The friction tables have been successfully saved to {folder}.xlsx.')

# Main execution


def frict_table_maker(folder):
	"""
	Generates a friction table based on the data in the specified folder.
	Args:
		folder (str): The path to the folder containing the data.
	Returns:
		None
	Output:
		Adds sheet to pre-existing excel file with the friction tables based on the user input.
	"""

	display_initial_message()
	total_dataframe = read_total_dataframe(folder)
	table_title = get_table_title()
	include_percent = get_include_percent()
	reference_sample = get_reference_sample(total_dataframe)
	dataframe_dict = process_dataframe(total_dataframe, table_title, include_percent, reference_sample)
	save_to_excel(dataframe_dict, folder)

#-------------------MARK: friction table making code------------------

# file_name='DATA\\OA-10_10N_100mms_test4_May8'
# data,xpos,Fx,Fz=get_data(file_name,"all", True)





# xpos_peaks, xpos_valleys = find_x_segments(xpos)

# print("peaks and valleys",xpos_peaks, xpos_valleys)
# upward_sections, forward_friction, downward_sections, backward_friction=segment_data(xpos, xpos_peaks, xpos_valleys,Fx,Fz)

# plot_CoF(upward_sections, forward_friction, downward_sections, backward_friction, cutoff)


# reported_CoF=calculate_CoF(upward_sections, forward_friction, downward_sections, backward_friction,cutoff)


# plt.plot(reported_CoF)
# plt.show()	

# np.savetxt(file_name+"_reported_CoF.csv", reported_CoF, delimiter=',', header="CoF at each cycle")   # X is an array



