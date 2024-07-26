import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import re
import scipy.signal as signal


'''
	Tribo is a work in progress module that is designed to process tribology data from a tribometer.

	Authored by: 	Nicolas Cordoba 	|| cordobagarzonnicolas@gmail.com
					Albert Kalayil 		|| 

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
		# print(f"{file_path}\tfirst and last zeroes {first_zero_index},{last_zero_index}\tRemoved {xpos.shape[0]-last_zero_index+first_zero_index} data points")	
		data = data.iloc[first_zero_index:last_zero_index+10]
		# if type(num_rows) != int and num_rows != 'all':
		# 	raise ValueError("num_rows must be an integer or a named value like 'all'")
		
		# data = data[data['x-Position'] >= 0].head(num_rows)
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
		for df,speed  in zip(speed_sheets_df,speeds): 
			rows_total=[]
			total_dfs=[]

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
			total_dfs.append(df)
			df.to_excel(writer, sheet_name=speed, index=False)


		complete_df=pd.concat(total_dfs)
		complete_df.to_excel(writer, sheet_name='Total', index=False)
		complete_CoF.to_excel(writer, sheet_name='CoF', index=True)
		sg_smoothing_df.to_excel(writer, sheet_name='Smoothed CoF', index=True)



# ----------------- MARK: Ex:loading data -----------------

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



