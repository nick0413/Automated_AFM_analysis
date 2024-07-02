import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import scipy.signal as signal

cutoff = 0.1

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
		print("first and last zeroes", first_zero_index, last_zero_index)	
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

def plot_CoF(upward_sections, forward_friction, downward_sections, backward_friction):
	"""
	Plots the Coefficient of Friction (CoF) for upward and downward sections of a tribology test.

	Parameters:
	- upward_sections (List[np.ndarray]): List of arrays representing the upward sections of the test.
	- forward_friction (List[np.ndarray]): List of arrays representing the friction in the forward direction for each upward section.
	- downward_sections (List[np.ndarray]): List of arrays representing the downward sections of the test.
	- backward_friction (List[np.ndarray]): List of arrays representing the friction in the backward direction for each downward section.
	"""
	for ii in range(len(upward_sections)):
		plt.plot(upward_sections[ii], forward_friction[ii], c='b',label=labels["upward_sections"] if ii == 0 else "")
		
	for ii in range(len(downward_sections)):
		plt.plot(downward_sections[ii], backward_friction[ii],c='r',label=labels["downward_sections"] if ii == 0 else "")
		# plt.show()
		
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

		
	return upward_sections, forward_friction, downward_sections, backward_friction




data,xpos,Fx,Fz=get_data('DATA\\1_IL_20N_20mms_Test1',300, True)



# print(data.head())

xpos_peaks, xpos_valleys = find_x_segments(xpos)

print("peaks and valleys",xpos_peaks, xpos_valleys)
upward_sections, forward_friction, downward_sections, backward_friction=segment_data(xpos, xpos_peaks, xpos_valleys,Fx,Fz)

plot_CoF(upward_sections, forward_friction, downward_sections, backward_friction)

def calculate_CoF(upward_sections, forward_friction, downward_sections, backward_friction):
	print("section lenght",len(upward_sections), len(downward_sections))

	pass


calculate_CoF(upward_sections, forward_friction, downward_sections, backward_friction)


# print((upward_sections[0]))

# max_length = len(upward_sections[0])
# for section in upward_sections:
# 	current_length = len(section)
# 	if current_length > max_length:
# 		max_length = current_length
# 	plt.plot(np.arange(0,current_length) ,section)

# for section in downward_sections:
# 	plt.plot(np.arange(max_length,len(section)+max_length) ,section)
	
	
# plt.xlim(0, 1000)
# plt.plot(xpos)
# plt.plot(xpos_peaks, xpos[xpos_peaks], "x")
# plt.plot(xpos_valleys, xpos[xpos_valleys], "x")
