import agilent_loader as ag
import os 
import sys
import re 
import numpy as np
import agilent_loader as ag
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import scipy.optimize
import warnings
import shutil
import copy

meters_conversion_dictionary={
	'm':1,
	'um':1e6,
	'nm':1e9,
	'pm':1e12
	
}
latex_units_dictionary={
	'm':r'$m$',
	'um':r'$\mu m$',
	'nm':r'$nm$',
	'pm':r'$pm$'
	

}
newton_conversion_dictionary={
	'N':1,
	'uN':1e6,
	'nN':1e9,
	'pN':1e12,
	'V':1

}


def fit_image_to_polynomial(image, degree):
	'''
	Fits a 2D image to a polynomial of a given degree.

	Parameters
	----------
	image: np.ndarray
		2D array of image data
	degree: int
		Degree of the polynomial to fit to the image
	'''
	
	y, x = np.indices(image.shape)
	x = x.flatten()
	y = y.flatten()
	z = image.flatten()  

	
	def poly_model(coeffs, x, y, degree, z=None):

		idx = 0
		model = np.zeros_like(x, dtype=np.float64)
		for i in range(degree + 1):
			for j in range(degree + 1 - i):
				model += coeffs[idx] * (x ** i) * (y ** j)
				idx += 1
		return model - z if z is not None else model

	
	def num_coeffs(degree):
		return (degree + 1) * (degree + 2) // 2

	
	initial_guess = np.zeros(num_coeffs(degree))
	res = scipy.optimize.least_squares(poly_model, initial_guess, args=(x, y, degree, z))

	
	fitted_image = poly_model(res.x, x, y, degree).reshape(image.shape)

	return fitted_image, res.x


def check_and_prepare_folder(folder_path):
	'''
	Checks if a folder exists and is empty. If it does not exist, it creates it. If it exists and is not empty, it deletes all files and subfolders.

	Parameters
	----------
	folder_path: str
		The path to the folder to be checked and prepared
	
	'''
	
	if not os.path.exists(folder_path):						# Create the folder if it does not exist
		
		os.makedirs(folder_path)
		print(f"Folder '{folder_path}' created.")

	else:															# Delete all files and subfolders if the folder is not empty || Make sure you dont mix previous with new results										
	
		for filename in os.listdir(folder_path):
			file_path = os.path.join(folder_path, filename)
			try:
				if os.path.isfile(file_path) or os.path.islink(file_path):
					os.unlink(file_path)  
				elif os.path.isdir(file_path):
					shutil.rmtree(file_path) 

			except Exception as e:
				print(f'Failed to delete {file_path}. Reason: {e}')


		print(f"Folder '{folder_path}' already exists and is now empty.")


def get_current_mi_files_in_folder(folder):
	'''
	Returns a list of all .mi files in a folder.

	Parameters
	----------
	folder: str
		The path to the folder to search for .mi files in
	'''
	return [f for f in os.listdir(folder) if '.mi' in f]


def get_mi_files_in_folder(folder):
	'''
	Returns the list of all .mi files in a folder with file renaming.
	The function replaces spaces with underscores and removes "@" from the filenames so this is also compatible with older files. 

	Parameters
	----------
	folder: str
		The path to the folder to search for .mi files in

	
	'''
	#TODO Add a check for the file extension
	#TODO Add string comprehension for the file renaming, check units and add a warnings if the file doesnt contain the expected units or values.

	files_in_folder = get_current_mi_files_in_folder(folder)

	for filename in files_in_folder:
		# Generate the new filename by replacing spaces with underscores and removing "@"
		new_filename = filename.replace(" ", "_").replace("@", "")
		
		# Construct the full old and new file paths
		old_file_path = os.path.join(folder, filename)
		new_file_path = os.path.join(folder, new_filename)
		
		# Rename the file
		os.rename(old_file_path, new_file_path)

	files_in_folder = get_current_mi_files_in_folder(folder)
	return files_in_folder





def graph_friction_n_topography(file, averaged_friction: np.ndarray, topography: np.ndarray,results_folder: str,file_path:str, title:str,resolution=300,aspect_ratio=(10,5),poly_degree=2 , scale_length=1,friction_color_range=2, show=False,current=None, bar_position=(0.8,0.1), scale_unit='um', axis_ticks=False, axis_labels=False, scale_factor=None,conversion_factor=None,Normal_force=None, force_unit='nN'):
	

	'''
	Prints the friction and topography images side by side with a scale bar on the top right corner of the image. The scale bar is in the units of the scale_unit parameter. The friction image is scaled to the average friction value +- the friction_color_range*standard deviation of the friction values. The images are saved to a file in the results_folder with the name Friction_force_and_topography_file_path.png

	Parameters
	----------
	file: agilent_loader.MiFile
		The MiFile object to be graphed
	averaged_friction: np.ndarray
		The array of averaged friction values
	topography: np.ndarray
		The array of topography values
	results_folder: str
		The folder to save the results
	file_path: str
		The path to the file to be processed
	title: str
		The title of the plot
	resolution: int
		The resolution of the plot
	aspect_ratio: tuple
		The aspect ratio of the plot
	poly_degree: int
		The degree of the polynomial to fit to the image
	scale_length: float
		The length of the scale bar in the units of m*scale_factor
	friction_color_range: float
		The range of the friction color scale in terms of standard deviations from the mean
	show: bool
		Whether to show the plot or not
	current: np.ndarray
		The array of current values, to be used if the user wants to plot the current in Amps as well
	bar_position: tuple
		The position of the scale bar in the plot, as a fraction of the image size, to be read as (x% from the left, y% from the bottom). Default is (0.8,0.1)
	scale_unit: str
		The unit of the scale bar. Default is 'um' for micrometers, can also be 'nm' or any other string, made to recive LaTex strings as well.
	axis_ticks: bool
		Whether to show the axis ticks or not
	axis_labels: bool
		Whether to show the axis labels or not
	scale_factor: float
		The scale factor to change the units to arbitrary units, it change the scale in the way of 
		new extent = old extent * scale_factor. Remember .mi files stores data in meters.



	
	'''
	# F_normal=-spring_k*setpoint

	# force_unit='V'
	if scale_unit not in meters_conversion_dictionary.keys():
		raise Exception(f"Scale unit {scale_unit} not recognized, please use one of the following: {meters_conversion_dictionary.keys()}")
	
	if scale_factor is None:
		print(type(scale_factor))
		scale_factor=meters_conversion_dictionary[scale_unit]
	else:
		print(f"Using custom scale factor {scale_factor} to convert units while using {scale_unit} as the unit, this can lead to unexpected results")
	

	average_friction_value=np.average(averaged_friction)

	print(averaged_friction)
	plt.imshow(averaged_friction, cmap='inferno', extent=file.extent)
	plt.title(title)
	plt.colorbar()
	plt.show()
	plt.clf()
	plt.close()


	if conversion_factor is not None:
		averaged_friction=averaged_friction*conversion_factor
		average_friction_value=np.average(averaged_friction)
		print('finding conversion factor')
		for factor in newton_conversion_dictionary.keys():
			if average_friction_value*newton_conversion_dictionary[factor]>1:
				force_unit=factor
				averaged_friction=averaged_friction*newton_conversion_dictionary[force_unit]
				break
	else:
		force_unit='V'
		# continue
	
	
		

	


	
	if current is not None:

		fig,ax=plt.subplots(1,3,figsize=aspect_ratio,dpi=resolution)
	else:
		fig,ax=plt.subplots(1,2,figsize=aspect_ratio,dpi=resolution)

	file= center_sample(file, scale_factor)
	topography=topography*1e9
	
	fit_topology, _ = fit_image_to_polynomial(topography, poly_degree)
	topography=topography-fit_topology
	topography=topography-np.min(topography)

	

	if current is not None: 
		im3=ax[2].imshow(current,cmap='inferno', extent=file.extent)
		ax[2].set_title(f'Current avg: {np.average(current):4f}')
		if not axis_ticks:
			ax[2].set_xticks([])
			ax[2].set_yticks([])



	

	average_friction_value=np.average(averaged_friction)
	im1=ax[0].imshow(averaged_friction, cmap='inferno', extent=file.extent)
	im2=ax[1].imshow(topography, cmap='inferno', extent=file.extent)
		
	ax[0].set_title(f'Friction avg: {np.average(averaged_friction):.2f}{force_unit}')


	if not axis_ticks:
		ax[0].set_xticks([])
		ax[0].set_yticks([])

	print("graphing topography")

	ax[1].set_title('Topography')
	if not axis_ticks:
		ax[1].set_xticks([])
		ax[1].set_yticks([])

	print('2',scale_length)

	scale_length_um = scale_length
	print(scale_length_um, 'scale length',scale_length)
	x_pad=bar_position[0]
	y_pad=bar_position[1]
	x_low=file.extent[1]*x_pad
	y_low=file.extent[3]*y_pad


	scale_bar1 = Line2D([x_low, x_low+ scale_length_um], [y_low,y_low], color='white', linewidth=3)
	scale_bar2 = Line2D([x_low, x_low+ scale_length_um], [y_low,y_low], color='white', linewidth=3)
	
	scale_unit=latex_units_dictionary[scale_unit]
	ax[0].add_line(scale_bar1)
	ax[0].text(x_low+ scale_length_um/2, y_low, f'{scale_length_um} {scale_unit}', color='white', ha='center', va='bottom')
	ax[1].add_line(scale_bar2)
	ax[1].text(x_low+ scale_length_um/2, y_low, f'{scale_length_um} {scale_unit}', color='white', ha='center', va='bottom')

	if current is not None:
		scale_bar3 = Line2D([x_low, x_low+ scale_length_um], [y_low,y_low], color='white', linewidth=3)
		ax[2].add_line(scale_bar3)
		ax[2].text(x_low+ scale_length_um/2, y_low, f'{scale_length_um} {scale_unit}', color='white', ha='center', va='bottom')


	friction_std=np.std(averaged_friction)
	

	# im1.set_clim(vmin=average_friction_value-friction_color_range*friction_std, vmax=average_friction_value+friction_color_range*friction_std)

	cbar1=fig.colorbar(im1,ax=ax[0],fraction=0.046, pad=0.04)
	cbar2=fig.colorbar(im2,ax=ax[1],fraction=0.046, pad=0.04)

	if current is not None:
		cbar3=fig.colorbar(im3,ax=ax[2],fraction=0.046, pad=0.04)


	



	cbar1.set_label(f"Friction force [{force_unit}]")
	cbar2.set_label("Height $[ nm]$")

	plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
	plt.savefig(results_folder+f"\\Friction_force_and_topography_{file_path}.png")
	if show:
		print("showing")
		plt.show()
		
	
	
	plt.clf()
	plt.close()


def center_sample(file, scale_factor=1e6):
	'''
	turns the lower left corner of the sample to (0,0) and shits the extent accordingly
	Also changes the units from m to arbitrary units given a scale factor
	
	Parameters
	----------
	file: agilent_loader.MiFile
		The MiFile object to be centered
	scale_factor: float
		The scale factor to change the units to arbitrary units, it change the scale in the way of 
		new extent = old extent * scale_factor

	
	'''
	file.extent[1]=file.extent[1]-file.extent[0]
	file.extent[0]=0
	file.extent[3]=file.extent[3]-file.extent[2]
	file.extent[2]=0

	file.extent[0]=file.extent[0]*scale_factor
	file.extent[1]=file.extent[1]*scale_factor
	file.extent[2]=file.extent[2]*scale_factor
	file.extent[3]=file.extent[3]*scale_factor
	return file


def plot_CoF(Cof_for_runs,Cof_for_runs_std,results_folder, show=False):
	'''
	Plots the Coefficient of Friction as a function of cycles over the sample with error areas for the standard deviation, saves the plot to a file.

	Parameters
	----------
	Cof_for_runs: np.ndarray
		Array of Coefficient of Friction values
	Cof_for_runs_std: np.ndarray
		Array of Coefficient of Friction standard deviations
	results_folder: str
		The folder to save the results
	'''
	average_friction_value=np.average(Cof_for_runs)

	
	Cof_for_runs=Cof_for_runs
	Cof_for_runs_std=Cof_for_runs_std
	x_axis=np.arange(len(Cof_for_runs))
	plt.figure(figsize =(10, 5),dpi=300) 
	plt.plot(x_axis,Cof_for_runs)
	plt.fill_between(x_axis,Cof_for_runs-Cof_for_runs_std,Cof_for_runs+Cof_for_runs_std,alpha=0.5)
	plt.title("Friction force as a function of cycles")
	plt.xlabel("Cycles over the sample")
	plt.ylabel(f"Friction Coefficient")
	plt.savefig(results_folder+"\\Friction_force_for_cycles.png")
	# plt.ylim(-1,10)
	if show:
		plt.show()
	plt.clf()
	plt.close()
	return


def load_buffers_from_file(file):
	'''
	Loads the friction and topography buffers from a MiFile object.
	Cycles through the buffers in the MiFile object and appends the friction and topography buffers to separate lists, ingoring the other buffers.
	
	Parameters
	----------
	file: agilent_loader.MiFile
		The MiFile object to load the buffers from
	
	'''
	friction_arrays=[]
	topography_arrays=[]
	current_arrays=[]

	results=[]


	for buffer in file.buffers:
		
		if buffer.bufferLabel == "Friction":
			friction_arrays.append(buffer.data)
			
		elif buffer.bufferLabel == "Topography":
			topography_arrays.append(buffer.data)

		elif buffer.bufferLabel == "CSAFM/Aux_BNC":
			current_arrays.append(buffer.data)
		
	if friction_arrays:
		results.append(friction_arrays)
	if topography_arrays:
		results.append(topography_arrays)
	if current_arrays:
		results.append(current_arrays)



	return results

def calculate_CoF(friction_array: list[np.ndarray],file_path: str, conversion_factor=None, Normal_force=None):
	'''
	Calculates the coefficient of friction for the cycles over a sample, returns the averaged CoF, the mean and the standard deviation of the CoF.

	Parameters
	----------
	friction_array: list[np.ndarray]
		List of arrays of friction data
	file_path: str
		The path to the file to be processed

	'''
	if conversion_factor is None:
		conversion_factor=1
	
	if len(friction_array)==2:
		

		averaged_friction = ((friction_array[1]) - (friction_array[0]))*0.5

		print(averaged_friction)

		# if conversion_factor is not None:
		# 	friction_array[0]=friction_array[0]*conversion_factor
		# 	friction_array[1]=friction_array[1]*conversion_factor

			
		Cof_std=np.std(averaged_friction*conversion_factor)
		Cof=np.mean(averaged_friction*conversion_factor)
		return averaged_friction,Cof,Cof_std

	else:
		warnings.warn(f"{file_path} doesnt contain both trace and retrace friction chunks\nExpected 2 friction arrays, got %d" % len(friction_array)+f" with file {file_path}")
		
		raise Exception(f"{file_path} doesnt contain both trace and retrace friction chunks\nExpected 2 friction arrays, got %d" % len(friction_array)+f" with file {file_path}\n "+
				  "the most likly culpurit is a spectroscopy file. Please check the file and try again")
		



