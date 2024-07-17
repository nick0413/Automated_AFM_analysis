import numpy as np
import os
from types import SimpleNamespace
import re

''' === AGILENT MI LOADER ===
	v0.1 2024-06-23
	
	Kim Lambert      kim.lambert@uni-goettingen.de
	Sascha Lambert   sascha.lambert@phys.uni-goettingen.de
'''


''' CORE LOADING FUNCTIONS
------------------------------------------------------------------------------------------------------
	These functions parse the input files.
'''

def find_binary_data_offset(filename):
	'''
		Find the offset given the position of the "data" keyword in the file.
		
		divinding the file from the number of pixels and the data type and starting from the end,
		I found can lead to data from a buffer ending up in the next buffer.

		As i understand it, the data keyword marks the end of the ascii header and the beginning of the binary data.
		Thats what this function is for. It returns the offset in bytes where the binary data starts.
		
	'''
	with open(filename, 'rb') as file:  
		while True:
			line = file.readline()
			if b"data" in line:  
				break  
	
		offset = file.tell()  
	return offset


def _load_mi_image(filename, allow_16_bit=False):

	''' filename = .mi file to load.
		allow_16_bit is untested, might produce garbage images
		
		This function returns a Namespace with all the data from the
		.mi file parsed.

		Example:

		my_mi = load_mi('01_KFM_x5.525y2.7_40x40.mi')
		my_mi.buffers[2].data # data of the third buffer
		my_mi.dateAcquired # Date of when the data was recorded
		
	'''
	

	''' LIST OF AUTOMATIC CONVERSIONS '''
	
	ints = ['xPixels', 'yPixels']
	floats = ['xSensitivity', 'xNonlinearity', 'xNonlinearity',
			'xHisteresis', 'xRestPosition', 'ySensitivity',
			'yNonlinearity', 'yNonlinearity', 'yHisteresis',
			'yRestPosition', 'zSensitivity', 'zHysteresis',
			'zSensorSens', 'zSensorSwGain', 'preampSens',
			'bufferRange', 'xOffset', 'yOffset', 'xLength',
			'yLength',
			]


	''' PREPARE THE PARSE '''
	
	filesize = os.path.getsize(filename) # Needed for finding the exact location of the binary blob
	meta = dict(buffers=dict()) # All data is stored in meta, all image data goes to buffers
	latest_buffer = None # Buffers are listed sequentially in the header, this is the last buffer label we saw
	buffer_count = 0 # Counting the buffers, since theirs names are not unique

	''' PARSE LOOP '''
	
	with open(filename, encoding='latin-1') as file: # latin-1 seems to work, fingers crossed
		
		line_counter = 0 	#for knowing where the binary blob starts
							#We know the binary blob starts after the line where the data is defined
		for line in file:
			''' First 14 characters seem to be the name of the property. Value is the rest of the line '''
			key = line[:14].strip()
			val = line[14:].strip()

			''' Automatic conversions '''
			if key in ints:
				val = int(val)

			if key in floats:
				val = float(val)

			''' End of human readable stuff '''
			if "data" in key:
				meta[key] = val
				break

			''' bufferLabel marks a new image channel. It's followed by properties of the specific buffer or the data at the end '''
			if key == 'bufferLabel':
				latest_buffer = buffer_count
				buffer_count += 1
				meta['buffers'][latest_buffer] = dict()

			''' Properties that don't belong to a specific buffer '''
			if latest_buffer == None:
				meta[key] = val
			
			else:
				meta['buffers'][latest_buffer][key] = val


	''' BUFFER DATA EXTRACTION
	
		The image data ("buffers") are stored sequentially at
		the end of the file. Each buffer is a sequence of signed
		integers of 32bit (apparently 16 is also possible and
		indicated by the 'data' field).

		We use np.fromfile to load the bytes corresponding to the
		buffer from the file. For this we need to point at the right
		starting point of the file, where the buffer begins and then
		tell the function, how many integers to retrieve.

		We have xPixels*yPixels of each 32bit = 4byte. Therefore each
		buffer is xPixels*yPixels*4 bytes in size. If we found N
		buffers, the last N * xPixels*yPixels*4 bytes of the file are
		the buffers. Chunk it into N pieces to get the individual buffers.
	'''
	
	meta['buffer_count'] = buffer_count # How many buffers we saw and that we need to extract from the binary blob

	if not allow_16_bit:
		# print("not allowing 16 bit")
		assert meta['data'] == 'BINARY_32', "doesn't seem to be 32bit. Set allow_16_bit=True, but the functionality is UNTESTED!"
	
	for b in meta['buffers']:
	
		
		bytes_per_pixel = 4 if meta['data'] == 'BINARY_32' else 2 # int32 ~ _4_*8bit
		pixels = meta['xPixels'] * meta['yPixels']
		field_size_bytes = pixels * bytes_per_pixel # Size of a single buffer
		
		
		
		
		offset = find_binary_data_offset(filename) + b * field_size_bytes # Starting location of the buffer b
		
		
		
		data = np.fromfile(filename, dtype=np.int32, offset=offset, count=pixels) # count is num of ints to load, not bytes.
		# print("old shape shape", data.shape)
		data = data.reshape((meta['yPixels'], meta['xPixels']))
		#Changed the X-Y order to Y-X, the X-Y order works fine for NxN images, but not for N*M images

		# print("new shape",data.shape)
		''' Convert the range and units of the buffer data '''
		range_ = meta['buffers'][b]['bufferRange']
		zFactor = range_ / (2.**31) # 32-1 - First bit is for sign!
		conversions = dict(mV=1e-3, V=1, um=1e-6, deg=1)
		unit = meta['buffers'][b]['bufferUnit']
		zFactor *= conversions[unit]
		meta['buffers'][b]['data'] = data * zFactor


	
	''' PHYSICAL LOCATION OF THE IMAGE '''

	xmin = meta['xOffset']
	xmax = xmin + meta['xLength']
	ymin = meta['yOffset']
	ymax = ymin + meta['yLength']

	meta['xmin'] = xmin
	meta['xmax'] = xmax
	meta['ymin'] = ymin
	meta['ymax'] = ymax

	meta['extent'] = np.array([xmin, xmax, ymin, ymax])
	meta['extent_zero'] = np.array([0, xmax-xmin, 0, ymax-ymin])
	

	''' CONVERT THE DATASTRUCTURE TO A NAMESPACE FOR EASE OF USE '''
	
	buffs = meta['buffers']
	meta['buffers'] = list() # just using a simple list instead of numbered keys
	for b in buffs:
		meta['buffers'].append(SimpleNamespace(**buffs[b]))
		meta['buffers'][-1].properties = list(meta['buffers'][-1].__dict__.keys())

	ns = SimpleNamespace(**meta)
	ns.properties = list(ns.__dict__.keys())
	return ns

'''
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
'''

def __parse_grid_line(line, point_list):
	'''Structure of the line:
	ID  centerX CenterY countX countY stepsize ? ? ?
	0   1       2       3      4      5        6 7 8
	'''
	parts = line.split()
	centerX  = float(parts[1])
	centerY  = float(parts[2])
	countX   = int(parts[3])
	countY   = int(parts[4])
	stepsize = float(parts[5])

	startX = centerX - stepsize * (countX / 2. - 0.5)
	startY = centerY - stepsize * (countY / 2. - 0.5)

	for y in range(countY):
		for x in range(countX):
			point = SimpleNamespace(x=startX + x * stepsize,
									y=startY + y * stepsize,
									buffers=list()
								)
			point_list.append(point)
	
	
	
def __parse_point_line(line, point_list):
	parts = line.split()
	point = SimpleNamespace(x=float(parts[1]),
							y=float(parts[2]),
							buffers=list()
						)
	point_list.append(point)

def __parse_chunk_line(line):
	parts = line.split()
	chunk = SimpleNamespace(
		chunk_id = int(parts[0]),
		sample_count = int(parts[1]),
		time1 = parts[2],
		time2 = parts[3],
		startZ = float(parts[4]),
		stepZ = float(parts[5]),
		# part[6] = ??? unknown boolean parameter 
		direction = parts[7],
		point_id = int(parts[8]),
		curve_id = int(parts[9]),
	)

	return chunk


def _load_mi_spectroscopy(filename):

	filesize = os.path.getsize(filename) # Needed for finding the exact location of the binary blob
	meta = dict(points=list()) # All data is stored in meta, all measurement points are in the list points
	latest_buffer = None # Buffers are listed sequentially in the header, this is the last buffer label we saw
	buffer_count = -1 # Counting the buffers, since theirs names are not unique

	reached_point_definitions = False # Check if we're past the points / grids section of the file
	reached_buffer_chunks = False

	current_data_index = 0

	with open(filename, encoding='latin-1') as file: # latin-1 seems to work, fingers crossed
		
		for line in file:
			
			''' First 14 characters seem to be the name of the property. Value is the rest of the line '''
			key = line[:14].strip()
			val = line[14:].strip()

			if "bufferLabel" in key and not reached_point_definitions:
				reached_point_definitions = True
			elif "bufferLabel" in key and reached_point_definitions:
				reached_buffer_chunks = True
				buffer_count += 1

				for point in meta['points']:
					point.buffers.append(SimpleNamespace(chunks=list()))
				
			if "point" == key:
				__parse_point_line(val, meta["points"])

			if "grid" == key:
				__parse_grid_line(val, meta["points"])

			if "data" in key:
				meta[key] = val
				assert val == "BINARY", "Data type is not 'BINARY'. This has not been seen in testing, unsure if this will be an issue." 
				break

			if not reached_buffer_chunks:
				meta[key] = val
				
			else:
				
				if "chunk" in key:
					chunk = __parse_chunk_line(val)
					chunk.data_pointer = current_data_index
					current_data_index += chunk.sample_count
					meta["points"][chunk.point_id].buffers[buffer_count].chunks.append(chunk)
				
				else:
					for point in meta["points"]:
						setattr(point.buffers[buffer_count], key, val)


	number_of_measurements = current_data_index
	binary_blob_size_bytes = number_of_measurements * 4 # 4bytes per measurement (float32?)
	offset = filesize - binary_blob_size_bytes
	raw_data = np.fromfile(filename, dtype=np.float32, offset=offset)

	for point in meta["points"]:
		for buffer in point.buffers:

			''' Prepare approach curve collection '''
			chunk0 = buffer.chunks[0]
			assert chunk0.direction=="Approach", "First chunk is not an approach as expected"
			n_values = chunk0.sample_count
			startZ = chunk0.startZ
			stopZ = startZ + chunk0.stepZ * n_values
			z_approach = np.linspace(startZ, stopZ, n_values)
			buffer.approach = SimpleNamespace(z=z_approach, curves=list())

			''' Prepare retract curve collection '''
			chunk1 = buffer.chunks[1]
			assert chunk1.direction=="Retract", "Second chunk is not a retract as expected"
			n_values = chunk1.sample_count
			startZ = chunk1.startZ
			stopZ = startZ + chunk1.stepZ * n_values
			z_retract = np.linspace(startZ, stopZ, n_values)
			buffer.retract  = SimpleNamespace(z=z_retract, curves=list())

			for chunk in buffer.chunks:
				collection = buffer.approach if chunk.direction=="Approach" else buffer.retract
				start = chunk.data_pointer
				stop =  start + chunk.sample_count
				collection.curves.append(raw_data[start:stop])

			''' Flip the approach, so the data is aligned equally for retract and approach '''
			buffer.approach.curves = [np.flip(curve) for curve in buffer.approach.curves]
			buffer.approach.z = np.flip(buffer.approach.z)                
			
	return SimpleNamespace(**meta)


''' USER FUNCTION
------------------------------------------------------------------------------------------------------
	These functions are used by the user. They return the desired
	data.
'''
def load_mi(filename, allow_16_bit=False):

	''' Primary loading function of Agilante .mi files.
		This function just checks the first line of the .mi file and
		decides if it is a spectroscopy or an image file. It then
		calls the appropriate routine.
	'''

	with open(filename, encoding='latin-1') as file:
		
		header = file.readline()
		
		if "Spectroscopy" in header:
			return _load_mi_spectroscopy(filename)
			
		elif "Image" in header:
			return _load_mi_image(filename, allow_16_bit)
			
		else:
			raise Exception(f"Didn't find 'Spectroscopy' or 'Image' type in the first line of {filename}, I don't know how to handle this")
