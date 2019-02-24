import os
import numpy as np
import pickle

from extract_smb_frames import extract_smb_frames
from const import *
from extraction import readCSV, create_directory


for folder in os.listdir(csv_smb_files_path):
	path = csv_smb_files_path + folder + '/'
	pkl_directory = real_data_path + folder
	
	create_directory(pkl_directory)
			
	allFiles = os.listdir(path)
	for smb in allFiles:
		if smb.endswith("smb"):
			smb_path = path + smb
			name = smb[-19 : -4]
			csvfiles = [path + f for f in allFiles if ((name in f) and f.endswith(".csv"))]
				
			for csvfile in csvfiles:
				annotations = readCSV(csvfile)
				part = csvfile.split('/')[-1]
				part = part[25 : -4]

				pkl_path = pkl_directory + "/" + name
				part = '1' if len(part) == 0 else part
				
				pkl_path = pkl_path + "_" + part

				annotations = np.array(annotations)
				cleanAnnots = []
				cleanImages = []

				print("Extracting frames of {}".format(csvfile)) 
				for i, annots in enumerate(annotations):
					print("Extracting frame {}".format(i+1), end = '\r')
					cameraIndex = annots['CameraIndex']
					frameNumber = int(annots['FrameNumber'])
					severity = int(annots['Severity'])
					if severity != -999:
						cleanImages.append(extract_smb_frames(smb_path, start_frame_index = i + (int(part) - 1) * MAX_SEGMENT_SIZE, nb_frames = 1, face_rect_scale = 0.8, verbose=False)[0])
						cleanAnnots.append({'CameraIndex': cameraIndex, 'FrameNumber' : frameNumber, 'Severity' : severity})


				cleanAnnots = np.array(cleanAnnots)
				cleanImages = np.array(cleanImages)
				with open(pkl_path + ".pkl", "wb") as file:
					pickle.dump((cleanImages, cleanAnnots), file) 

