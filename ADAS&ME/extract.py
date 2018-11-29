import os
import numpy as np
import pandas as pd

from extraction import *
from sift import extract_all_sift_features


smb_data_path = '/Volumes/External/ProjectData/ADAS&ME/ADAS&ME_data/smb'
frames_path = '/Volumes/External/ProjectData/ADAS&ME/ADAS&ME_data/frames'
annotations_path = '/Volumes/External/ProjectData/ADAS&ME/ADAS&ME_data/annotated'


# Extract frames from smb files

smb_files_list =  ['20180828_142136',
                 '20180828_142244',
                 '20180828_142331',
                 '20180828_150143',
                 '20180828_150234',
                 '20180828_150322',
                 '20180828_150425']
subject = 'TS7_DRIVE'
subj_smb_path = smb_data_path+'/'+subject
dest_folder = frames_path+'/'+subject

extract_smb_frames(subj_smb_path, smb_files_list, dest_folder=dest_folder, align_resize=True, nb_frames=None)



# Extract SIFT and VGG features
# smb_files_list =  ['20180827_115840', '20180827_120035']
# subject = 'TS11_DRIVE'
# dest_path = '/Volumes/External/ProjectData/ADAS&ME/data'

# vgg_weights_path = '/Volumes/External/ProjectData/models/customised_VGG.h5'
# vgg_fc6 = create_vgg_extractor(vgg_weights_path, output_layer='fc6')

# subj_frames_path = frames_path+'/'+subject
# subj_annotations_path = annotations_path+'/'+subject
# dest_folder = dest_path+'/'+subject
# extract_sift_vgg(extract_all_sift_features, vgg_fc6.predict, subj_frames_path, subj_annotations_path, smb_files_list, dest_folder)