import os
import numpy as np
import pandas as pd

from extraction import *
#from extraction import get_conv_1_1_weights, create_tcnn_bottom, extract_vgg_tcnn
from sift import extract_all_sift_features
from const import *


#smb_data_path = '/Volumes/External/ProjectData/ADAS&ME/ADAS&ME_data/smb'
# smb_data_path = '/Volumes/Maxtor/data'
# frames_path = '/Volumes/Ternalex/ProjectData/ADAS&ME/ADAS&ME_data/frames'
# annotations_path = '/Volumes/Ternalex/ProjectData/ADAS&ME/ADAS&ME_data/annotated'
vggCustom_weights_path = '/Users/tgyal/Documents/EPFL/MA3/Project/fer-project/models/customised_VGG.h5'

frames_data_path = '/Volumes/Ternalex/ProjectData/ADAS&ME/ADAS&ME_data/Real_data'


# Extract frames from smb files

subjects = ['TS7_DRIVE', 'TS9_DRIVE', 'UC_B1', 'UC_B2', 'UC_B4', 'UC_B5','VP03', 'VP17', 'VP18']
smb_files = [['20180829_114712'],
             ['20180828_143018'],
             ['20180814_101543'],#UC_B1
             ['20181121_162205'],
             ['20181121_163847'],
             ['20181121_165322'],
             ['20180822_162114'],
             ['20180823_095005'],
             ['20180823_095958']]

# for i, subject in enumerate(subjects):
#     print(subject)
#     smb_files_list =  smb_files[i]
#     subj_smb_path = smb_data_path+'/'+subject
#     dest_folder = frames_path+'/'+subject

#     extract_smb_frames(subj_smb_path, smb_files_list, dest_folder=dest_folder, align_resize=True, nb_frames=None, stop_at_frame_nb=[])


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

# Extract facial landmarks
print('Facial Landmarks')
dest_folder = '/Volumes/Ternalex/ProjectData/ADAS&ME/data'

for i, subject in enumerate(subjects):
    print(subject)

    subj_data_path = frames_data_path+'/'+subject
    subj_dest_folder = dest_folder+'/'+subject
    extract_facial_landmarks(subj_data_path, subj_dest_folder)


# Extract SIFT features
print('SIFT Features')
dest_folder = '/Volumes/Ternalex/ProjectData/ADAS&ME/data'

for i, subject in enumerate(subjects):
    print(subject)

    subj_data_path = frames_data_path+'/'+subject
    subj_dest_folder = dest_folder+'/'+subject
    extract_sift(extract_all_sift_features, subj_data_path, subj_dest_folder)


# Extract VGG-TCNN features for the late-fusion model
print('VGG-TCNN Features')
dest_folder = '/Volumes/Ternalex/ProjectData/ADAS&ME/data'

conv1_1_weigths = get_conv_1_1_weights(vggCustom_weights_path)
tcnn_bottom = create_tcnn_bottom(vggCustom_weights_path, conv1_1_weigths)

for i, subject in enumerate(subjects):
    print(subject)

    subj_data_path = frames_data_path+'/'+subject
    subj_dest_folder = dest_folder+'/'+subject
    extract_vgg_tcnn(tcnn_bottom.predict, 5, subj_data_path, subj_dest_folder)
