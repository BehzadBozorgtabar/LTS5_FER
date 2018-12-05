import os
import numpy as np
import pandas as pd

#from extraction import *
from extraction import get_conv_1_1_weights, create_tcnn_bottom, extract_vgg_tcnn
#from sift import extract_all_sift_features


smb_data_path = '/Volumes/External/ProjectData/ADAS&ME/ADAS&ME_data/smb'
frames_path = '/Volumes/External/ProjectData/ADAS&ME/ADAS&ME_data/frames'
annotations_path = '/Volumes/External/ProjectData/ADAS&ME/ADAS&ME_data/annotated'
vggCustom_weights_path = '/Users/tgyal/Documents/EPFL/MA3/Project/fer-project/models/customised_VGG.h5'

files_list = [  ('TS4_DRIVE', '20180824_150225'),
                 ('TS4_DRIVE', '20180824_150302'),
                 ('TS4_DRIVE', '20180824_150401'),
                 ('TS4_DRIVE', '20180824_150443'),
                 ('TS4_DRIVE', '20180824_150543'),
                 ('TS4_DRIVE', '20180829_135202'),
                ('TS6_DRIVE', '20180828_162845'),
                 ('TS6_DRIVE', '20180828_163043'),
                 ('TS6_DRIVE', '20180828_163156'),
                 ('TS6_DRIVE', '20180829_083936'),
                 ('TS6_DRIVE', '20180829_084019'),
                 ('TS6_DRIVE', '20180829_084054'),
                 ('TS6_DRIVE', '20180829_084134'),
                 ('TS6_DRIVE', '20180829_091659'),
                ('TS7_DRIVE', '20180828_142136'),
                 ('TS7_DRIVE', '20180828_142244'),
                 ('TS7_DRIVE', '20180828_142331'),
                 ('TS7_DRIVE', '20180828_150143'),
                 ('TS7_DRIVE', '20180828_150234'),
                 ('TS7_DRIVE', '20180828_150322'),
                 ('TS7_DRIVE', '20180828_150425'),
                ('TS8_DRIVE',  '20180827_174358'),
                ('TS8_DRIVE',  '20180827_174508'),
                ('TS8_DRIVE',  '20180827_174552'),
                ('TS8_DRIVE',  '20180827_174649'),
                ('TS8_DRIVE',  '20180827_174811'),
                ('TS9_DRIVE',  '20180827_165431'),
                ('TS9_DRIVE',  '20180827_165525'),
                ('TS9_DRIVE',  '20180827_165631'),
                ('TS10_DRIVE', '20180827_164836'),
                ('TS10_DRIVE', '20180827_164916'),
                ('TS10_DRIVE', '20180827_165008'),
                ('TS10_DRIVE', '20180828_082231'),
                ('TS10_DRIVE', '20180828_082326'),
                ('TS10_DRIVE', '20180828_085245'),
                ('TS10_DRIVE', '20180828_114716'),
                ('TS10_DRIVE', '20180828_114905'),
                ('TS10_DRIVE', '20180828_143226'),
                ('TS10_DRIVE', '20180828_143304'),
                ('TS11_DRIVE', '20180827_115441'),
                ('TS11_DRIVE', '20180827_115509'),
                ('TS11_DRIVE', '20180827_115620'),
                ('TS11_DRIVE', '20180827_115730'),
                ('TS11_DRIVE', '20180827_115840'),
                ('TS11_DRIVE', '20180827_120035')]

def get_files_from_subject(subject):
    return [f for (s,f) in files_list if s is subject]

# Extract frames from smb files

# smb_files_list =  ['20180828_142136',
#                  '20180828_142244',
#                  '20180828_142331',
#                  '20180828_150143',
#                  '20180828_150234',
#                  '20180828_150322',
#                  '20180828_150425']
# subject = 'TS7_DRIVE'
# subj_smb_path = smb_data_path+'/'+subject
# dest_folder = frames_path+'/'+subject

# extract_smb_frames(subj_smb_path, smb_files_list, dest_folder=dest_folder, align_resize=True, nb_frames=None)



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



# Extract VGG-TCNN features for the late-fusion model

conv1_1_weigths = get_conv_1_1_weights(vggCustom_weights_path)
tcnn_bottom = create_tcnn_bottom(vggCustom_weights_path, conv1_1_weigths)

frames_path = '/Volumes/External/ProjectData/ADAS&ME/ADAS&ME_data/frames'
annotations_path = '/Volumes/External/ProjectData/ADAS&ME/ADAS&ME_data/annotated'
dest_path = '/Volumes/External/ProjectData/ADAS&ME/data'

subjects = ['TS7_DRIVE', 'TS8_DRIVE', 'TS9_DRIVE', 'TS10_DRIVE', 'TS11_DRIVE']

for subject in subjects:
	print(subject)
	smb_files_list =  get_files_from_subject(subject)

	subj_frames_path = frames_path+'/'+subject
	subj_annotations_path = annotations_path+'/'+subject
	dest_folder = dest_path+'/'+subject

	extract_vgg_tcnn(tcnn_bottom.predict, 5, subj_frames_path, subj_annotations_path, smb_files_list, dest_folder)
