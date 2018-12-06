import numpy as np

img_size = 224 # resize images to 224*224 px

emotions = ['Neutral', 'Positive','Negative','Anxiety']
nb_emotions = len(emotions)

# Define the masks corresponding to each cluster of landmarks
lm_range = np.array(range(51))
eyebrows_mask = (lm_range < 10) # 10 landmarks
nose_mask = np.logical_and(10 <= lm_range, lm_range < 19) # 9 landmarks
eyes_mask = np.logical_and(19 <= lm_range, lm_range < 31)# 12 landmarks
outside_lip_mask = np.logical_and(31 <= lm_range, lm_range < 43) # 12 landmarks
inside_lip_mask = (43 <= lm_range) # 8 landmarks

FILES_LIST = [  ('TS4_DRIVE', '20180824_150225'),
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