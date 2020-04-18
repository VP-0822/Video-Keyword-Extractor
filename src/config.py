import os
root_dir = os.path.dirname(__file__)
CSV_FILE_PATH = os.path.join(root_dir, '..\\data\\youtube2TextCaptions\\MSR_Video_Description_Corpus.csv')

NPZ_VIDEO_FEATURE_FILE = os.path.join(root_dir, '..\\data\\videoFeatures')
PICKLE_FILE_PATH = os.path.join(root_dir,'..\\data\\videoFeaturesPickle\\all_video_frames.pkl')

GLOVE_200_DIM_FILE = os.path.join(root_dir,'..\\data\\glove\\glove.6B.200d.txt')

TRAINED_MODEL_HDF5_FILE = os.path.join(root_dir,'..\\data\\trained.15vid.hdf5')
TRAINED_VIDEO_ID_NPY_FILE = os.path.join(root_dir,'..\\data\\trained.videoid.npy')