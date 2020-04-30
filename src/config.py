import os
root_dir = os.path.dirname(__file__)
CSV_FILE_PATH = os.path.join(root_dir, '../data/youtube2TextCaptions/MSR_Video_Description_Corpus.csv')

NPZ_VIDEO_FEATURE_FILE = os.path.join(root_dir, '../data/videoFeatures')
PICKLE_FILE_PATH = os.path.join(root_dir,'../data/videoFeaturesPickle/all_video_frames.pkl')

GLOVE_200_DIM_FILE = os.path.join(root_dir,'../data/glove/glove.6B.200d.txt')

TRAINED_MODEL_HDF5_FILE = os.path.join(root_dir,'../data/output/trained.15vid.hdf5')
TRAINED_VIDEO_ID_NPY_FILE = os.path.join(root_dir,'../data/output/trained.videoid.npy')
TRAINED_MODEL_FOLDER = os.path.join(root_dir,'../data/output/')
TRAINING_VIDEOID_FILE = os.path.join(root_dir,'../data/output/training.videoid.text')
VALIDATION_VIDEOID_FILE = os.path.join(root_dir,'../data/output/validation.videoid.text')
TESTING_VIDEOID_FILE = os.path.join(root_dir,'../data/output/testing.videoid.text')
LOSS_ACCURACY_FILE = os.path.join(root_dir,'../data/output/lossAndAccuracy.text')