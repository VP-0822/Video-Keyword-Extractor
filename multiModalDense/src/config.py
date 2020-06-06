import os
root_dir = os.path.dirname(__file__)
# Config parameters
CURRENT_EXPERIMENT_NAME = 'firstrun'
COMMENT = 'Executing for the first time'
USE_CATEGORIES = False
USE_SUBTITLES = True
BATCH_SIZE = 28
MINIMUM_WORD_OCCURANCE_FREQUENCY = 2

# file paths
LOG_PATH = os.path.join(root_dir,'/log')
EXPERIMENT_CHECKPOINT_FOLDER = os.path.join(LOG_PATH, CURRENT_EXPERIMENT_NAME)
VIDEO_HDF5_FILE_PATH = os.path.join(root_dir,'')
AUDIO_HDF5_FILE_PATH = os.path.join(root_dir,'')
TRAIN_META_FILE_PATH = os.path.join(root_dir,'')
VALIDATION_1_META_FILE_PATH = os.path.join(root_dir,'')
VALIDATION_2_META_FILE_PATH = os.path.join(root_dir,'')
