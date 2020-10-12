import os
root_dir = os.path.dirname(__file__)
# Config parameters
USE_TPU = False
CURRENT_EXPERIMENT_NAME = 'Ninth run'
COMMENT = 'Video mixed encoder'
USE_LINEAR_EMBEDDER = False # Try using False
USE_CATEGORIES = False
USE_SUBTITLES = True
BATCH_SIZE = 28
VALIDATION_BATCH_SIZE = 56
MINIMUM_WORD_OCCURANCE_FREQUENCY = 1
AUDIO_FEATURE_DIMENSION = 128
VIDEO_FEATURE_DIMENSION = 1024
AUDIO_MODEL_DIMENSION = 128
VIDEO_MODEL_DIMENSION = 1024
SUBTITLE_MODEL_DIMENSION = 512
AUDIO_FEEDFORWARD_UNITS = 2048
VIDEO_FEEDFORWARD_UNITS = 2048
SUBTITLE_FEEDFORWARD_UNITS = 2048
AUDIO_ENCODER_DECODER_LAYERS = 1
VIDEO_ENCODER_DECODER_LAYERS = 1
SUBTITLE_ENCODER_DECODER_LAYERS = 1
DROPOUT_PERCENTAGE = 0.1
NUMBER_OF_ATTENTION_HEADS = 4
LABEL_SMOOTHING_FACTOR = 0.7 #TODO: Play with smoothing
ADAM_BETA_1 = 0.9
ADAM_BEAT_2 = 0.98
ADAM_EPS = 1e-8
LEARNING_RATE = 1e-5
STARTING_EPOCH = 0
TOTAL_EPOCHS = 50
EARLY_STOP_EPOCH_NUMBERS = 51
EPOCH_NUMBER_TO_START_EVALUATION = 50 #TODO: Set appropriate
VIDEOS_TO_MONITOR = ['v_GGSY1Qvo990', 'v_bXdq2zI1Ms0', 'v_aLv03Fznf5A']
tIoUs_FOR_EVALUATION_METRIC = [0.5] # [0.3, 0.5, 0.7, 0.9]
CONTINUE_TRAINING = False
SAVE_MODEL_ON_LAST_EPOCH = True
SAVE_INTERMEDIATE = True
SAVE_INTERMEDIATE_AT_EVERY_NTH_EPOCH = 5
USE_DEFAULT_CAPTION_LENGTH = 50

# file paths
LOG_PATH = os.path.join(root_dir,'./log')
CONTINUE_TRAINING_MODEL_FILE_PATH = os.path.join(root_dir,'')
EXPERIMENT_CHECKPOINT_FOLDER = os.path.join(LOG_PATH, CURRENT_EXPERIMENT_NAME)
VIDEO_HDF5_FILE_PATH = './../data/ActivityNet-available/sub_activitynet_v1-3.i3d_25fps_stack24step24_2stream.hdf5'# os.path.join(root_dir,'')
AUDIO_HDF5_FILE_PATH = './../data/ActivityNet-available/sub_activitynet_v1-3.vggish.hdf5'# os.path.join(root_dir,'')
TRAIN_META_FILE_PATH = os.path.join(root_dir,'./../data/train_meta.csv')
VALIDATION_1_META_FILE_PATH = os.path.join(root_dir,'./../data/val_1_meta.csv')
VALIDATION_2_META_FILE_PATH = os.path.join(root_dir,'./../data/val_2_meta.csv')
VALIDATION_1_REFERENCE_JSON = './../data/val_1.json'
VALIDATION_2_REFERENCE_JSON = './../data/val_2.json'
