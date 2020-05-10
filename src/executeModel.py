import os
import numpy as np
from tensorflow.keras.optimizers import Adam, RMSprop
import config
import fileUtils
from datasetPreprocessor import DatasetPreprocessor
from basicLSTMModel import BasicLSTMModel
from bidirectionalLSTMModel import BidirectionalLSTMModel
from bidirectionalGRUModel import BidirectionalGRUModel

# Load already defined video Ids
sample_video_ids = None
if os.path.exists(config.TRAINED_VIDEO_ID_NPY_FILE):
    sample_video_ids = np.load(config.TRAINED_VIDEO_ID_NPY_FILE)
    print(str(len(sample_video_ids)) + ' video_ids loaded')

# Load preprocessed data
NUMBER_OF_SAMPLES = 1969
TRAIN_TEST_SPLIT = 0.15
NUMBER_OF_VALIDATION_SAMPLES = 50
EMBEDDING_DIM = 200
CAPTIONS_PER_VIDEO = 4
SHUFFLE_SAMPLES = False
datasetPreprocessor = DatasetPreprocessor(NUMBER_OF_SAMPLES, TRAIN_TEST_SPLIT, NUMBER_OF_VALIDATION_SAMPLES, sample_video_ids)
# pickleFileList = [config.PICKLE_FILE_PATH]
pickleFileList = [config.PICKLE_FILE_PART_1_PATH, config.PICKLE_FILE_PART_2_PATH, config.PICKLE_FILE_PART_3_PATH]
datasetPreprocessor.loadVideoFeatureSamples(pickleFileList)
datasetPreprocessor.splitIntoTrainTestAndValidation(config.TRAINING_VIDEOID_FILE, config.TESTING_VIDEOID_FILE, config.VALIDATION_VIDEOID_FILE)
datasetPreprocessor.attachInputCaptionsToVideos(config.CSV_FILE_PATH, config.GLOVE_200_DIM_FILE, EMBEDDING_DIM, CAPTIONS_PER_VIDEO)
datasetPreprocessor.expandTrainAndValidationSet(config.TRAINING_ORDER_VIDEO_ID_FILE,SHUFFLE_SAMPLES)

trainSamples = datasetPreprocessor.getTrainSamples()
validationSamples= datasetPreprocessor.getValidationSamples()
testSamples = datasetPreprocessor.getTestSamples()
cp = datasetPreprocessor.getCaptionPreprocessorInstance()
vocabEmbedding = datasetPreprocessor.getVocabWordEmbeddings()
# Save all videos used from dataset
all_video_ids_np = np.asarray(datasetPreprocessor.getAllVideoIds())
np.save(config.TRAINED_VIDEO_ID_NPY_FILE, all_video_ids_np)

# Create Model
ONLY_PREDICT = False
CONTINUE_TRAINING = False
VIDEO_INPUT_SHAPE = (None,2048)
NO_EPOCHS = 1
# For Single caption per video
# STEPS_FOR_TRAIN_SAMPLES = 48
# TRAINING_BATCH_SIZE = 34
# STEPS_FOR_VALIDATION_SAMPLES = 2
# VALIDATION_BATCH_SIZE = 25

# For multiple captions per video (4 captions)
STEPS_FOR_TRAIN_SAMPLES = 96
TRAINING_BATCH_SIZE = 68
STEPS_FOR_VALIDATION_SAMPLES = 8
VALIDATION_BATCH_SIZE = 25

OPTIMIZER = Adam(learning_rate=0.00005, beta_1=0.9, beta_2=0.99, epsilon=1e-07, amsgrad=True, name='Adam')

if ONLY_PREDICT is False:
    model = BidirectionalGRUModel(cp, EMBEDDING_DIM, VIDEO_INPUT_SHAPE, vocabEmbedding)
    model.buildModel()
    model.compileModel(OPTIMIZER)
    # Load pretrained Model
    if CONTINUE_TRAINING is True:
        print('continuing previous training')
        model.loadTrainedModel(config.TRAINED_MODEL_HDF5_FILE)

    print('Starting training......')
    trainingHistory = model.trainModel(trainSamples, validationSamples, NO_EPOCHS, TRAINING_BATCH_SIZE, STEPS_FOR_TRAIN_SAMPLES, VALIDATION_BATCH_SIZE, STEPS_FOR_VALIDATION_SAMPLES, config.TRAINED_MODEL_FOLDER, config.TRAINED_MODEL_HDF5_FILE)

    loss_train = trainingHistory.history['loss']
    loss_val = trainingHistory.history['val_loss']
    acc_train = trainingHistory.history['accuracy']
    acc_val = trainingHistory.history['val_accuracy']
    fileUtils.writeLossAndAccuracyToFile(config.LOSS_ACCURACY_FILE, NO_EPOCHS, loss_train, loss_val, acc_train, acc_val)
else:
    model = BidirectionalGRUModel(cp, EMBEDDING_DIM, VIDEO_INPUT_SHAPE)
    model.buildModel()
    model.compileModel(OPTIMIZER)
    print('Trained model loaded.')
    model.loadTrainedModel(config.TRAINED_MODEL_HDF5_FILE)

predictedSamples = model.predictTestSamples(testSamples)
fileUtils.writeArrayToFile(config.TEST_SET_PREDICTION_CSV_FILE, predictedSamples)
print('Prediction done!')
