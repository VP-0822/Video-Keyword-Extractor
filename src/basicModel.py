from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from keras import callbacks
from keras import optimizers
import os
import numpy as np
import math
import videoFrameFeatures as vff
import youtube2TextCaptions as y2tc
import config
import util
import captionPreprocess as cp
import gloveEmbeddings as ge
import matplotlib.pyplot as plt
import random
import tqdm
from models import basicLSTMModel as lstmModel
from models import bidirectionalLSTMModel as bidirectionLstmModel
from models import bidirectionalGRUModel as bidirectionalGruModel

def extractVideoInputs(video_frames, no_samples, train_test_split, no_validation_samples):
    train_test_split = math.floor((no_samples - no_validation_samples) * train_test_split)
    train_samples = dict()
    val_samples = dict()
    test_samples = dict()

    if not os.path.exists(config.TRAINING_VIDEOID_FILE):
        test_size_counter = 0
        train_size_counter = 0
        for key in video_frames.keys():
            if (test_size_counter + train_size_counter) == (no_samples - no_validation_samples):
                val_samples[key] = [video_frames[key]]
                continue
            if(test_size_counter < train_test_split):
                test_samples[key] = [video_frames[key]]
                test_size_counter += 1
                continue
            train_samples[key] = [video_frames[key]]
            train_size_counter += 1
        util.writeArrayToFile(config.TRAINING_VIDEOID_FILE, list(train_samples.keys()))
        util.writeArrayToFile(config.VALIDATION_VIDEOID_FILE, list(val_samples.keys()))
        util.writeArrayToFile(config.TESTING_VIDEOID_FILE, list(test_samples.keys()))
    else:
        training_video_id_list = util.readArrayFromFile(config.TRAINING_VIDEOID_FILE)
        validation_video_id_list = util.readArrayFromFile(config.VALIDATION_VIDEOID_FILE)
        testing_video_id_list = util.readArrayFromFile(config.TESTING_VIDEOID_FILE)
        for key in video_frames.keys():
            if key in training_video_id_list:
                train_samples[key] = [video_frames[key]]
            if key in validation_video_id_list:
                val_samples[key] = [video_frames[key]]
            if key in testing_video_id_list:
                test_samples[key] = [video_frames[key]]
    
    print('Number of Training videos: ' + str(len(train_samples)))
    print('Number of Validation videos: ' + str(len(val_samples)))
    print('Number of Testing videos: ' + str(len(test_samples)))
    return train_samples, val_samples, test_samples

def prepareDataset(no_samples=200, train_test_split=0.15, no_validation_samples=50, video_ids=None):
    video_frames = vff.loadVideoFrameFeatures(config.PICKLE_FILE_PATH, no_samples, video_ids)
    print('video frame features loaded')
    
    train_samples, val_samples, test_samples = extractVideoInputs(video_frames, no_samples, train_test_split, no_validation_samples)
    
    all_video_ids = list(test_samples.keys())
    all_video_ids.extend(list(train_samples.keys()))
    all_video_ids.extend(list(val_samples.keys()))
    
    final_video_ids = list(train_samples.keys())
    final_video_ids.extend(list(val_samples.keys()))

    video_captions = y2tc.filterCaptionsForSamples(config.CSV_FILE_PATH, final_video_ids, load_single_caption=True)
    print('video captions loaded')
    caption_preprocessor = cp.CaptionPreprocessor(video_captions, word_freq_threshold=2)
    print('Final word count: ' + str(caption_preprocessor.getVocabSize()))
    #print(caption_preprocessor.getCaptionsVocabList())
    print('video captions preprocessed')
    glove_embedding = ge.GloveEmbedding(config.GLOVE_200_DIM_FILE, 200)
    print('glove embedding loaded')
    vocab_word_embeddings = glove_embedding.getEmbeddingVectorFor(caption_preprocessor.getCaptionsVocabList(), caption_preprocessor.getVocabSize())
    preprocessed_video_captions = caption_preprocessor.caption_inputs

    for key in train_samples.keys():
        train_samples[key].append(preprocessed_video_captions[key])
    for key in val_samples.keys():
        val_samples[key].append(preprocessed_video_captions[key])

    test_video_captions = y2tc.filterCaptionsForSamples(config.CSV_FILE_PATH, list(test_samples.keys()))
    for key, value in test_video_captions.items():
        test_samples[key].append(value)
    
    final_train_samples, final_val_samples = getExpandedTrainAndValidationSet(train_samples, val_samples)

    return final_train_samples, final_val_samples, test_samples, all_video_ids, caption_preprocessor, vocab_word_embeddings

# This method returns expanded train and validation set across different videos
def getExpandedTrainAndValidationSet(train_samples, validation_samples):
    final_train_samples = dict()
    for key, value in train_samples.items():
        # Value is list with index 0 as frame_inputs and index 1 as all_video_captions
        for index, caption in enumerate(value[1]):
            sample_key = key + '^' + str(index + 1)
            final_train_samples[sample_key] = [value[0], caption]
    train_keys = list(final_train_samples.keys())
    random.shuffle(train_keys)
    shuffled_train_samples = dict()
    for key in train_keys:
        shuffled_train_samples[key] = final_train_samples[key]
    
    final_val_samples = dict()
    for key, value in validation_samples.items():
        # Value is list with index 0 as frame_inputs and index 1 as all_video_captions
        for index, caption in enumerate(value[1]):
            sample_key = key + '^' + str(index + 1)
            final_val_samples[sample_key] = [value[0], caption]
    val_keys = list(final_val_samples.keys())
    random.shuffle(val_keys)
    shuffled_val_samples = dict()
    for key in val_keys:
        shuffled_val_samples[key] = final_val_samples[key]
    
    if os.path.exists(config.TRAINING_ORDER_VIDEO_ID_FILE):
        temp_train_samples = dict()
        temp_val_samples = dict()
        print('Following the previous training order')
        trainedVideoSampleIds = util.readArrayFromFile(config.TRAINING_ORDER_VIDEO_ID_FILE)
        for videoId in trainedVideoSampleIds:
            if videoId in shuffled_train_samples.keys():
                temp_train_samples[videoId] = shuffled_train_samples[videoId]
                continue
            if videoId in shuffled_val_samples.keys():
                temp_val_samples[videoId] = shuffled_val_samples[videoId]
                continue
        shuffled_train_samples = temp_train_samples
        shuffled_val_samples = temp_val_samples
    else:
        trainingSampleOrder = list()
        trainingSampleOrder.extend(list(shuffled_train_samples.keys()))
        trainingSampleOrder.extend(list(shuffled_val_samples.keys()))
        util.writeArrayToFile(config.TRAINING_ORDER_VIDEO_ID_FILE, trainingSampleOrder)
        print('Saved training sample order for continuing training')

    print('Expanded train samples: ' + str(len(shuffled_train_samples)))
    print('Expanded validation samples: ' + str(len(shuffled_val_samples)))
    return shuffled_train_samples, shuffled_val_samples

def getModel(final_caption_length, embedding_dim, video_frame_shape, total_vocab_size, embedding_weights=None):
    
    #model = lstmModel.getModel(final_caption_length, embedding_dim, video_frame_shape, total_vocab_size, embedding_weights)
    model = bidirectionLstmModel.getModel(final_caption_length, embedding_dim, video_frame_shape, total_vocab_size, embedding_weights, dropOutAtFinal=0.3)
    #model = bidirectionalGruModel.getModel(final_caption_length, embedding_dim, video_frame_shape, total_vocab_size, embedding_weights, dropOutAtFinal=0.3)

    optimizer = Adam(learning_rate=0.00005, beta_1=0.9, beta_2=0.99, epsilon=1e-07, amsgrad=True, name='Adam')
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    print('Model created successfully')
    return model

def dataGenerator(training_set, wordtoidx, max_caption_length, num_samples_per_batch, vocab_size, dataset_name='Training'):
    # x1 - Training data for videos
    # x2 - The caption that goes with each photo
    # y - The predicted rest of the caption
    x1, x2, y = [], [], []
    current_batch_item_count=0
    #print('Dataset name: ' + dataset_name)
    while True:
        for key, values in training_set.items():
            current_batch_item_count+=1
            frame_features = values[0]
            single_video_captions = values[1]
            in_caption_item = f'{cp.START_KEYWORD} {single_video_captions}'
            in_seq = [wordtoidx[word] for word in in_caption_item.split(' ') if word in wordtoidx]
            for i in range(max_caption_length-len(in_seq)):
                in_seq.append(wordtoidx[cp.NONE_KEYWORD])

            out_caption_item = f'{single_video_captions} {cp.STOP_KEYWORD}'
            out_caption_seq = [wordtoidx[word] for word in out_caption_item.split(' ') if word in wordtoidx]
            for i in range(max_caption_length-len(out_caption_seq)):
                out_caption_seq.append(wordtoidx[cp.NONE_KEYWORD])
            out_seq = list()
            for count in range(max_caption_length):
                out_seq.append(to_categorical([out_caption_seq[count]], num_classes=vocab_size)[0])
            x1.append(in_seq)
            x2.append(frame_features)
            y.append(out_seq)
            if current_batch_item_count==num_samples_per_batch:
                # print('##########################')
                # print('Dataset name: ' + dataset_name)
                # print("=== x1 ***********************************====") 
                # print(', '.join(map(str,np.array(x1).shape)) +' && '+ ', '.join(map(str,np.array(x2).shape)) +' && '+ ', '.join(map(str,np.array(y).shape)))
                # print(np.array(x1[0]).shape)
                # print("=== x2 ***********************************====")
                # print(np.array(x2).shape)
                # print(np.array(x2[0]).shape)
                yield ([np.array(x1), np.array(x2)], np.array(y))
                current_batch_item_count=1
                x1, x2, y = [], [], []

def predictFromModel(model, video_sample, wordtoidx, idxtoword, max_caption_length):
    video_frame_input = video_sample[0]
    original_video_caption_input = video_sample[1][0]
    dummy_caption = [cp.START_KEYWORD]
    dummy_caption.extend([cp.STOP_KEYWORD] * (max_caption_length - 1))
    #dummy_caption = f'{cp.START_KEYWORD} {original_video_caption_input}'
    video_dummy_caption = [wordtoidx[word] for word in dummy_caption if word in wordtoidx]
    # for i in range(max_caption_length-len(video_dummy_caption)):
    #     video_dummy_caption.append(wordtoidx[cp.NONE_KEYWORD])
    #print(video_dummy_caption)
    input_sequence = pad_sequences([video_dummy_caption], maxlen=max_caption_length)
    #print(list(input_sequence))
    #print(np.argmax(captionoutput[0][3]))
    #print(np.argmax(captionoutput[0][7]))
    input_seq_list = list(input_sequence)
    output_caption = []
    caption_length_counter = 0
    while caption_length_counter < max_caption_length:
        captionoutput = model.predict([np.array(input_seq_list), np.array([video_frame_input])])
        #print('Shape of predict model: ' + str(captionoutput.shape))
        yhat = np.argmax(captionoutput[0][caption_length_counter])
        word = idxtoword[yhat]
        if word == cp.STOP_KEYWORD:
            break
        output_caption.append(word)
        if caption_length_counter + 1 != max_caption_length:
            input_seq_list[0][caption_length_counter+1] = wordtoidx[word]
        caption_length_counter += 1
        #for i, newOneHotWord in enumerate()
    output_caption_text = ' '.join(output_caption)
    return original_video_caption_input, output_caption_text

def predictTestSamples(model, testSampleSet, wordtoidx, idxtoword, max_caption_length):
    all_prediction_statements = []
    all_prediction_statements.append('Video Id,Original caption,Predicted caption')
    print('Predicting test samples...')
    for key, values in tqdm.tqdm(testSampleSet.items()):
        video_id = key
        original_video_caption_input, output_caption_text = predictFromModel(model, values, wordtoidx, idxtoword, max_caption_length)
        # print('Predicting for video: ' + video_id)
        csv_line = video_id
        # print('[Original caption]:')
        # print(original_video_caption_input)
        csv_line += ',' + original_video_caption_input
        # print('[Predicted caption]:')
        # print(output_caption_text)
        csv_line += ',' + output_caption_text
        # print('###########')
        all_prediction_statements.append(csv_line)
    util.writeArrayToFile(config.TEST_SET_PREDICTION_CSV_FILE, all_prediction_statements)

class BasicModelCallback(callbacks.Callback):
    def __init__(self, final_model, folderName):
        self.final_model = final_model
        self.folderName = folderName
        pass
    def on_epoch_end(self, epoch, logs={}):
        print("Epoch %d End " % epoch)
        loss = logs['loss']
        # acc  = logs['acc']
        #valloss = logs['val_loss']
        #valacc  = logs['val_acc']
        print('Epoch Training loss: ' + str(loss))
        # print('Epoch Training Accuracy: ' + str(acc))
        print('=========================================')
        if epoch % 20 is 0 and epoch is not 0:
            print('writing to model weights file')
            self.final_model.save_weights(self.folderName + 'trainedModel_' + str(epoch) + '.hdf5')
    
    def on_batch_end(self, batch, logs={}):
        print("Batch %d ends" % batch)
        loss = logs['loss']
        # acc  = logs['acc']
        print('Batch Training loss: ' + str(loss))
        # print('Batch Training Accuracy: ' + str(acc))

if __name__ == "__main__":
    video_ids = None
    CONTINUE_TRAINING = False
    NO_EPOCHS = 100
    # For Single caption per video
    STEPS_FOR_TRAIN_SAMPLES = 48
    STEPS_FOR_VALIDATION_SAMPLES = 4
    # # For multiple captions per video (4 captions)
    # STEPS_FOR_TRAIN_SAMPLES = 96
    # STEPS_FOR_VALIDATION_SAMPLES = 4
    if os.path.exists(config.TRAINED_VIDEO_ID_NPY_FILE):
        video_ids = np.load(config.TRAINED_VIDEO_ID_NPY_FILE)
        print('video_ids loaded')
        print(len(video_ids))
    train_samples, validation_samples, test_samples, all_video_ids, caption_preprocessor, vocab_word_embeddings = prepareDataset(no_samples=1969, video_ids=video_ids)
    CAPTION_LEN = caption_preprocessor.caption_max_length
    OUTDIM_EMB = 200
    video_frame_input_shape = (None,2048)
    VOCAB_SIZE = caption_preprocessor.getVocabSize()
    print('Caption len : ' + str(CAPTION_LEN))
    print('Vocab size : ' + str(VOCAB_SIZE))
    print('Embedding weight matrix shape: ' + str(vocab_word_embeddings.shape))
    print('Starting training......')
    # not batch size cannot be less than number of captions per video 
    train_generator = dataGenerator(train_samples, caption_preprocessor.getWordToIndexDict(), CAPTION_LEN + 1, 92, VOCAB_SIZE, dataset_name='Training')
    val_generator = dataGenerator(validation_samples, caption_preprocessor.getWordToIndexDict(), CAPTION_LEN + 1, 50, VOCAB_SIZE, dataset_name='Validation')
    if CONTINUE_TRAINING is True or not os.path.exists(config.TRAINED_MODEL_HDF5_FILE):
        final_model = getModel(CAPTION_LEN + 1, OUTDIM_EMB, video_frame_input_shape, VOCAB_SIZE, embedding_weights=vocab_word_embeddings)
        if CONTINUE_TRAINING is True:
            print('continuing previous training')
            final_model.load_weights(config.TRAINED_MODEL_HDF5_FILE)
        all_video_ids_np = np.asarray(all_video_ids)
        np.save(config.TRAINED_VIDEO_ID_NPY_FILE, all_video_ids_np)
        history = final_model.fit_generator(train_generator, steps_per_epoch=STEPS_FOR_TRAIN_SAMPLES, epochs=NO_EPOCHS,
                                 verbose=1, validation_data=val_generator, validation_steps=STEPS_FOR_VALIDATION_SAMPLES,
                                 initial_epoch=0, callbacks=[BasicModelCallback(final_model, config.TRAINED_MODEL_FOLDER)])
        final_model.save_weights(config.TRAINED_MODEL_HDF5_FILE)

        loss_train = history.history['loss']
        loss_val = history.history['val_loss']
        acc_train = history.history['accuracy']
        acc_val = history.history['val_accuracy']

        util.writeLossAndAccuracyToFile(config.LOSS_ACCURACY_FILE, NO_EPOCHS, loss_train, loss_val, acc_train, acc_val)
        print(loss_train)
        print(loss_val)
        print(acc_train)
        print(acc_val)
        #loss_val.extend(loss_val_list[len(loss_val_list) - 1] * (len(loss_train_list) - len(loss_val_list)))
        
        epochs = range(1, NO_EPOCHS + 1)
        plt.plot(epochs, loss_train, 'g', label='Training loss')
        plt.plot(epochs, loss_val, 'b', label='validation loss')
        plt.title('Training and Validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
    else:
        final_model = getModel(CAPTION_LEN + 1, OUTDIM_EMB, video_frame_input_shape, VOCAB_SIZE)
        final_model.load_weights(config.TRAINED_MODEL_HDF5_FILE)
        print('Trained model weights exported')
    # test_video_index =  3
    # video_id = list(test_samples.keys())[test_video_index]

    # #print(vocab_word_embeddings[caption_preprocessor.getWordToIndexDict()[tokens[1]]])
    # print('Predicting for video: ' + video_id)
    # predictFromModel(final_model, list(test_samples.values())[test_video_index], caption_preprocessor.getWordToIndexDict(), caption_preprocessor.getIndexToWordDict(), CAPTION_LEN + 1)
    
    predictTestSamples(final_model, test_samples, caption_preprocessor.getWordToIndexDict(), caption_preprocessor.getIndexToWordDict(), CAPTION_LEN + 1)