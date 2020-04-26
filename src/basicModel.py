from tensorflow.keras import Input, layers
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Dropout, BatchNormalization, Activation, Bidirectional, GRU, RepeatVector, Embedding, add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.regularizers import l2
from tensorflow.keras.backend import concatenate
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

def extractVideoInputs(video_frames, no_samples, train_validation_split, no_test_samples):
    train_validation_split = math.floor((no_samples - no_test_samples) * train_validation_split)
    train_samples = dict()
    val_samples = dict()
    test_samples = dict()

    if not os.path.exists(config.TRAINING_VIDEOID_FILE):
        validation_size_counter = 0
        train_size_counter = 0
        for key in video_frames.keys():
            if (validation_size_counter + train_size_counter) == (no_samples - no_test_samples):
                test_samples[key] = [video_frames[key]]
                continue
            if(validation_size_counter < train_validation_split):
                val_samples[key] = [video_frames[key]]
                validation_size_counter += 1
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
    
    print('Number of Training samples: ' + str(len(train_samples)))
    print('Number of Validation samples: ' + str(len(val_samples)))
    print('Number of Testing samples: ' + str(len(test_samples)))
    return train_samples, val_samples, test_samples


def prepareDataset(no_samples=200, train_validation_split=0.2, no_test_samples=15, video_ids=None):
    video_frames = vff.loadVideoFrameFeatures(config.PICKLE_FILE_PATH, no_samples, video_ids)
    print('video frame features loaded')
    
    train_samples, val_samples, test_samples = extractVideoInputs(video_frames, no_samples, train_validation_split, no_test_samples)
    final_video_ids = list(train_samples.keys())
    final_video_ids.extend(list(val_samples.keys()))

    video_captions = y2tc.filterCaptionsForSamples(config.CSV_FILE_PATH, final_video_ids)
    print('video captions loaded')
    caption_preprocessor = cp.CaptionPreprocessor(video_captions, word_freq_threshold=3)
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
    
    return train_samples, val_samples, test_samples, caption_preprocessor, vocab_word_embeddings

def getBasicModel(final_caption_length, embedding_dim, video_frame_shape, total_vocab_size, embedding_weights=None):
    # Caption input shape is (Captionlen+1, Outputembeddeding_dimension) i.e. (16, 300)
    # And when we fit, we will do it in batches so currently batch dimension will be (None, 16, 300)
    cmodel_input = Input(shape=(final_caption_length,), name='caption_input')
    cmodel_embedding = Embedding(total_vocab_size, embedding_dim, mask_zero=True, name='caption_embedding') (cmodel_input)
    cmodel_dense = TimeDistributed(Dense(512,kernel_initializer='random_normal')) (cmodel_embedding)
    #cmodel_dropout = TimeDistributed(Dropout(0.50)) (cmodel_dense)
    #cmodel_lstm = LSTM(512, return_sequences=True,kernel_initializer='random_normal') (cmodel_dropout)
    cmodel_lstm = LSTM(512, return_sequences=True,kernel_initializer='random_normal') (cmodel_dense)
    #cmodel.summary()

    # Video frames input shape is (number_of_frames, frame_features) i.e. (None, 2048) In case of Video2Description model it is (40, 2048)
    # And when we fit, we will do it in batches so currently batch dimension will be (None, None, 2048)

    input_shape_vid = video_frame_shape
    imodel_input = Input(shape=input_shape_vid)
    imodel_dense = TimeDistributed(Dense(1024, kernel_initializer='random_normal')) (imodel_input)
    imodel_dropout = TimeDistributed(Dropout(0.2)) (imodel_dense)
    imodel_batchnorm = TimeDistributed(BatchNormalization(axis=-1)) (imodel_dropout)
    imodel_active = Activation('tanh') (imodel_batchnorm)
    imodel_lstm = Bidirectional(LSTM(1024, return_sequences=False, kernel_initializer='random_normal')) (imodel_active)
    imodel_repeatvector = RepeatVector(final_caption_length) (imodel_lstm)

    combined_model = concatenate([cmodel_lstm, imodel_repeatvector], axis=-1)
    combined_model_dropout = TimeDistributed(Dropout(0.2)) (combined_model)
    combined_model_lstm = LSTM(1024,return_sequences=True, kernel_initializer='random_normal',recurrent_regularizer=l2(0.01)) (combined_model_dropout)
    combined_model_outputs = TimeDistributed(Dense(total_vocab_size, activation='softmax'))(combined_model_lstm)

    # using default optimizer and loss function
    final_model = Model(inputs=[cmodel_input, imodel_input], outputs= [combined_model_outputs])
    #final_model.summary()
    print('Model created successfully')
    return final_model

def applyEmbeddingsAndCompile(model, embedding_weights):
    #optimizer = RMSprop(lr=0.0005, rho=0.9, epsilon=1e-8, decay=0)
    optimizer = Adam(learning_rate=0.00005, beta_1=0.9, beta_2=0.99, epsilon=1e-07, amsgrad=True, name='Adam')
    if embedding_weights is not None:
        #model_dict = {i: v for i, v in enumerate(model.layers)}
        #print(model_dict)
        print('embedding weights found. Set layer to non-trainable')
        model.layers[5].set_weights([embedding_weights])
        model.layers[5].trainable = False
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    print('Model compiled successfully')
    return model

def dataGenerator(training_set, wordtoidx, max_caption_length, num_videos_per_batch, vocab_size, dataset_name='Training'):
    # x1 - Training data for videos
    # x2 - The caption that goes with each photo
    # y - The predicted rest of the caption
    x1, x2, y = [], [], []
    current_batch_item_count=1
    #print('Dataset name: ' + dataset_name)
    while True:
        for key, values in training_set.items():
            current_batch_item_count+=1
            frame_features = values[0]
            single_video_captions = values[1]
            video_id = key
            for caption_item in single_video_captions:
                in_caption_item = f'{cp.START_KEYWORD} {caption_item}'
                in_seq = [wordtoidx[word] for word in in_caption_item.split(' ') if word in wordtoidx]
                for i in range(max_caption_length-len(in_seq)):
                    in_seq.append(wordtoidx[cp.NONE_KEYWORD])

                out_caption_item = f'{caption_item} {cp.STOP_KEYWORD}'
                out_caption_seq = [wordtoidx[word] for word in out_caption_item.split(' ') if word in wordtoidx]
                for i in range(max_caption_length-len(out_caption_seq)):
                    out_caption_seq.append(wordtoidx[cp.NONE_KEYWORD])
                out_seq = list()
                for count in range(max_caption_length):
                    out_seq.append(to_categorical([out_caption_seq[count]], num_classes=vocab_size)[0])
                x1.append(in_seq)
                x2.append(frame_features)
                y.append(out_seq)
            if current_batch_item_count==num_videos_per_batch:
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
    # video_dummy_caption = [wordtoidx[word] for word in dummy_caption.split(' ') if word in wordtoidx]
    # for i in range(max_caption_length-len(video_dummy_caption)):
    #     video_dummy_caption.append(wordtoidx[cp.NONE_KEYWORD])
    print(video_dummy_caption)
    input_sequence = pad_sequences([video_dummy_caption], maxlen=max_caption_length)
    print(list(input_sequence))
    captionoutput = model.predict([list(input_sequence), [video_frame_input]])
    print('Prediction done!')
    print(captionoutput.shape)
    print(np.argmax(captionoutput[0][3]))
    print(np.argmax(captionoutput[0][7]))
    in_text = ''
    for oneword in captionoutput[0]:
        yhat = np.argmax(oneword)
        word = idxtoword[yhat]
        in_text += ' ' + word
        if word == cp.STOP_KEYWORD:
            break
    print(original_video_caption_input)
    print(in_text)

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
        if epoch % 9 is 0 and epoch is not 0:
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
    if os.path.exists(config.TRAINED_VIDEO_ID_NPY_FILE):
        video_ids = np.load(config.TRAINED_VIDEO_ID_NPY_FILE)
        print('video_ids loaded')
        print(len(video_ids))
    train_samples, validation_samples, test_samples, caption_preprocessor, vocab_word_embeddings = prepareDataset(no_samples=915, video_ids=video_ids)
    # print(train_samples.keys())
    # print(test_samples.keys())
    all_video_ids = list(validation_samples.keys())
    all_video_ids.extend(list(train_samples.keys()))
    all_video_ids.extend(list(test_samples.keys()))
    # print(train_samples_list[0][1])
    CAPTION_LEN = caption_preprocessor.caption_max_length
    OUTDIM_EMB = 200
    video_frame_input_shape = (None,2048)
    VOCAB_SIZE = caption_preprocessor.getVocabSize() + 1
    print('Caption len : ' + str(CAPTION_LEN))
    print('Vocab size : ' + str(VOCAB_SIZE))
    print('Embedding weight matrix shape: ' + str(vocab_word_embeddings.shape))
    final_model = getBasicModel(CAPTION_LEN + 1, OUTDIM_EMB, video_frame_input_shape, VOCAB_SIZE)
    print('Starting training......')
    train_generator = dataGenerator(train_samples, caption_preprocessor.getWordToIndexDict(), CAPTION_LEN + 1, 20, VOCAB_SIZE, dataset_name='Training')
    val_generator = dataGenerator(validation_samples, caption_preprocessor.getWordToIndexDict(), CAPTION_LEN + 1, 10, VOCAB_SIZE, dataset_name='Validation')
    if CONTINUE_TRAINING is True or not os.path.exists(config.TRAINED_MODEL_HDF5_FILE):
        applyEmbeddingsAndCompile(final_model, vocab_word_embeddings)
        if CONTINUE_TRAINING is True:
            print('continuing previous training')
            final_model.load_weights(config.TRAINED_MODEL_HDF5_FILE)
        all_video_ids_np = np.asarray(all_video_ids)
        np.save(config.TRAINED_MODEL_FOLDER + 'trainedVideoIds_1.npy', all_video_ids_np)
        history = final_model.fit_generator(train_generator, steps_per_epoch=36, epochs=10,
                                 verbose=1, validation_data=val_generator, validation_steps=18,
                                 initial_epoch=0, callbacks=[BasicModelCallback(final_model, config.TRAINED_MODEL_FOLDER)])
        final_model.save_weights(config.TRAINED_MODEL_HDF5_FILE)
        all_video_ids_np = np.asarray(all_video_ids)
        np.save(config.TRAINED_VIDEO_ID_NPY_FILE, all_video_ids_np)

        loss_train = history.history['loss']
        loss_val = history.history['val_loss']
        print(loss_train)
        print(loss_val)
        #loss_val.extend(loss_val_list[len(loss_val_list) - 1] * (len(loss_train_list) - len(loss_val_list)))
        
        epochs = range(1,10)
        plt.plot(epochs, loss_train[:9], 'g', label='Training loss')
        plt.plot(epochs, loss_val[:9], 'b', label='validation loss')
        plt.title('Training and Validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
    else:
        final_model.load_weights(config.TRAINED_MODEL_HDF5_FILE)
        print('Trained model weights exported')
    test_video_index =  8
    video_id = list(test_samples.keys())[test_video_index]

    original_video_caption_input = list(test_samples.values())[test_video_index][1][0]
    tokens = original_video_caption_input.split(' ')
    #print(vocab_word_embeddings[caption_preprocessor.getWordToIndexDict()[tokens[1]]])
    #print('Predicting for video: ' + video_id)
    predictFromModel(final_model, list(train_samples.values())[test_video_index], caption_preprocessor.getWordToIndexDict(), caption_preprocessor.getIndexToWordDict(), CAPTION_LEN + 1)