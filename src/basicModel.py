from tensorflow.keras import Input, layers
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Dropout, BatchNormalization, Activation, Bidirectional, GRU, RepeatVector, Embedding, add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.regularizers import l2
from tensorflow.keras.backend import concatenate
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from keras import callbacks
import os
import numpy as np
import math
import videoFrameFeatures as vff
import youtube2TextCaptions as y2tc
import config
import captionPreprocess as cp
import gloveEmbeddings as ge

def prepareDataset(no_samples=100, test_size=0.2, video_ids=None):
    video_frames = vff.loadVideoFrameFeatures(config.PICKLE_FILE_PATH, no_samples, video_ids)
    print('video frame features loaded')
    final_video_ids = list(video_frames.keys())
    video_captions = y2tc.filterCaptionsForSamples(config.CSV_FILE_PATH,final_video_ids)
    print('video captions loaded')
    caption_preprocessor = cp.CaptionPreprocessor(video_captions, word_freq_threshold=1)
    print('Final word count: ' + str(caption_preprocessor.getVocabSize()))
    #print(caption_preprocessor.getCaptionsVocabList())
    print('video captions preprocessed')
    glove_embedding = ge.GloveEmbedding(config.GLOVE_200_DIM_FILE, 200)
    print('glove embedding loaded')
    vocab_word_embeddings = glove_embedding.getEmbeddingVectorFor(caption_preprocessor.getCaptionsVocabList(), caption_preprocessor.getVocabSize())
    preprocessed_video_captions = caption_preprocessor.caption_inputs
    test_size = math.floor(no_samples * test_size)
    test_size_counter = 0
    train_samples = dict()
    test_samples = dict()
    for key in video_frames.keys():
        if(test_size_counter < test_size):
            test_samples[key] = [video_frames[key], preprocessed_video_captions[key]]
            test_size_counter += 1
            continue
        train_samples[key] = [video_frames[key], preprocessed_video_captions[key]]
    
    return train_samples, test_samples, caption_preprocessor, vocab_word_embeddings

def getBasicModel(final_caption_length, embedding_dim, video_frame_shape, total_vocab_size, embedding_weights=None):
    # Caption input shape is (Captionlen+1, Outputembeddeding_dimension) i.e. (16, 300)
    # And when we fit, we will do it in batches so currently batch dimension will be (None, 16, 300)

    cmodel_input = Input(shape=(final_caption_length,), name='caption_input')
    cmodel_embedding = Embedding(total_vocab_size, embedding_dim, mask_zero=True, name='caption_embedding') (cmodel_input)
    cmodel_dense = TimeDistributed(Dense(512,kernel_initializer='random_normal')) (cmodel_embedding)
    cmodel_dropout = TimeDistributed(Dropout(0.50)) (cmodel_dense)
    cmodel_lstm = LSTM(512, return_sequences=True,kernel_initializer='random_normal') (cmodel_dropout)
    #cmodel.summary()

    # Video frames input shape is (number_of_frames, frame_features) i.e. (None, 2048) In case of Video2Description model it is (40, 2048)
    # And when we fit, we will do it in batches so currently batch dimension will be (None, None, 2048)

    input_shape_vid = video_frame_shape
    imodel_input = Input(shape=input_shape_vid)
    imodel_dropout = TimeDistributed(Dropout(0.5)) (imodel_input)
    imodel_dense = TimeDistributed(Dense(1024, activation='relu')) (imodel_dropout)
    imodel_batchnorm = TimeDistributed(BatchNormalization(axis=-1)) (imodel_dense)
    imodel_dense = TimeDistributed(Dense(512, activation='relu')) (imodel_batchnorm)
    imodel_lstm = LSTM(512, return_sequences=False, kernel_initializer='random_normal') (imodel_dense)
    imodel_repeatvector = RepeatVector(final_caption_length) (imodel_lstm)

    combined_model = concatenate([cmodel_lstm, imodel_repeatvector], axis=-1)
    combined_model_dropout = TimeDistributed(Dropout(0.2)) (combined_model)
    combined_model_lstm = LSTM(1024,return_sequences=True, kernel_initializer='random_normal',recurrent_regularizer=l2(0.01)) (combined_model_dropout)
    combined_model_outputs = TimeDistributed(Dense(total_vocab_size, activation='softmax'))(combined_model_lstm)

    # using default optimizer and loss function
    final_model = Model(inputs=[cmodel_input, imodel_input], outputs= [combined_model_outputs])
    final_model.summary()
    print('Model created successfully')
    return final_model

def applyEmbeddingsAndCompile(model, embedding_weights):
    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-8, decay=0)
    if embedding_weights is not None:
        print('embedding weights found. Set layer to non-trainable')
        model.layers[4].set_weights([embedding_weights])
        model.layers[4].trainable = False
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    print('Model compiled successfully')
    return model

def dataGenerator(training_set, wordtoidx, max_caption_length, num_videos_per_batch, vocab_size):
    # x1 - Training data for videos
    # x2 - The caption that goes with each photo
    # y - The predicted rest of the caption
    x1, x2, y = [], [], []
    current_batch_item_count=0
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
                yield ([np.array(x1), np.array(x2)], np.array(y))
                x1, x2, y = [], [], []
                current_batch_item_count=0

def predictFromModel(model, video_sample, wordtoidx, idxtoword, max_caption_length):
    video_frame_input = video_sample[0]
    original_video_caption_input = video_sample[1][0]
    dummy_caption = [cp.START_KEYWORD]
    dummy_caption.extend([cp.NONE_KEYWORD] * (max_caption_length - 1))
    video_dummy_caption = [wordtoidx[word] for word in dummy_caption if word in wordtoidx]
    print(video_dummy_caption)
    input_sequence = pad_sequences([video_dummy_caption], maxlen=max_caption_length)
    print(list(input_sequence))
    captionoutput = model.predict([list(input_sequence), [video_frame_input]])
    print('Prediction done!')
    print(captionoutput.shape)
    print(wordtoidx[cp.START_KEYWORD])
    print(wordtoidx[cp.STOP_KEYWORD])
    print(captionoutput[0][3][0])
    print(captionoutput[0][3][1])
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
    def __init__(self):
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
    
    def on_batch_end(self, batch, logs={}):
        print("Batch %d ends" % batch)
        loss = logs['loss']
        # acc  = logs['acc']
        print('Batch Training loss: ' + str(loss))
        # print('Batch Training Accuracy: ' + str(acc))

if __name__ == "__main__":
    video_ids = None
    if os.path.exists(config.TRAINED_VIDEO_ID_NPY_FILE):
        video_ids = np.load(config.TRAINED_VIDEO_ID_NPY_FILE)
        print('video_ids loaded')
        print(video_ids)
    train_samples, test_samples, caption_preprocessor, vocab_word_embeddings = prepareDataset(video_ids=video_ids)
    # print(train_samples.keys())
    # print(test_samples.keys())
    all_video_ids = list(test_samples.keys())
    all_video_ids.extend(list(train_samples.keys()))
    train_samples_list = list(train_samples.values())
    test_samples_list = list(test_samples.values())
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
    train_generator = dataGenerator(train_samples, caption_preprocessor.getWordToIndexDict(), CAPTION_LEN + 1, 1, VOCAB_SIZE)
    if not os.path.exists(config.TRAINED_MODEL_HDF5_FILE):
        applyEmbeddingsAndCompile(final_model, vocab_word_embeddings)
        final_model.fit_generator(train_generator, steps_per_epoch=80, epochs=15,
                                 verbose=1,
                                 initial_epoch=0, callbacks=[BasicModelCallback()])
        final_model.save_weights(config.TRAINED_MODEL_HDF5_FILE)
        all_video_ids_np = np.asarray(all_video_ids)
        np.save(config.TRAINED_VIDEO_ID_NPY_FILE, all_video_ids_np)
    else:
        final_model.load_weights(config.TRAINED_MODEL_HDF5_FILE)
        print('Trained model weights exported')
    predictFromModel(final_model, test_samples_list[0], caption_preprocessor.getWordToIndexDict(), caption_preprocessor.getIndexToWordDict(), CAPTION_LEN + 1)