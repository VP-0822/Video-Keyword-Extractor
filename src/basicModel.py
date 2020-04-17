from tensorflow.keras import Input, layers
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Dropout, BatchNormalization, Activation, Bidirectional, GRU, RepeatVector, Embedding, add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.regularizers import l2
from tensorflow.keras.backend import concatenate
import numpy as np
import math
import videoFrameFeatures as vff
import youtube2TextCaptions as y2tc
import config
import captionPreprocess as cp
import gloveEmbeddings as ge

def prepareDataset(no_samples=15, test_size=0.2):
    video_frames = vff.loadVideoFrameFeatures(config.PICKLE_FILE_PATH, no_samples)
    print('video frame features loaded')
    video_ids = list(video_frames.keys())
    video_captions = y2tc.filterCaptionsForSamples(config.CSV_FILE_PATH,video_ids)
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

def getBasicModel(final_caption_length, embedding_dim, video_frame_shape, total_vocab_size):
    # Caption input shape is (Captionlen+1, Outputembeddeding_dimension) i.e. (16, 300)
    # And when we fit, we will do it in batches so currently batch dimension will be (None, 16, 300)

    cmodel_input = Input(shape=(final_caption_length,), name='caption_input')
    cmodel_embedding = Embedding(final_caption_length, embedding_dim, mask_zero=True, name='caption_embedding') (cmodel_input)
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
    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-8, decay=0)
    final_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    final_model.summary()

# def dataGenerator(training_set, wordtoidx, max_length, num_videos_per_batch):
#     # x1 - Training data for videos
#     # x2 - The caption that goes with each photo
#     # y - The predicted rest of the caption
#     x1, x2, y = [], [], []
#     n=0
#     while True:
#         for key, values in training_set:
#             frame_features = values[0]
#             caption_vector = values[1]
#             video_id = key



#     while True:
#         for key, desc_list in training_captions.items():
#             n+=1
#             photo = photos[key+'.jpg']
#             # Each photo has 5 descriptions
#             for desc in desc_list:
#                 # Convert each word into a list of sequences.
#                 seq = [wordtoidx[word] for word in desc.split(' ') if word in wordtoidx]
#                 # Generate a training case for every possible sequence and outcome
#                 for i in range(1, len(seq)):
#                     in_seq, out_seq = seq[:i], seq[i]
#                     in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
#                     out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
#                     x1.append(photo)
#                     x2.append(in_seq)
#                     y.append(out_seq)
#                 if n==num_photos_per_batch:
#                     yield ([np.array(x1), np.array(x2)], np.array(y))
#                     x1, x2, y = [], [], []
#                     n=0

if __name__ == "__main__":
    train_samples, test_samples, caption_preprocessor, vocab_word_embeddings = prepareDataset()
    print(train_samples.keys())
    print(test_samples.keys())
    train_samples_list = list(train_samples.values())
    print(train_samples_list[0][1])
    CAPTION_LEN = caption_preprocessor.caption_max_length
    OUTDIM_EMB = 200
    video_frame_input_shape = (None,2048)
    VOCAB_SIZE = caption_preprocessor.getVocabSize() + 1
    #final_model = getBasicModel(CAPTION_LEN +1, OUTDIM_EMB, video_frame_input_shape, VOCAB_SIZE)