#import tensorflow as tf
# from keras.models import Sequential
# from keras.models import Model
# from keras.layers import TimeDistributed, Dense, Input, Flatten, GlobalAveragePooling2D, Bidirectional
# from keras.layers import Embedding, LSTM, GRU, BatchNormalization, Lambda
# from keras.layers import Dropout, Flatten, Activation
# from keras.backend import concatenate, repeat
# from keras.regularizers import l2
# from keras.optimizers import RMSprop
# from keras.layers.core import RepeatVector
from tensorflow.keras import Input, layers
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Dropout, BatchNormalization, Activation, 
                Bidirectional, GRU, RepeatVector, Embedding
from tensorflow.keras.backend import concatenate, repeat
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.regularizers import l2

CAPTION_LEN = 15
OUTDIM_EMB = 300
video_frame_input_shape = (None,2048)
VOCAB_SIZE = 9448

# def prepareModel():
#     cmodel  = Sequential()
#     cmodel.add(TimeDistributed(Dense(512,kernel_initializer='random_normal'), input_shape=(CAPTION_LEN+1,OUTDIM_EMB )))
#     cmodel.add(LSTM(512, return_sequences=True,kernel_initializer='random_normal'))
#     cmodel.summary()

#     # input_shape_audio = VideoHandler.AUDIO_FEATURE
#     # amodel = Sequential()
#     # amodel.add(GRU(128,
#     #                 dropout=0.2,
#     #                 recurrent_dropout=0.2,
#     #                 return_sequences=True,
#     #                 input_shape=input_shape_audio))
#     # amodel.add(BatchNormalization())
#     # amodel.add(GRU(64,
#     #                 dropout=0.2,
#     #                 recurrent_dropout=0.2,
#     #                 return_sequences=True))
#     # amodel.add(BatchNormalization())
#     # amodel.add(Flatten())
#     # amodel.add(RepeatVector(CAPTION_LEN + 1))
#     # amodel.summary()

#     input_shape_vid = video_frame_input_shape
#     imodel = Sequential()
#     imodel.add(TimeDistributed(Dense(1024,kernel_initializer='random_normal'), input_shape=input_shape_vid))
#     imodel.add(TimeDistributed(Dropout(0.20)))
#     imodel.add(TimeDistributed(BatchNormalization(axis=-1)))
#     imodel.add(Activation('tanh'))
#     imodel.add(Bidirectional(GRU(1024, return_sequences=False, kernel_initializer='random_normal')))
#     imodel.add(RepeatVector(CAPTION_LEN + 1))

#     imodel.summary()

#     model = Sequential()
#     # model.add(Merge([cmodel,amodel,imodel],mode='concat'))
#     model.add(Concatenate([cmodel,imodel]))
#     model.add(TimeDistributed(Dropout(0.2)))
#     model.add(LSTM(1024,return_sequences=True, kernel_initializer='random_normal',recurrent_regularizer=l2(0.01)))
#     model.add(TimeDistributed(Dense(VOCAB_SIZE,kernel_initializer='random_normal')))
#     model.add(Activation('softmax'))
#     optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-8, decay=0)
#     model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
#     model.build((None, CAPTION_LEN+1, OUTDIM_EMB))
#     model.summary()

def prepareModelFunctionalWay():
    # Caption input shape is (Captionlen+1, Outputembeddeding_dimension) i.e. (16, 300)
    # And when we fit, we will do it in batches so currently batch dimension will be (None, 16, 300)

    cmodel_input = Input(shape=(CAPTION_LEN+1,OUTDIM_EMB), name='caption_input')
    cmodel_dense = TimeDistributed(Dense(512,kernel_initializer='random_normal')) (cmodel_input)
    cmodel_lstm = LSTM(512, return_sequences=True,kernel_initializer='random_normal') (cmodel_dense)
    #cmodel.summary()

    # Video frames input shape is (number_of_frames, frame_features) i.e. (None, 2048) In case of Video2Description model it is (40, 2048)
    # And when we fit, we will do it in batches so currently batch dimension will be (None, None, 2048)

    input_shape_vid = video_frame_input_shape
    imodel_input = Input(shape=input_shape_vid)
    #imodel_input = 
    imodel_dense = TimeDistributed(Dense(1024,kernel_initializer='random_normal')) (imodel_input)
    imodel_dropout = TimeDistributed(Dropout(0.20)) (imodel_dense)
    imodel_batchnorm = TimeDistributed(BatchNormalization(axis=-1)) (imodel_dropout)
    imodel_activatation = Activation('tanh') (imodel_batchnorm)
    imodel_biGRU = Bidirectional(GRU(1024, return_sequences=False, kernel_initializer='random_normal')) (imodel_activatation)
    imodel_repeatvector = RepeatVector(CAPTION_LEN + 1) (imodel_biGRU)
    #imodel_repeatvector = repeat(imodel_biGRU,CAPTION_LEN + 1)
    #repeater=Lambda(lambda x: repeat(x, CAPTION_LEN + 1) )
    #imodel_repeatvector = repeater(imodel_biGRU)


    # Merge captions and video frames inputs
    mixed_model = concatenate([cmodel_lstm,imodel_repeatvector], axis=-1)
    mixed_model_drop = TimeDistributed(Dropout(0.2)) (mixed_model)
    mixed_model_lstm = LSTM(1024,return_sequences=True, kernel_initializer='random_normal',recurrent_regularizer=l2(0.01)) (mixed_model_drop)
    mixed_model_dense = TimeDistributed(Dense(VOCAB_SIZE,kernel_initializer='random_normal', activation='softmax')) (mixed_model_lstm)
    #mixed_model_softmax = Activation('softmax') (mixed_model_dense)

    final_model = Model(inputs=[cmodel_input, imodel_input], outputs= [mixed_model_dense])
    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-8, decay=0)
    final_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    final_model.summary()

#prepareModel()
prepareModelFunctionalWay()