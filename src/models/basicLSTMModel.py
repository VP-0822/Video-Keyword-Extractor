from tensorflow.keras import Input, layers
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Dropout, BatchNormalization, Activation, RepeatVector, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.backend import concatenate
from tensorflow.keras.regularizers import l2

def getModel(final_caption_length, embedding_dim, video_frame_shape, total_vocab_size, embedding_weights=None, dropOutAtCaption=False):
    # Caption input shape is (Captionlen+1, Outputembeddeding_dimension) i.e. (16, 300)
    # And when we fit, we will do it in batches so currently batch dimension will be (None, 16, 300)
    cmodel_input = Input(shape=(final_caption_length,), name='caption_input')
    cmodel_embedding = Embedding(total_vocab_size, embedding_dim, mask_zero=True, name='caption_embedding') (cmodel_input)
    cmodel_dense = TimeDistributed(Dense(512,kernel_initializer='random_normal', name='caption_dense')) (cmodel_embedding)
    if dropOutAtCaption is True:
        cmodel_dropout = TimeDistributed(Dropout(0.20, name='caption_dropout')) (cmodel_dense)
        cmodel_lstm = LSTM(512, return_sequences=True,kernel_initializer='random_normal', name='caption_lstm') (cmodel_dropout)
    else:
        cmodel_lstm = LSTM(512, return_sequences=True,kernel_initializer='random_normal', name='caption_lstm') (cmodel_dense)

    # Video frames input shape is (number_of_frames, frame_features) i.e. (None, 2048) In case of Video2Description model it is (40, 2048)
    # And when we fit, we will do it in batches so currently batch dimension will be (None, None, 2048)
    input_shape_vid = video_frame_shape
    imodel_input = Input(shape=input_shape_vid, name='frame_lstm')
    imodel_dense = TimeDistributed(Dense(1024, kernel_initializer='random_normal', name='frame_dense')) (imodel_input)
    imodel_dropout = TimeDistributed(Dropout(0.2, name='frame_dropuout')) (imodel_dense)
    imodel_batchnorm = TimeDistributed(BatchNormalization(axis=-1, name='frame_batchnorm')) (imodel_dropout)
    imodel_active = Activation('tanh', name='frame_tanh_activation') (imodel_batchnorm)
    imodel_lstm = LSTM(1024, return_sequences=False, kernel_initializer='random_normal', name='frame_lstm') (imodel_active)
    imodel_repeatvector = RepeatVector(final_caption_length, name='frame_repeatvector') (imodel_lstm)

    combined_model = concatenate([cmodel_lstm, imodel_repeatvector], axis=-1)
    combined_model_dropout = TimeDistributed(Dropout(0.2, name='final_dropout')) (combined_model)
    combined_model_lstm = LSTM(1024,return_sequences=True, kernel_initializer='random_normal',recurrent_regularizer=l2(0.01), name='final_lstm') (combined_model_dropout)
    combined_model_outputs = TimeDistributed(Dense(total_vocab_size, activation='softmax', name='frame_dense_with_activation'))(combined_model_lstm)

    final_model = Model(inputs=[cmodel_input, imodel_input], outputs= [combined_model_outputs])

    if embedding_weights is not None:
        print('embedding weights found. Set layer to non-trainable')
        final_model.layers[5].set_weights([embedding_weights])
        final_model.layers[5].trainable = False
    
    return final_model

