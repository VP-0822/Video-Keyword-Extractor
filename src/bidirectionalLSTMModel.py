from tensorflow.keras import Input, layers
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Dropout, BatchNormalization, Activation, Bidirectional, RepeatVector, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.backend import concatenate
from tensorflow.keras.regularizers import l2
from captionFrameModel import CaptionFrameModel
from modelCallback import BasicModelCallback
from simpleAttention import Attention
import tqdm
class BidirectionalLSTMModel(CaptionFrameModel):
    def __init__(self, captionPreprocessor, embedding_vector_dim, video_frame_shape, embedding_weights=None, dropOutAtFrame=0.2, dropOutAtCaption=None, dropOutAtFinal=0.2):
        self.embedding_dim = embedding_vector_dim
        self.video_frame_shape = video_frame_shape
        self.embedding_weights =  embedding_weights
        self.dropOutAtFrame = dropOutAtFrame
        self.dropOutAtCaption = dropOutAtCaption
        self.dropOutAtFinal = dropOutAtFinal
        self.captionPreprocessor = captionPreprocessor
        super().__init__(self.captionPreprocessor.getMaximumCaptionLength(), self.captionPreprocessor.getVocabSize())

    def buildModel(self, use_attention=False):
        # Caption input shape is (Captionlen+1, Outputembeddeding_dimension) i.e. (16, 300)
        # And when we fit, we will do it in batches so currently batch dimension will be (None, 16, 300)
        cmodel_input = Input(shape=(self.final_caption_length,), name='caption_input')
        cmodel_embedding = Embedding(self.total_vocab_size, self.embedding_dim, mask_zero=True, name='caption_embedding') (cmodel_input)
        cmodel_dense = TimeDistributed(Dense(512,kernel_initializer='random_normal', name='caption_dense')) (cmodel_embedding)
        if self.dropOutAtCaption is not None:
            cmodel_dropout = TimeDistributed(Dropout(self.dropOutAtCaption, name='caption_dropout')) (cmodel_dense)
            cmodel_lstm = LSTM(512, return_sequences=True,kernel_initializer='random_normal', name='caption_lstm') (cmodel_dropout)
        else:
            cmodel_lstm = LSTM(512, return_sequences=True,kernel_initializer='random_normal', name='caption_lstm') (cmodel_dense)

        # Video frames input shape is (number_of_frames, frame_features) i.e. (None, 2048) In case of Video2Description model it is (40, 2048)
        # And when we fit, we will do it in batches so currently batch dimension will be (None, None, 2048)
        input_shape_vid = self.video_frame_shape
        imodel_input = Input(shape=input_shape_vid, name='frame_lstm')
        imodel_dense = TimeDistributed(Dense(1024, kernel_initializer='random_normal', name='frame_dense')) (imodel_input)
        imodel_dropout = TimeDistributed(Dropout(self.dropOutAtFrame, name='frame_dropuout')) (imodel_dense)
        imodel_batchnorm = TimeDistributed(BatchNormalization(axis=-1, name='frame_batchnorm')) (imodel_dropout)
        imodel_active = Activation('tanh', name='frame_tanh_activation') (imodel_batchnorm)
        if use_attention is True:
            imodel_lstm = Bidirectional(LSTM(1024, return_sequences=True, kernel_initializer='random_normal', name='frame_bidirectional_lstm')) (imodel_active)
            imodel_lstm = Attention() (imodel_lstm)
        else:
            imodel_lstm = Bidirectional(LSTM(1024, return_sequences=False, kernel_initializer='random_normal', name='frame_bidirectional_lstm')) (imodel_active)
        imodel_repeatvector = RepeatVector(self.final_caption_length, name='frame_repeatvector') (imodel_lstm)

        combined_model = concatenate([cmodel_lstm, imodel_repeatvector], axis=-1)
        combined_model_dropout = TimeDistributed(Dropout(self.dropOutAtFinal, name='final_dropout')) (combined_model)
        combined_model_lstm = LSTM(1024,return_sequences=True, kernel_initializer='random_normal',recurrent_regularizer=l2(0.01), name='final_lstm') (combined_model_dropout)
        combined_model_outputs = TimeDistributed(Dense(self.total_vocab_size, activation='softmax', name='frame_dense_with_activation'))(combined_model_lstm)

        final_model = Model(inputs=[cmodel_input, imodel_input], outputs= [combined_model_outputs])

        if self.embedding_weights is not None:
            if use_attention is True:
                embedding_layer_index = 6
            else:
                embedding_layer_index = 5
            print('embedding weights found. Set layer to non-trainable')
            final_model.layers[embedding_layer_index].set_weights([self.embedding_weights])
            final_model.layers[embedding_layer_index].trainable = False
            
        self.model = final_model

    def compileModel(self, optimizer):
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.summary()

        print('Model created successfully')
    
    def loadTrainedModel(self, hdf5File):
        self.finalHdf5File = hdf5File
        print('continuing previous training')
        self.model.load_weights(hdf5File)
    
    def trainModel(self, trainingSamples, validationSamples,noOfEpochs, trainingBatchSize, trainingSteps, validationBatchSize, validationSteps, trainedModelFolderPath, hdf5File):
        self.finalHdf5File = hdf5File
        train_generator = self._dataGenerator(trainingSamples, self.captionPreprocessor.getWordToIndexDict(), trainingBatchSize, dataset_name='Training')
        val_generator = self._dataGenerator(validationSamples, self.captionPreprocessor.getWordToIndexDict(), validationBatchSize, dataset_name='Validation')

        history = self.model.fit_generator(train_generator, steps_per_epoch=trainingSteps, epochs=noOfEpochs,
                                 verbose=1, validation_data=val_generator, validation_steps=validationSteps,
                                 initial_epoch=0, callbacks=[BasicModelCallback(self.model, trainedModelFolderPath)])

        self.model.save_weights(self.finalHdf5File)
        return history
    
    def predictTestSamples(self, testSampleSet):
        all_prediction_statements = []
        all_prediction_statements.append('Video Id,Original caption,Predicted caption')
        print('Predicting test samples...')
        for key, values in tqdm.tqdm(testSampleSet.items()):
            video_id = key
            original_video_caption_input, output_caption_text = self._predictFromModel(self.model, values, self.captionPreprocessor.getWordToIndexDict(), self.captionPreprocessor.getIndexToWordDict())
            # print('Predicting for video: ' + video_id)
            csv_line = video_id
            # print('[Original caption]:')
            # print(original_video_caption_input)
            csv_line += '|' + original_video_caption_input
            # print('[Predicted caption]:')
            # print(output_caption_text)
            csv_line += '|' + output_caption_text
            # print('###########')
            all_prediction_statements.append(csv_line)
        return all_prediction_statements