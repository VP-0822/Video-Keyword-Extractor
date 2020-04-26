# from keras.models import Model
# from keras.layers import Input, Dense, TimeDistributed
# from keras.layers import LSTM
# from numpy import array
# # define model
# inputs1 = Input(shape=(3, 1)) # 3 timeframes/video frames with 1 feature
# # [Following statement] should output 1 output (2 features) which is hidden state/output of last timeframe
# #lstm1 = LSTM(2)(inputs1) # has 2 hidden units
# # [Following statement] should output 3 outputs (2 features) which are hidden states/outputs of all 3 timeframes
# lstm1 = LSTM(2, return_sequences=True)(inputs1) # has 2 hidden units

# #dense = Dense(2) (lstm1)

# dense = TimeDistributed(Dense(2)) (lstm1)
# model = Model(inputs=inputs1, outputs=dense)
# # define input data
# data = array([0.1, 0.2, 0.3]).reshape((1,3,1))
# # make and show predic)tion
# print(model.predict(data))

from tensorflow.keras import Input, layers
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Activation, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from numpy import array

input_shape_vid = (3, 1)
imodel_input = Input(shape=input_shape_vid) # (1, 3, 1)

# MODEL-1
# # [Following layer] returns (1, 2), 2 hidden unit outputs for last frames
# imodel_lstm = LSTM(2, kernel_initializer='random_normal', return_sequences=False) (imodel_input)
# # [Following layer without TimeDistributed] returns (1, 2), 2 outputs for all timeframe together
# imodel_dense = Dense(2, kernel_initializer='random_normal') (imodel_lstm) 

# MODEL-2
# # [Following layer] returns (1, 3, 2), 2 outputs for each of the frames
# imodel_lstm = LSTM(2, kernel_initializer='random_normal', return_sequences=True) (imodel_input)
# # [Following layer without TimeDistributed] returns (1, 3, 2), 2 outputs for all timeframe together
# imodel_dense = TimeDistributed(Dense(2, kernel_initializer='random_normal')) (imodel_lstm)

# MODEL-3
# # [Following layer] returns (1, 3, 2), 2 outputs for each of the frames
imodel_lstm = LSTM(2, kernel_initializer='random_normal', return_sequences=True) (imodel_input)
# # [Following layer without TimeDistributed] returns (1, 3, 2), 2 outputs for all timeframe together
imodel_dense = Dense(5, kernel_initializer='random_normal') (imodel_lstm) 



#imodel_dropout = TimeDistributed(Dropout(0.2)) (imodel_dense)
#imodel_batchnorm = TimeDistributed(BatchNormalization(axis=-1)) (imodel_dropout)
# imodel_active = Activation('tanh') (imodel_dense)

final_model = Model(inputs=imodel_input, outputs= [imodel_dense])


optimizer = Adam(learning_rate=0.00005, beta_1=0.9, beta_2=0.99, epsilon=1e-07, amsgrad=True, name='Adam')
final_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
final_model.summary()

# for layer in final_model.layers:
#     print(layer.output_shape)

length = 3
seq = array([i/float(length) for i in range(length)])
print(seq)

X = seq.reshape(1, 3, 1) # 1 batch size, 5 no of time frames/video frames, last 1 no of features in one frame
print('======= X ======')
print(X)
y = seq.reshape(1, 3) # 1 batch size, 5 outputs for 5 frames
print('========= y ========')
print(y)
#history = final_model.fit(X, y, epochs=500, batch_size=1, verbose=0)

result = final_model.predict(X, batch_size=1, verbose=0)
print('========= result ========')
print(result)