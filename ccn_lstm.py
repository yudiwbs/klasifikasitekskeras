'''Train a recurrent convolutional network on classification task.
  sumber: https://github.com/fchollet/keras/blob/master/examples/imdb_cnn_lstm.py

  tadinya dari imbd, tapi di modifikasi untuk file

'''
from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D

#yw untuk plot model
from keras.utils import plot_model
import matplotlib.pyplot as plt
import data_helpers

from keras.datasets import imdb

# Embedding
max_features = 20000
maxlen = 100
embedding_size = 128

# Convolution
kernel_size = 5
filters = 64
pool_size = 4

# LSTM
lstm_output_size = 70

# Training
batch_size = 5  #default 30
epochs = 10  #default 2

'''
Note:
batch_size is highly sensitive.
Only 2 epochs are needed as the dataset is very small.
'''

#hati2 label1 dan label0 jangan sampai lupa
# file_pos = "/media/yudiwbs/programdata/ubuntu/lombalazada/data/persiapanrun12/out_clarity_train_rapi_casefold_vocabbuang.label1"
# file_neg = "/media/yudiwbs/programdata/ubuntu/lombalazada/data/persiapanrun12/out_clarity_train_rapi_casefold_vocabbuang.label0"
# file_model = "/media/yudiwbs/programdata/ubuntu/lombalazada/data/validasi/run12/model_run12coba1.h5"
#coba

file_pos = "/media/yudiwbs/programdata/ubuntu/lombalazada/data/persiapanrun12/kecil.label1"
file_neg = "/media/yudiwbs/programdata/ubuntu/lombalazada/data/persiapanrun12/kecil.label0"
file_model = "/media/yudiwbs/programdata/ubuntu/lombalazada/data/validasi/run12/model_kecil_run12coba1.h5"

print('Loading data...')
# (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

(x_train, y_train), (x_test, y_test)  = data_helpers.load_data_and_labels(file_pos, file_neg)
print(len(x_train), 'train sequences')
print(len(x_test),  'test sequences')

# print("train")
# for i in range(len(x_train)):
#     print("y="+str(y_train[i]))
#
# print("test")
# for i in range(len(x_test)):
#     print("y="+str(y_test[i]))


print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')

model = Sequential()
model.add(Embedding(max_features, embedding_size, input_length=maxlen))
model.add(Dropout(0.25))
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
model.add(MaxPooling1D(pool_size=pool_size))
model.add(LSTM(lstm_output_size))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))

score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print('\n\nTest score:', score)
print('Test accuracy:', acc)

#save model

model.save(file_model)

#arsitektur model
#plot_model(model, to_file='model.png')


# cumua dua epoch, tdk terlalu terlihat
# list all data in history
# print(history.history.keys())
# # summarize history for accuracy
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# # summarize history for loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
#

