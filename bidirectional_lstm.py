'''Train a Bidirectional LSTM on the IMDB sentiment classification task.
https://github.com/fchollet/keras/blob/master/examples/imdb_bidirectional_lstm.py
Output after 4 epochs on CPU: ~0.8146
Time per epoch on CPU (Core i7): ~150s.
'''

from __future__ import print_function
import numpy as np

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.datasets import imdb

from keras.utils import plot_model
import matplotlib.pyplot as plt
import data_helpers

max_features = 20000
# cut texts after this number of words
# (among top max_features most common words)
maxlen = 100
batch_size = 32

# print('Loading data...')
# (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

#hati2 label1 dan label0 jangan sampai lupa
file_pos = "/media/yudiwbs/programdata/ubuntu/lombalazada/data/persiapanrun12/out_clarity_train_rapi_casefold_vocabbuang.label1"
file_neg = "/media/yudiwbs/programdata/ubuntu/lombalazada/data/persiapanrun12/out_clarity_train_rapi_casefold_vocabbuang.label0"
file_model = "/media/yudiwbs/programdata/ubuntu/lombalazada/data/validasi/run12/model_run12_clarity_bidirectional_lstm.h5"
#coba

#file_model = "/media/yudiwbs/programdata/ubuntu/lombalazada/data/validasi/run12/model_kecil_run12coba1.h5"

print('Loading data...')
# (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

(x_train, y_train), (x_test, y_test)  = data_helpers.load_data_and_labels(file_pos, file_neg)

print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
y_train = np.array(y_train)
y_test = np.array(y_test)

model = Sequential()
model.add(Embedding(max_features, 128, input_length=maxlen))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

print('Train...')
history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=4,
          validation_data=[x_test, y_test])



score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print('\n\nTest score:', score)
print('Test accuracy:', acc)

print("Save model:")
model.save(file_model)

# cumua dua epoch, tdk terlalu terlihat
# list all data in history
print(history.history.keys())
# # summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# # summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()