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
from keras.models import load_model
import data_helpers

from keras.datasets import imdb


'''
Note:
batch_size is highly sensitive.
'''

#hati2 label1 dan label0 jangan sampai lupa
#file_pred = "/media/yudiwbs/programdata/ubuntu/lombalazada/data/validasi/data_valid_rapi_casefold_vocabbuang_2.csv"
file_pred = "//media/yudiwbs/programdata/ubuntu/lombalazada/data/persiapanrun12/cobatest"
maxlen = 100
#file_model = "/media/yudiwbs/programdata/ubuntu/lombalazada/data/validasi/run12/model_run12coba1"
file_model = "/media/yudiwbs/programdata/ubuntu/lombalazada/data/validasi/run12/model_kecil_run12coba1.h5"

batch_size = 30


print('Loading data...')
# (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_test  = data_helpers.pred_load_data_and_labels(file_pred)
print(len(x_test),  'test sequences')

print('Pad sequences (samples x time)')
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_test shape:', x_test.shape)


#load model
model = load_model(file_model)

#score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
#self, x, batch_size=32,
hasil =  model.predict(x=x_test, batch_size=batch_size, verbose=1)

print ("hasil:")
for i in range(len(hasil)):
    print("i ke "+str(i)+":"+str(hasil[i]))


#print('\n\nTest score:', score)
#print('Test accuracy:', acc)

