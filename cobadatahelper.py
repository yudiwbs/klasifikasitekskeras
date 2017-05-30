
import data_helpers

file_pos = "/media/yudiwbs/programdata/ubuntu/lombalazada/data/datacobakecil/gede.label1"
file_neg = "/media/yudiwbs/programdata/ubuntu/lombalazada/data/datacobakecil/gede.label0"

print("Loading data...")
(x_train, y_train), (x_test, y_test)  = data_helpers.load_data_and_labels(file_pos, file_neg)

print(len(x_train), 'train sequences')
print(len(x_test),  'test sequences')


for x in range(0, 3):
    print(x_train[x])
    print(y_train[x])