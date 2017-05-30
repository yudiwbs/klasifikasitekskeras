import numpy as np
import re
from tensorflow.contrib import learn
import itertools
from collections import Counter

dev_sample_percentage = 0.1

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.

    hasil tahap pertama
    gabung data positif dan negatif
    contoh kalau ada dua data positif dan dua data negatif (dua file):
    returnya akan seperti ini (0 dan 1 diganti)
    my , , gallery square hijab light pink , fashion , women , muslim wear , ul li material non sheer shimmer chiffon li li sizes 52 x 52 inches or 56 x 56 inches li li cut with curved ends li ul , 49 0 , local
    0
    my , , eau de parfum spray 100ml 3 3oz , health beauty , bath body , hand foot care , formulated with oil free hydrating botanicals remarkably improves skin texture of hands restores soft , smooth refined hands , 128 0 , international
    0
    my , , 5 units of korea soothing amp moisture 99 aloe vera soothing gel 300ml , health beauty , skin care , moisturizers and cream , ul li the most popular and best aloe vera soothing gel li li aloe vera extracts 99 li li for all skin types li li moisturizing for firmer and healthier skin li ul , 59 0 , local
    1
    my , , 2016 new arrive women four seasons flats casual shoes women flat heel cow muscle outsole fashion flat women genuine leather shoes , fashion , women , shoes , ul li 2016 women genuine leather flats shoes li li fashion cow muscle slip on shoes li li women four season soft leather shoes li li female round toe pure color casual shoes li li women flat heel shoes with two style li li slip on loafers shoes and slippers shoes li
    1

    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words (dibersihkan)
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    #positive_labels = [[0, 1] for _ in positive_examples]
    #negative_labels = [[1, 0] for _ in negative_examples]
    positive_labels = [1 for _ in positive_examples]
    negative_labels = [0 for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)

    #y = positive_labels + negative_labels

    #print("ukuran y="+str(len(y)))
    #print("ukuran x="+str(len(x_text)))
    #setelah itu diproses menjadi vektor
    #return [x_text, y]
    #yw
    #input adalah hasil load_data_and_label
    #output mengikuti keras (x_train, y_train), (x_test, y_test)
    #def loadVektor(x_text,y):

    # Build vocabulary
    max_document_length = max([len(x.split(" ")) for x in x_text])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x = np.array(list(vocab_processor.fit_transform(x_text)))
    #print("ukuran x="+str(len(x)))
    #print(x[1])

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    return (x_train, y_train), (x_dev, y_dev)


def pred_load_data_and_labels(data_file):
    # Load data from files
    rows = list(open(data_file, "r").readlines())
    rows = [s.strip() for s in rows]

    # Split by words (dibersihkan)
    x_text = rows
    x_text = [clean_str(sent) for sent in x_text]

    # Build vocabulary
    max_document_length = max([len(x.split(" ")) for x in x_text])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x = np.array(list(vocab_processor.fit_transform(x_text)))
    return x


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
