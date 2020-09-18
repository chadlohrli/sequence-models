import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

# word2vec utils
def countWords(str):
    wordCount = 0;
    for c in str:
        if (c == ' '):
            wordCount = wordCount + 1
    return wordCount

def storeText(file_name, data):
    file = open(file_name, "w")
    file.write(data)
    file.close()


def save_embed(embed,fileName):
    if(not os.path.exists(os.getcwd() + '/w2vdata')):
        os.mkdir(os.getcwd() + '/w2vdata')
    with open('w2vdata/' + fileName + '.pkl', 'wb') as f:
        pickle.dump(parameters, f)
 

def load_embed(fileName):
    with open('w2vdata/'+ fileName + '.pkl','rb') as f:
        weights = pickle.load(f)
        return weights

# sequential utils
def softmax(x, deriv=False):
    e_x = np.exp(x - np.max(x))
    sx = e_x / e_x.sum(axis=0)
    if deriv:
        sx[y] -= 1.0
    return sx;


def loss(x, y):
    probs = softmax(x)
    return -np.log(probs[y])


def sigmoid(x, deriv=False):
    sigm = 1. / (1. + np.exp(-x))
    if deriv:
        return sigm * (1. - sigm)
    return sigm


def save_weights(parameters,fileName,iters):
    if(not os.path.exists(os.getcwd() + '/weights')):
        os.mkdir(os.getcwd() + '/weights')
    with open('weights/' + fileName + '_' + str(iters) + '.pkl', 'wb') as f:
        pickle.dump(parameters, f)
 

def load_weights(fileName):
    with open('weights/'+ fileName + '.pkl','rb') as f:
        weights = pickle.load(f)
        return weights


def print_sample(sample_ix, ix_to_char):
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    print (txt)


def smooth(loss, cur_loss, alpha):
    return loss * (1-alpha) + cur_loss * alpha


def map_data(data):
    data = open(data,'r').read()
    data = data.lower()
    chars = list(set(data))
    data_size, vocab_size = len(data), len(chars)
    print('There are %d total characters and %d unique characters in your data.' % (data_size, vocab_size))

    char_to_ix = { ch:i for i,ch in enumerate(sorted(chars)) }
    ix_to_char = { i:ch for i,ch in enumerate(sorted(chars)) }

    return vocab_size, char_to_ix, ix_to_char


def generate_training_set(fileName):
    print(fileName)
    with open(fileName) as f:
        examples = f.readlines()
        examples = [x.lower().strip() for x in examples]
        return examples


def gen_data(fileName):
    with open('./data/' + fileName) as f:
        
        # generate mappings
        data = f.read()
        data = data.lower()
        chars = list(set(data))
        data_size, vocab_size = len(data), len(chars)

        print('There are %d total characters and %d unique characters in your data.' % (data_size, vocab_size))

        char_to_ix = { ch:i for i,ch in enumerate(sorted(chars)) }
        ix_to_char = { i:ch for i,ch in enumerate(sorted(chars)) }

        # generate dataset
        examples = data.split('\n')
        examples = [x for x in examples]

        X = []
        Y = []
        
        for i in range(len(examples)):
            X.append(([None] + [char_to_ix[ch] for ch in examples[i]]))
            Y.append((X[i][1:] + [char_to_ix["\n"]]))

        return X, Y, vocab_size, char_to_ix, ix_to_char


def gen_data_words(fileName):
    with open(fileName) as f:
        
        dataset = []
        flatten = []
        # generate mappings
        data = f.read()
        data = data.split('\n')
        j=0
        for i in range(len(data)):
            d = data[i].split(' ')
            if len(d) > 4:
                d[0] = '<START>'
                d[len(d)-1] = '<END>'
                dataset.append(d)
                
                for w in dataset[j]:
                    flatten.append(w)
               
                j += 1
         
        words = list(set(flatten))
        vocab_size = len(words)
        
        char_to_ix = { ch:i for i,ch in enumerate(sorted(words)) }
        ix_to_char = { i:ch for i,ch in enumerate(sorted(words)) }
              
        
        X = []
        Y = []
        
        for i in range(len(dataset)):
            x  = [char_to_ix[ch] for ch in dataset[i]]
            X.append(x[:len(x)-1])
            Y.append((x[1:]))

        return dataset, X, Y, vocab_size, char_to_ix, ix_to_char
