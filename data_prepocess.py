from __future__ import print_function
import json
from config import Config
import numpy as np
import cPickle as pickle
import sys


def preprocess(train_data, dev_data, glove_path):
    glove = {}
    with open(glove_path, 'rb') as f:
        for line in f.readlines():
            line = line.strip().split()
            if len(line) == 301:
                glove[line[0]] = reduce(lambda x, y: x+' '+y, line[1:])
    f = open(train_data, 'r')
    train = json.load(f)
    f.close()
    f = open(dev_data, 'r')
    dev = json.load(f)
    f.close()
    vocab = {}
    vocab = parsing(dev, vocab, glove)
    vocab = parsing(train, vocab, glove)
    f = open('./data/vocab.pkl', 'wb')
    pickle.dump(vocab, f)
    f.close()
    

def parsing(dataset, vocab, glove):
    dataset = dataset['data']

    def get_vec(word):
        if word in glove:
            vocab[word] = glove[word]
        elif word.lower() in glove:
            vocab[word.lower()] = glove[word.lower()]

    for article in dataset:
        for k, v in article.items():
            if k == 'title':
                title = v.strip().split('_')
                for word in title:
                    get_vec(word)
        for paragraph in article['paragraphs']:
            for k, v in paragraph.items():
                if k == 'context':
                    context = v.strip().split()
                    for word in context:
                        get_vec(word)
                elif k == 'qas':
                    for itm in v:
                        for qk, qv in itm.items():
                            if qk == 'id':
                                pass
                            elif qk == 'question':
                                question = qv.strip().split()
                                for word in question:
                                    get_vec(word)
                            elif qk == 'answers':
                                for answers in qv:
                                    for ak, av in answers.items():
                                        if ak == 'text':
                                            for word in av.strip().split():
                                                get_vec(word)
    return vocab


if __name__ == '__main__':
    preprocess('./data/train-v1.1.json', './data/dev-v1.1.json', sys.argv[1])
