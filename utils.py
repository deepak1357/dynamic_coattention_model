from __future__ import print_function
import numpy as np
import json
from config import Config
import re


class QAPair:
    def __init__(self, qid, context=[], question=[], answer=None):
        self.qid = qid
        self.context = context
        self.question = question
        self.answer = answer
        self.context_len = len(context)
        self.question_len = len(question)

        
def parsing(dataset, max_ctx_len=600):
    """helper function to parse dataset.
    """
    QAdata = []
    for article in dataset['data']:
        for paragraph in article['paragraphs']:
            str_context = paragraph['context']
            context = paragraph['context'].strip().split()
            context = list(map(lambda x: re.sub(r'\(|\)|\.|\?|\,|\!|\"', '', x), context))
            for question in paragraph['qas']:
                qid = question['id']
                qap = QAPair(qid)
                qes = question['question'].strip().split()
                qes = list(map(lambda x: re.sub(r'\(|\)|\.|\?|\,|\!|\"', '', x), qes))
                qap.context = context
                qap.question = qes
                qap.context_len = len(context)
                qap.question_len = len(qes)
                for ans in question['answers']:
                    sp = int(ans['answer_start'])
                    cnt = 0
                    for i, word in enumerate(paragraph['context'].strip().split()):
                        if cnt + i >= sp:
                            break
                        cnt += len(word) 
                    ans_len = len(ans['text'].strip().split())
                    tmp = reduce(lambda x, y: x+' '+y, context[i: i+ans_len])
                    # qap.answer = (sp, sp+len(ans['text']))
                    qap.answer = (min(max_ctx_len, i), min(max_ctx_len, i+ans_len-1))
                QAdata.append(qap)
    return QAdata


class DataUtils:

    def __init__(self, config):
        self.cfg = config
        train = self.load_data(self.cfg.train_path)
        dev = self.load_data(self.cfg.dev_path)
        self.train = self.batching(train)
        self.dev = self.batching(dev)

    def load_data(self, data_path):
        f = open(data_path, 'r')
        dataset = json.load(f)
        f.close()
        QAdata = parsing(dataset, self.cfg.max_ctx_len)
        QAdata.sort(key=lambda x:x.context_len)
        return QAdata

    def batching(self, dataset):
        batch_size = self.cfg.batch_size
        batches = []
        for i in range(len(dataset)//batch_size):
            batches.append(dataset[i*batch_size:(i+1)*batch_size])
        # if len(dataset) > (i+1)*batch_size:
        #     batches.append(dataset[(i+1)*batch_size+1:])
        return batches
   

def vectorizing(dataset, emb_size, vocab, max_ctx_len=600):
    data_vec = []
    max_context_len = min(max_ctx_len, max([x.context_len for x in dataset]))
    max_question_len = max([x.question_len for x in dataset])
    for data in dataset:
        qap = QAPair(qid=data.qid)
        qap.answer = data.answer
        qap.context_len = data.context_len
        qap.question_len = data.question_len
        context = []
        question = []
        for i in range(max_context_len):
            if i < len(data.context) and data.context[i] in vocab:
                vec = np.array(map(float, vocab[data.context[i]].strip().split()))
            else:
                vec = np.zeros(emb_size)
            context.append(vec)
        context.append(vec)
        qap.context = np.array(context)
        for i in range(max_question_len):
            if i < len(data.question) and data.question[i] in vocab:
                vec = np.array(map(float, vocab[data.question[i]].strip().split()))
            else:
                vec = np.zeros(emb_size)
            question.append(vec)
        question.append(vec)
        qap.question = np.array(question)
        data_vec.append(qap)
    return data_vec


if __name__ == "__main__":
    config = Config()
    du = DataUtils(config)
    """
    import cPickle as pickle
    with open(config.vocab_path) as f:
        vocab = pickle.load(f)
    for batch in du.dev:
        data = vectorizing(batch, config.emb_size, vocab)
        for itm in data:
            print('ID: ', itm.qid)
            #print(itm.context_len)
            #print(itm.question_len)
            print('Question:', itm.question.shape)
            print('Document:', itm.context.shape)
            print('==================')
    """
