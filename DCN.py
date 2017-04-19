from __future__ import print_function
import numpy as np
import tensorflow as tf
import random
import sys
import os
import time
import json
import cPickle as pickle
from config import Config
from utils import DataUtils, vectorizing
from evaluate import evaluate


class DCN:

    def __init__(self, config):
        self.cfg = config
        self.load_data()
        self.add_placeholders()
        outputs = self.add_model()
        self.add_loss_op(outputs)
        self.add_train_op()
        self.init_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

    def load_data(self):
        du = DataUtils(self.cfg)
        self.train = du.train
        self.dev = du.dev

    def add_placeholders(self):
        self.doc = tf.placeholder(tf.float32, [self.cfg.batch_size, None, self.cfg.emb_size])
        self.qus = tf.placeholder(tf.float32, [self.cfg.batch_size, None, self.cfg.emb_size])
        self.doc_len = tf.placeholder(tf.int32, [self.cfg.batch_size])
        self.qus_len = tf.placeholder(tf.int32, [self.cfg.batch_size])
        self.ans_s = tf.placeholder(tf.int32, [self.cfg.batch_size])
        self.ans_e = tf.placeholder(tf.int32, [self.cfg.batch_size])
        self.dp = tf.placeholder(tf.float32)

    def create_feed_dict(self, doc, qus, doc_len, qus_len, dp=1., ans_s=None, ans_e=None):
        feed_dict = {}
        feed_dict[self.doc] = doc
        feed_dict[self.qus] = qus
        feed_dict[self.doc_len] = doc_len
        feed_dict[self.qus_len] = qus_len
        feed_dict[self.dp] = dp
        if ans_s is not None and ans_e is not None:
            feed_dict[self.ans_s] = ans_s
            feed_dict[self.ans_e] = ans_e
        return feed_dict

    def add_model(self):
        
        with tf.variable_scope('Encoder') as scope:
            enc_cell = tf.contrib.rnn.core_rnn_cell.LSTMCell(self.cfg.hid_size)
            D, _ = tf.nn.dynamic_rnn(enc_cell, self.doc, self.doc_len, dtype=tf.float32)
            scope.reuse_variables()
            Q_prime, _ = tf.nn.dynamic_rnn(enc_cell, self.qus, self.qus_len, dtype=tf.float32)
        
        with tf.variable_scope('Coattention') as scope:
            W_Q = tf.get_variable('W_Q', (self.cfg.hid_size, self.cfg.hid_size))
            b_Q = tf.get_variable('b_Q', (self.cfg.hid_size,))
            Q = tf.tanh(tf.matmul(tf.reshape(Q_prime, (-1, self.cfg.hid_size)), W_Q) + b_Q)
            Q = tf.reshape(Q, tf.shape(Q_prime))
            
            L = tf.matmul(D, Q, transpose_b=True)
            A_D = tf.nn.softmax(L)
            A_Q = tf.nn.softmax(tf.transpose(L, (0, 2, 1)))
            
            C_Q = tf.matmul(A_Q, D)
            C_D = tf.matmul(A_D, tf.concat([Q, C_Q], 2))
            DC = tf.concat([D, C_D], 2)
            
            fw_cell = tf.contrib.rnn.core_rnn_cell.LSTMCell(self.cfg.hid_size)
            bw_cell = tf.contrib.rnn.core_rnn_cell.LSTMCell(self.cfg.hid_size)
            U, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, DC, 
                    dtype=tf.float32, sequence_length=self.doc_len)
            U = tf.concat(U, 2)
        
        with tf.variable_scope('Decoder') as scope:
            shape = (self.cfg.batch_size, self.cfg.hid_size)
            h_i = tf.zeros(shape)
            cell = tf.zeros(shape)
            u_s = U[:, 0]
            u_e = U[:,-1]
            decoder_outputs = []
            for i in range(self.cfg.max_iter):
                if i > 0:
                    scope.reuse_variables()
                X = tf.concat((u_s, u_e), 1)
                h_i, cell = LSTM_decoder(X, h_i, cell, self.cfg.hid_size, self.dp)

                UT = tf.transpose(U, (1, 0, 2))
                U_flat = tf.reshape(U, (-1, 2*self.cfg.hid_size))
                
                with tf.variable_scope('HMN_start'):
                    alphas = tf.map_fn(lambda u_t: HMN(h_i, u_t, u_s, u_e, 
                        self.cfg.hid_size, self.cfg.maxout_pool_size, self.dp), UT)
                    alphas = tf.transpose(tf.squeeze(alphas))
                    _, alpha_max = tf.nn.top_k(alphas)
                    alpha_t = tf.map_fn(lambda ind: alpha_max[ind]+ind*tf.shape(U)[1], 
                            tf.range(0, tf.shape(U)[0]))
                    u_s = tf.gather_nd(U_flat, alpha_t)
                    
                with tf.variable_scope('HMN_end'):
                    betas = tf.map_fn(lambda u_t: HMN(h_i, u_t, u_s, u_e,
                        self.cfg.hid_size, self.cfg.maxout_pool_size, self.dp), UT)
                    betas = tf.transpose(tf.squeeze(betas))
                    _, beta_max = tf.nn.top_k(betas)
                    beta_t = tf.map_fn(lambda ind: beta_max[ind]+ind*tf.shape(U)[1],
                            tf.range(0, tf.shape(U)[0]))
                    u_e = tf.gather_nd(U_flat, beta_t)
                
                decoder_outputs.append((alphas, betas))
            self.prediction = (alpha_max, beta_max)
            return decoder_outputs

    def add_loss_op(self, outputs):
        loss = 0
        for i in range(self.cfg.max_iter):
            loss += tf.losses.sparse_softmax_cross_entropy(self.ans_s, outputs[i][0])
            loss += tf.losses.sparse_softmax_cross_entropy(self.ans_e, outputs[i][1])
        tvars = tf.trainable_variables()
        l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tvars if 'HMN' in var.name])
        self.loss = loss + self.cfg.weight_decay * l2_loss

    def add_train_op(self):
        optimizer = tf.train.AdamOptimizer(self.cfg.learning_rate)
        self.train_op = optimizer.minimize(self.loss)

    def run_epoch(self, session, save='./save', load=None):
        if load:
            ckpt = self.train.get_checkpoint_state(load)
            self.saver.restore(session, ckpt.model_checkpoint_path)
        else:
            session.run(self.init_op)
        if not os.path.exists(save):
            os.mkdir(save)
        with open(self.cfg.vocab_path) as f:
            vocab = pickle.load(f)
        start_time = time.time()
        best_val_loss = float('inf')
        best_val_epoch = 0
        for epoch in range(self.cfg.epochs):
            random.shuffle(self.train)
            loss = 0
            for step, batch in enumerate(self.train):
                random.shuffle(batch)
                dataset = vectorizing(batch, self.cfg.emb_size, vocab)
                doc = [data.context for data in dataset]
                qus = [data.question for data in dataset]
                doc_len = np.array([data.context_len for data in dataset], dtype=np.int32)
                qus_len = np.array([data.question_len for data in dataset], dtype=np.int32)
                ans_s = np.array([data.answer[0] for data in dataset], dtype=np.float32)
                ans_e = np.array([data.answer[1] for data in dataset], dtype=np.float32)
                feed_dict = self.create_feed_dict(doc, qus, doc_len, qus_len, 
                                                  self.cfg.dropout, ans_s, ans_e)
                batch_loss, _ = session.run([self.loss, self.train_op], feed_dict=feed_dict)
                loss += batch_loss
                sys.stdout.write('\rEpoch {:>3}, step {:>4}, time {:8.2f}, loss {:.4f}'.format(
                    epoch, step, time.time()-start_time, loss/(step+1)))
                sys.stdout.flush()
            val_loss, val_results = self.evaluation(session, vocab)
            print('\rValidation: time {:8.2f}, loss {:.4f}, F1: {:.2f}, EM: {:.2f}'.format(
                time.time()-start_time, val_loss, val_results['f1'], val_results['exact_match']))
            if best_val_loss > val_loss:
                best_val_loss = val_loss
                self.saver.save(session, os.path.join(save, self.cfg.save_name))
                best_val_epoch = epoch
            if epoch - best_val_epoch > self.cfg.early_stopping:
                break
            
    def evaluation(self, session, vocab):
        loss = []
        answers = {}
        for step, batch in enumerate(self.dev):
            dataset = vectorizing(batch, self.cfg.emb_size, vocab)
            doc = [data.context for data in dataset]
            qus = [data.question for data in dataset]
            doc_len = np.array([data.context_len for data in dataset], dtype=np.int32)
            qus_len = np.array([data.question_len for data in dataset], dtype=np.int32)
            ans_s = np.array([data.answer[0] for data in dataset], dtype=np.float32)
            ans_e = np.array([data.answer[1] for data in dataset], dtype=np.float32)
            feed_dict = self.create_feed_dict(doc, qus, doc_len, qus_len, 1., ans_s, ans_e)
            batch_loss, pred = session.run([self.loss, self.prediction], feed_dict=feed_dict)
            loss.append(batch_loss)
            for i, qap in enumerate(batch):
                answer = qap.context[int(pred[0][i]):int(pred[1][i])+1]
                if len(answer) == 0:
                    answers[qap.qid] = ''
                else:
                    answers[qap.qid] = reduce(lambda x, y: x+' '+y, answer)
        with open(self.cfg.dev_path) as f:
            dev_data = json.load(f)['data']
            results = evaluate(dev_data, answers)
        return sum(loss)/(step+1), results
            

def maxout(inputs, num_units=1, axis=None):
    """reference: 
        https://github.com/philipperemy/tensorflow-maxout/blob/master/maxout.py
    """
    shape = inputs.get_shape().as_list()
    if shape[0] is None:
        shape[0] = -1
    if axis is None:  # Assume that channel is the last dimension
        axis = -1
    num_channels = shape[axis]
    if num_channels % num_units:
        raise ValueError('number of features({}) is not '
                         'a multiple of num_units({})'.format(num_channels, num_units))
    shape[axis] = num_units
    shape += [num_channels // num_units]
    outputs = tf.reduce_max(tf.reshape(inputs, shape), -1, keep_dims=False)
    return tf.squeeze(outputs)

    
def HMN(h_i, u_t, u_s, u_e, l, p, dp):
    W_D = tf.get_variable('W_D', (5*l, l))
    r = tf.nn.tanh(tf.matmul(tf.concat((h_i, u_s, u_e), 1), W_D))
    W_1 = tf.get_variable('W_1', (3*l, l*p))
    b_1 = tf.get_variable('b_1', (l*p))
    mt_1 = maxout(tf.reshape(tf.matmul(tf.concat((u_t, r), 1), W_1) + b_1, (-1, l, p)))
    mt_1 = tf.nn.dropout(mt_1, dp)
    W_2 = tf.get_variable('W_2', (l, p*l))
    b_2 = tf.get_variable('b_2', (l*p))
    mt_2 = maxout(tf.reshape(tf.matmul(mt_1, W_2) + b_2, (-1, l, p)))
    mt_2 = tf.nn.dropout(mt_2, dp)
    W_3 = tf.get_variable('W_3', (2*l, p))
    b_3 = tf.get_variable('b_3', (p))
    hmn = maxout(tf.reshape(tf.matmul(tf.concat((mt_1, mt_2), 1), W_3) + b_3, (-1, 1, p)))
    return hmn


def LSTM_decoder(X, H, cell, hid_size, dp):
    # note that shape of X is (batch_size, 4 * hid_size)
    X = tf.nn.dropout(X, dp)
    XH = tf.concat((X, H), 1)

    W_I = tf.get_variable('W_I', (hid_size*5, hid_size))
    b_I = tf.get_variable('b_I', (hid_size))
    I = tf.nn.sigmoid(tf.matmul(XH, W_I) + b_I)

    W_f = tf.get_variable('W_f', (hid_size*5, hid_size))
    b_f = tf.get_variable('b_f', (hid_size))
    f = tf.nn.sigmoid(tf.matmul(XH, W_f) + b_f)

    W_o = tf.get_variable('W_o', (hid_size*5, hid_size))
    b_o = tf.get_variable('b_o', (hid_size))
    o = tf.nn.sigmoid(tf.matmul(XH, W_o) + b_o)

    W_c = tf.get_variable('W_c', (hid_size*5, hid_size))
    b_c = tf.get_variable('b_c', (hid_size))
    c_hat = tf.nn.tanh(tf.matmul(XH, W_c) + b_c)

    cell = f * cell + I * c_hat
    H = o * tf.nn.tanh(cell)

    return H, cell


if __name__ == '__main__':
    config = Config()
    os.environ['CUDA_VISIBLE_DEVICES'] = config.device
    dcn = DCN(config)
    # tf_cfg = tf.ConfigProto()
    # tf_cfg.gpu_options.allow_growth = True
    with tf.Session() as sess:
        dcn.run_epoch(sess)
    
