#!/usr/bin/env python
# coding: utf-8

# ### import

# In[1]:


import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import pickle as pkl
import os
import math
import time
from datetime import timedelta
import torch
import torch.nn.functional as F
import torch.nn as nn

torch.cuda.set_device(0)


# ## Data preprocess

# ### split trian dev test

# see task1

# ### padding

# In[2]:


df_data = pd.read_csv('./raw_data/train.tsv', sep='\t')
df_data.head()


# In[3]:


df_data['Phrase_len'] = df_data['Phrase'].apply(lambda x: len(x))


# In[4]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[5]:


sns.countplot(df_data['Phrase_len'])
plt.xticks([])
plt.show()


# In[6]:


sns.distplot(df_data['Phrase_len']) # plt.hist and sns.kdeplot
plt.show()


# In[14]:


pad_size = sorted(df_data['Phrase_len'])[round(len(df_data)*0.95)]
pad_size


# ### build my vocab and dataset

# In[7]:


from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号

def tokenizer_lemma(line):
    '''
    diff to task 1
    '''
    def get_wordnet_pos(tag):
        '''
        get the part-of-speech
        '''
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN    
    wnl = WordNetLemmatizer()
    tokens = word_tokenize(line.lower())              # 分词,同时大写换小写
    tagged_sent = pos_tag(tokens, tagset='universal')     # 词性标注
    tokens_lemma = [wnl.lemmatize(word, pos=get_wordnet_pos(tag)) for word, tag in tagged_sent]
    return tokens_lemma
    
def build_vocab(data_path, vocab_path):
    data = [_.strip().split('\t')[0] for _ in open(data_path, 'r', encoding='utf-8').readlines()]
    word_cnt = dict()
    for sentence in tqdm(data):
        for token in tokenizer_lemma(sentence):
            word_cnt[token] = word_cnt.get(token, 0) + 1
    word_cnt = sorted(word_cnt.items(), key=lambda x:x[0], reverse=True)
    print(len(word_cnt))
    vocab = {_[0]: idx for idx, _ in enumerate(word_cnt)}
    vocab.update({UNK: len(vocab), PAD: len(vocab) + 1})
    pkl.dump(vocab, open(vocab_path, 'wb'))
    print("vocab build successed, size : %d" %len(vocab))
    return vocab

def build_dataset(config):
    '''
    变成[([],y),([],y),([],y),([],y)]
    '''
    if os.path.exists(config.vocab_path):
        vocab = pkl.load(open(config.vocab_path, 'rb'))
    else:
        vocab = build_vocab(config.train_path, config.vocab_path)
    config.n_vocab = len(vocab)
    print("vocab loaded sucessed, size : %d " %len(vocab))
    def load_data(file_path, output_path, pad_size=config.pad_size):
        if os.path.exists(output_path):
            data = pkl.load(open(output_path, 'rb'))
            print("%s loaded success, size: %d" %(output_path, len(data)))
            return data
        data = open(file_path, 'r', encoding='utf-8').readlines()
        all_data = list()
        for line in tqdm(data):
            try:
                x, y = line.strip().split('\t')
            except:
                print(line)
            tokens = tokenizer_lemma(x)
            if len(tokens) < pad_size:
                tokens.extend([PAD] * (pad_size - len(tokens)))
            else:
                tokens = tokens[:pad_size]
            token_map = [vocab.get(token, vocab.get(UNK)) for token in tokens] # diff to one-hot
            all_data.append((token_map, int(y)))
        pkl.dump(all_data, open(output_path, 'wb'))
        print("%s loaded success, size: %d" %(file_path, len(all_data)))
        return all_data
    test_set = load_data(config.test_path, config.test_set_path)
    dev_set = load_data(config.dev_path, config.dev_set_path)
    train_set = load_data(config.train_path, config.train_set_path)
    return train_set, dev_set, test_set


# ### glove process

# In[3]:


glove_path = r'D:\workspace\pretrained_models\glove\standford\glove.6B.300d.txt'
with open(glove_path, 'r', encoding='utf-8') as f:
    vocab = {}
    data = list()
    for i,line in enumerate(f.readlines()):
        w2vlist = line.split()
        word, vector = w2vlist[0], w2vlist[1:]
        vocab[word] = i
        data.append(vector)
    data.append([0]*len(data[0]))
    data.append([0]*len(data[0]))
    vocab.update({UNK: len(vocab), PAD: len(vocab) + 1})
    data = np.array(data,float)
pkl.dump(vocab, open(r'D:\workspace\pretrained_models\glove\standford\vocab.pkl', 'wb'))
pkl.dump(data, open(r'D:\workspace\pretrained_models\glove\standford\glove.6B.300d.pkl', 'wb'))


# ### word2vec process

# In[39]:


model_path = r'/home2/rzhao/pretrained_models/word2vec/GoogleNews/word2vec-google-news-300.model'
# vec_path = r'/home2/rzhao/pretrained_models/word2vec/GoogleNews/word2vec-google-news-300.model.vectors.npy'
# word_map_path = r'/home2/rzhao/pretrained_models/word2vec/GoogleNews/word2vec-combined-ms-256.json'


# In[40]:


from gensim.models import KeyedVectors
w2v = KeyedVectors.load(model_path)
# vec = np.load(vec_path)


# In[41]:


os.path.join(os.path.dirname(model_path), 'vocab.pkl')
os.path.join(os.path.dirname(vec_path), 'word2vec-google-news-300.pkl')


# In[ ]:


w2v.key_to_index['<PAD>'] = w2v.get_index('PAD')
w2v.key_to_index.pop('PAD')


# In[45]:


pkl.dump(w2v.key_to_index, open(r'/home2/rzhao/pretrained_models/word2vec/GoogleNews/vocab.pkl', 'wb'))
pkl.dump(w2v.vectors, open(r'/home2/rzhao/pretrained_models/word2vec/GoogleNews/word2vec-google-news-300.pkl', 'wb'))


# ## unit

# ### config

# In[2]:


class Config(object):
    def __init__(self, embed_path=None, vocab_path=None):
        self.train_path = r'./data_set/train.txt'
        self.dev_path = r'./data_set/dev.txt'
        self.test_path = r'./data_set/test.txt'
        self.vocab_path = vocab_path if vocab_path else r'./data_set/vocab.pkl'
        self.train_set_path = r'./data_set/train.pkl'
        self.dev_set_path = r'./data_set/dev.pkl'
        self.test_set_path = r'./data_set/test.pkl'
        self.class_list = [x.strip() for x in open(r'./data_set/labels.txt', encoding='utf-8').readlines()]
        self.label_num = len(self.class_list)
        self.n_vocab = 0    # assign after dataset built
        self.batch_size = 32
        self.epoch = 20
        self.learning_rate = 1e-3
        self.drop_out = 0.3
        self.log_step = 1000
        self.pad_size = 128
        self.embedding_pretrained = torch.FloatTensor(pkl.load(open(embed_path, 'rb'))) if embed_path is not None else None
        self.embedding_len = self.embedding_pretrained.size(1) if self.embedding_pretrained is not None else 300

        self.rnn_hidden = 128
        self.rnn_layer = 2
        self.bi_rnn = False
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


# In[4]:


config = Config(embed_path = r'/home2/rzhao/pretrained_models/glove/standford/glove.6B.50d.pkl',
                vocab_path = r'/home2/rzhao/pretrained_models/glove/standford/vocab.pkl')


# ### datasetIterater

# In[3]:


class DatasetIterater(object):
    '''
    return the batch index of dataset
    '''
    def __init__(self, data_set, batch_size, device):
        self.device = device
        self.data_set = data_set
        self.batch_size = batch_size
        self.n_batches = len(data_set) // batch_size
        self.residue = True if len(data_set) % self.n_batches != 0 else False
        self.index = 0
        
    def to_tensor(self, raw_batch):
        x = torch.LongTensor([_[0] for _ in raw_batch]).to(self.device)
        y = torch.LongTensor([_[1] for _ in raw_batch]).to(self.device)
        return x,y
        
    def __next__(self):
        if self.index == self.n_batches and self.residue:
            raw_batch = self.data_set[self.index * self.batch_size: len(self.data_set)]
            batch = self.to_tensor(raw_batch)
            self.index += 1
            return batch
        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            raw_batch = self.data_set[self.index * self.batch_size: (self.index+1) * self.batch_size]
            batch = self.to_tensor(raw_batch)
            self.index += 1
            return batch
           
    def __iter__(self):
        return self
    
    def __len__(self):
        return self.n_batches+1 if self.residue else self.n_batches


# Test

# In[6]:


config = Config()
train_set, dev_set, test_set = build_dataset(config)
data_iter = DatasetIterater(test_set, config.batch_size, config.device)
next(data_iter)


# ## model

# ### RNN

# In[4]:


class Text_RNN(nn.Module):
    def __init__(self, config):
        super(Text_RNN, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False, padding_idx=config.n_vocab - 1)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embedding_len, padding_idx=config.n_vocab - 1)
        self.rnn = nn.RNN(config.embedding_len, config.rnn_hidden, config.rnn_layer,
                         bidirectional=config.bi_rnn, batch_first=True, dropout=config.drop_out)
        self.fc = nn.Linear(config.rnn_hidden * (int(config.bi_rnn) + 1), config.label_num)
        
    def forward(self, x):
        out = self.embedding(x)
        out, hn = self.rnn(out) # 
        out = self.fc(out[:, -1, :]) # out[:, -1, :] equal to hn[-1, :, :]
        return out


# In[ ]:


config = Config()
train_set, dev_set, test_set = build_dataset(config)
test_iter = DatasetIterater(test_set, config.batch_size, config.device)
model = Text_RNN(config).to(config.device)
model(next(test_iter)[0])


# In[16]:


config = Config(embed_path = r'/home2/rzhao/pretrained_models/glove/standford/glove.6B.50d.pkl',
                vocab_path = r'/home2/rzhao/pretrained_models/glove/standford/vocab.pkl')
train_set, dev_set, test_set = build_dataset(config)
test_iter = DatasetIterater(test_set, config.batch_size, config.device)
model = Text_RNN(config).to(config.device)
model(next(test_iter)[0])


# ### CNN

# ## train and evl

# In[5]:


def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass

def train(model, config, train_set, dev_set, test_set):
    train_iter = DatasetIterater(train_set, config.batch_size, config.device)
    dev_iter = DatasetIterater(dev_set, config.batch_size, config.device)
    test_iter = DatasetIterater(test_set, config.batch_size, config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr = config.learning_rate)
    model.train()
    start_time = time.time()
    total_iter = 0
    best_dev_acc = 0
    for epoch in range(config.epoch):
        print('Epoch [{}/{}]'.format(epoch + 1, config.epoch))
        for i, (x, y_true) in enumerate(train_iter):
            out = model(x)
            model.zero_grad()
            loss = F.cross_entropy(out, y_true)
            loss.backward()
            optimizer.step()
            total_iter += 1
            if total_iter % config.log_step == 0  or i+1 == len(train_iter):
                acc_train, loss_train = evalute(model, config, DatasetIterater(train_set, config.batch_size, config.device))
                acc_dev, loss_dev = evalute(model, config, dev_iter)
                model.train()
                improve = ""
                if acc_dev > best_dev_acc:
                    best_dev_acc = acc_dev
                    improve = "*"
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5}{6}'
                print(msg.format(total_iter, loss_train, acc_train, loss_dev, acc_dev, time_dif, improve))
    acc_test, loss_test = evalute(model, config, test_iter)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(loss_test, acc_test))

def evalute(model, config, data_iter):
    model.eval()
    loss_total = 0
    labels_predict = np.arange(0)
    labels_true = np.arange(0)
    with torch.no_grad():
        for x, y_true in data_iter:
            out = model(x)
            loss = F.cross_entropy(out, y_true)
            loss_total += loss
            labels_predict = np.concatenate((labels_predict, out.data.cpu().numpy().argmax(axis=1)), axis=0)
            labels_true = np.concatenate((labels_true, y_true.data.cpu().numpy()), axis=0)
        try:
            assert len(labels_true) == labels_predict.size
        except AssertionError as e:
            print(len(x))
            print(out.shape)
            print("labels_true:", len(labels_true))
            print("labels_predict:",labels_predict.size)
            raise AssertionError
    acc = (labels_true == labels_predict).mean()
    loss = loss_total.data.cpu().numpy()/len(data_iter)
    return acc, loss_total


# ## run

# In[51]:


config = Config()
train_set, dev_set, test_set = build_dataset(config)
### customized hyper param
config.learning_rate = 1e-5 # lr = 1e-3 导致模型震荡，acc上不去
config.log_step = 1000
config.drop_out = 0.3
###
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True  # 保证每次结果一样

model = Text_RNN(config).to(config.device)
init_network(model)

train(model, config, train_set, dev_set, test_set)


# In[35]:


config = Config(embed_path = r'/home2/rzhao/pretrained_models/glove/standford/glove.6B.50d.pkl',
                vocab_path = r'/home2/rzhao/pretrained_models/glove/standford/vocab.pkl')
train_set, dev_set, test_set = build_dataset(config)
### customized hyper param
config.batch_size = 32
config.learning_rate = 3e-2
config.log_step = 1000
config.drop_out = 0.3
###
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True  # 保证每次结果一样

model = Text_RNN(config).to(config.device)
init_network(model)
train(model, config, train_set, dev_set, test_set)


# In[ ]:


config = Config(embed_path = r'/home2/rzhao/pretrained_models/word2vec/GoogleNews/word2vec-google-news-300.pkl',
                vocab_path = r'/home2/rzhao/pretrained_models/word2vec/GoogleNews/vocab.pkl')
train_set, dev_set, test_set = build_dataset(config)
### customized hyper param
config.batch_size = 32
config.learning_rate = 3e-2
config.log_step = 1000
config.drop_out = 0.3
###
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True  # 保证每次结果一样

model = Text_RNN(config).to(config.device)
init_network(model)
train(model, config, train_set, dev_set, test_set)


# ## Test code

# In[36]:


import  gc
del model
gc.collect()


# In[40]:


def empty_cache(var):
    del var
    gc.collect()
    torch.cuda.empty_cache()


# 似乎占用的显存只能通过重启kernel或者kill进程释放。
