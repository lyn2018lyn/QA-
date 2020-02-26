import pandas as pd
import pickle
import os
import torch
from tqdm import tqdm
import random
import torch
import numpy as np


class data_generator():
    def __init__(self, batch_size, shuffle=True):
        print("加载数据中·······")
        self.data = pd.read_csv("./data/train_candidates/train_candidates.txt") #question_id  pos_ans_id  neg_ans_id
        self.data = self.data.values.tolist()
        self.index = 0 #指示当前取到哪一个了
        self.len = len(self.data)
        self.batch_size = batch_size
        self.shuffle = shuffle
        if self.shuffle: self.shuffle_data
        self.text2id = dataprocess()
        self.id2Q = self.text2id.id2Q
        self.id2A = self.text2id.id2A
        self.vocab_size  = self.text2id.vocab_size
        print("数据加载完毕！")
    def generate_next_index(self):
        if self.index + self.batch_size > self.len:
            self.index = 0
        start = self.index
        end = self.index + self.batch_size
        self.index = end
        return self.data[start:end]

    def generate_next_batch(self):
        items = self.generate_next_index()
        q = []
        p = []
        n = []
        QA_len = []

        for item  in items:
            a = self.id2Q[item[0]]
            b = self.id2A[item[1]]
            c = self.id2A[item[2]]
            q.append(a)
            p.append(b)
            n.append(c)
            QA_len.append([len(a),len(b),len(c)])
        #找出每组的最长句子
        QA_len = np.array(QA_len).max(axis=0)
        q = [self.padding(item,QA_len[0]) for item in q]
        p = [self.padding(item,QA_len[1]) for item in p]
        n = [self.padding(item,QA_len[2]) for item in n]
        return torch.LongTensor(q).view(self.batch_size,1,-1),\
               torch.LongTensor(p).view(self.batch_size,1,-1),\
               torch.LongTensor(n).view(self.batch_size,1,-1)

    def padding(self,x,max_len):
        padd = max_len - len(x)
        if padd <=0:
            return x
        else:
            x.extend([0]*padd)
            return x

    def shuffle_data(self):
        self.data = random.sample(self.data, self.len)



class dataprocess:
    """
    1.把question和answer用字的id表示
    2.生成字到id的字典
    3.生成id到question和answer的字典
    """
    def __init__(self):
        self.question = pd.read_csv("./data/question/question.csv")
        self.answer = pd.read_csv("./data/answer/answer.csv")
        self.char2id={}
        if os.path.exists("./data/char2id.pkl"):
            f = open("./data/char2id.pkl", "rb")
            self.char2id = pickle.load(f)
            f.close()
        else:
            self.char2id = self.Char2id()

        if os.path.exists("./data/id2A.pkl") and os.path.exists("./data/id2Q.pkl"):
            f = open("./data/id2Q.pkl", "rb")
            self.id2Q = pickle.load(f)
            f = open("./data/id2A.pkl", "rb")
            self.id2A = pickle.load(f)
            f.close()
        else:
            self.id2Q, self.id2A = self.QA2id()
        self.vocab_size = len(self.char2id)

    def QA2id(self):
        print("开始数据转换")
        id2Q = {}
        id2A = {}
        for item in tqdm(self.question.values.tolist()):
            text = self.text2id(item[1])
            id2Q[item[0]] = text
        f = open("./data/id2Q.pkl","wb")
        pickle.dump(id2Q,f)

        for item in tqdm(self.answer.values.tolist()):
            text = self.text2id(item[2])
            id2A[item[0]] = text
        f = open("./data/id2A.pkl","wb")
        pickle.dump(id2A,f)
        f.close()
        return id2Q, id2A


    def text2id(self,text):
        temp = []
        for item in list(text.strip()):
            temp.append(self.char2id.get(item,0))
        return temp


    def Char2id(self):
        char2id = {"<unk>":0}
        for item in tqdm(self.question.values.tolist()):
            for char in list(item[1].strip()):
                if char not in char2id and char != "\n":
                    char2id[char] = len(char2id)

        for item in tqdm(self.answer.values.tolist()):
            for char in list(item[2].strip()):
                if char not in char2id and char != "\n":
                    char2id[char] = len(char2id)

        f = open("./data/char2id.pkl","wb")
        pickle.dump(char2id,f)
        f.close()
        return char2id


