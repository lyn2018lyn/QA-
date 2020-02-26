import torch
import torch.nn as nn
import copy

"""
注释说明：
文字向量长度dim
GRU输出的向量长度H
短接之后输出长度为E
每次只处理一对QA，即batch——size=1
每个尺度的filter重复次数：hc，即channel数
filter尺度矩阵t
句子长度max_len
每个卷积核的输出长度  L

则同一个尺度的filter在一个句子上的输出为hc*L
"""

class encoder(nn.Module):
    """
    输入：Q或者A的embedding表示
    输出：经过biGRU后的输出
    """
    def __init__(self,input_size ,hidden_size):
        super(encoder, self).__init__()
        self.biGRU = nn.GRU(hidden_size,
                            hidden_size,
                            bidirectional=True,
                            batch_first=True)
    def forward(self, X):
        X = X.squeeze(1) #GRU输入只能是3维
        out, _ = self.biGRU(X)
        return torch.cat((out,X), dim=-1)



class conv(nn.Module):
    """
    输入：Q 或者 A的biGRU结果
    输出：经过指定个filter之后的卷积输出O
    """
    def __init__(self, E, hc=5, t=[1,2,3]):
        """
        :param hc: 每个尺度的filter重复多少次
        :param t: filter大小列表
        :param E:短连接之后的向量长度
        """
        super(conv, self).__init__()
        self.hc = hc
        self.conv_list = []
        for i in t:
            self.conv_list.append(nn.Conv1d(1,self.hc,kernel_size=(i, E)))
    def forward(self, X):
        X = X.unsqueeze(1) #增加一个channel维度, 1*1*max_len*E
        temp = list()
        for conv_item in  self.conv_list:
            conv_item
            out = conv_item.forward(X) # 1*hc*L*1
            out = out.squeeze(-1) # 1*hc*L
            out = out.transpose(-2,-1) # 1*L*hc 和书上的保持一致即O-si
            temp.append(out)
        return temp



class att(nn.Module):
    """
    :param Q: question的卷积结果
    :param A: answer的卷积结果
    :return: att-q和att-a
    """
    def __init__(self, hc, t= [1,2,3]):
        super(att, self).__init__()
        self.w_list = []
        self.hc = hc
        for _ in t:
            self.w_list.append(torch.FloatTensor(hc,hc))


    def forward(self, Q, A):
        batch_size = Q[0].shape[0]
        q_result = torch.zeros(batch_size,1,self.hc)
        a_result = torch.zeros(batch_size,1,self.hc)
        for q,u,a in zip(Q, self.w_list, A): #每一个的维度是1*l*hc
            I = nn.Sigmoid()(q.matmul(u).matmul(a.transpose(-1,-2))) # 1*l_q*l_a
            temp = nn.Softmax(-1)(I).max(-1,keepdim=True)[0] # 1*l_q*1
            temp = temp.transpose(-1,-2).matmul(q) # 1*1*_lq X 1*l_q*hc = 1*1*hc
            q_result = torch.cat((q_result,temp),dim=-2) #1*t*hc

            temp = nn.Softmax(-2)(I).max(-2, keepdim=True)[0] # 1*1*l_a
            temp = temp.matmul(a)
            a_result = torch.cat((a_result,temp),dim=-2)
        q_result = q_result.max(-2, keepdim=False)[0]  # 1*hc
        a_result = a_result.max(-2,keepdim=False)[0] # 1*hc
        return q_result, a_result


class myModel(nn.Module):
    def __init__(self, encoderQ, encoderA, convA, convQ, att):
        """
        :param encoderQ:
        :param encoderA:
        :param convA: 输出A的卷积输出t*l*hc  t是指filter的尺度数量，l是每次卷积后的长度，hc是指每个尺度的卷积重复次数
        :param convQ:
        :param att:
        :param hc:
        :param st:
        """
        super(myModel, self).__init__()
        self.encoderQ = encoderQ
        self.encoderA = encoderA
        self.convA = convA
        self.convQ = convQ
        self.att = att

    def forward(self, Q, A):
        Q = self.encoderQ(Q) #batch_size*1*56*128
        A = self.encoderA(A)
        Q = self.convQ(Q)  # batch_size*hc*l_filter
        A = self.convA(A)
        Q,A = self.att(Q,A)
        #计算相似度
        # return torch.cosine_similarity(Q,A)
        return torch.mean(torch.cosine_similarity(Q,A))


class mainModel(nn.Module):
    def __init__(self, vocab_size, input_size, hidden_size, hc, t=[1,2,3]):
        super(mainModel, self).__init__()
        self.encoderQ = encoder(input_size, hidden_size)
        self.encoderA = encoder(input_size, hidden_size)
        self.convA = conv(input_size+2*hidden_size, hc ,t)
        self.convQ = conv(input_size+2*hidden_size, hc, t)
        self.att = att(hc, t)
        self.mymodel = myModel(self.encoderQ,self.encoderA,self.convA,self.convQ,self.att)
        self.embedding = nn.Embedding(vocab_size, input_size)

    def forward(self, Q, A):
        Q = self.embedding(Q)
        A = self.embedding(A)
        return self.mymodel(Q,A)

if __name__ == "__main__":
    Q = torch.LongTensor([[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9]])
    A = torch.LongTensor([[1,2,3,4,5,6,7],[1,2,3,4,5,6,7],[1,2,3,4,5,6,7],[1,2,3,4,5,6,7]])
    test = mainModel(10,250,250,5)
    a = test(Q,A)
    print(a)