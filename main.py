from model import mainModel
from dataLoader import data_generator
from torch import optim
from datetime import datetime, timedelta
import torch
import torch.nn as nn
from tqdm import tqdm

batch_size = 1
input_size = 128
hidden_size = 128
hc = 8
t = [1,2,3,4,5]
lr = 3e-4
epochs = 5
M = 0.5


data = data_generator(batch_size=batch_size,shuffle=True)
model = mainModel(data.vocab_size,input_size,hidden_size,hc,t)
optimizer = optim.Adam(params=model.parameters(),lr=lr)
loss_train = []
last_best_loss = None
current_time = datetime.utcnow()

def debug(i, epoch,loss,model):
    global  loss_train
    global last_best_loss
    global current_time

    this_loss = loss.item()
    loss_train.append(this_loss)
    loss_train = loss_train[-20:]
    new_current_time = datetime.utcnow()
    time_elapsed = str(new_current_time - current_time)
    current_time = new_current_time

    print("epoch-{}-step-{}:last_best_loss={},this_loss={}".format(epoch,i,last_best_loss,this_loss))
    try:
        train_loss = sum(loss_train)/len(loss_train)
        if last_best_loss is None or last_best_loss > train_loss:
            print("Loss improved from {} to {}".format(last_best_loss, train_loss))
            save_loc = "result/best_model.pkl"
            torch.save(model.state_dict(),save_loc)
            last_best_loss = train_loss
    except Exception as e:
        print("无法保存模型，因为：",e)

#加载模型

# model.load_state_dict(torch.load("result/best_model.pkl"))


for epoch in range(epochs):
    for i in range(data.len//batch_size):
        q, p, n = data.generate_next_batch()
        q_p = model(q, p)
        q_n = model(q, n)
        if (M-q_p+q_n).item() <0:
            pass
        else:
            loss = M-q_p+q_n
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if i % 100 ==0:
            debug(i, epoch, loss,model)
