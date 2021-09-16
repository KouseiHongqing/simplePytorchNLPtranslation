'''
函数说明: 
Author: hongqing
Date: 2021-09-03 11:31:51
LastEditTime: 2021-09-16 17:08:23
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from numpy.core.fromnumeric import mean, std
%matplotlib inline
np.random.seed(1)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import dataloader,Dataset
from torch.nn.utils.rnn import pack_padded_sequence as pack, pad_packed_sequence as unpack
from tqdm import tqdm
from nmt_utils import *
m = 10000
dataset, human_vocab, machine_vocab, inv_machine_vocab = load_dataset(m)
Tx = 30 # 这里是假设在人类日期格式中最多有30个字符，如果超过30，那么会被截断。
Ty = 10 # 电脑格式"YYYY-MM-DD"中字符数量是固定的，就是10个。
X, Y, Xoh, Yoh = preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty)

Xtest = Xoh[:100,:,:]
Ytest = Yoh[:100,:,:]
print("X.shape:", X.shape)
print("Y.shape:", Y.shape)
print("Xoh.shape:", Xoh.shape)
print("Yoh.shape:", Yoh.shape)

index = 0
print("Source date:", dataset[index][0])
print("Target date:", dataset[index][1])
print()
print("Source after preprocessing (indices):", X[index])
print("Target after preprocessing (indices):", Y[index])
print()
print("Source after preprocessing (one-hot):", Xoh[index])
print("Target after preprocessing (one-hot):", Yoh[index])

class traindata(Dataset):
    def __init__(self) -> None:
        super().__init__()
        X, Y, Xoh, Yoh = preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty)
        self.X,self.Y=Xoh,Yoh
        
    def __getitem__(self, index):
        return self.X[index],self.Y[index]

    def __len__(self):
        return self.X.shape[0]

def my_collate(batch):
    data = [item[0] for item in batch]
    data = torch.stack(data)
    target = [item[1] for item in batch]
    target = torch.stack(target)
    target = torch.argmax(target,2)
    return [data, target]

Xdata = traindata()
datas = dataloader.DataLoader(Xdata,100,True,collate_fn=my_collate)


def calacc(X_test,Y_test,net):
    out = torch.max(net(torch.FloatTensor(X_test)),dim=1)[1]
    acc = np.sum(out.numpy()==Y_test)/Y_test.shape[0]
    print('accuracy = {}%'.format(acc*100))
    
class EncoderRNN(nn.Module):
    def __init__(self,):
        super(EncoderRNN, self).__init__()
        self.gru = nn.GRU(37, 100,bidirectional=True)

    def forward(self, input):
        output, hidden = self.gru(input)
        # output = output[:, :, :self.hidden_size] + output[:, : ,self.hidden_size:]
        return output, hidden

class AttnDecoderRNN(nn.Module):
    def __init__(self,dropout_p=0.1, max_length=11):
        super(AttnDecoderRNN, self).__init__()
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.attn = nn.Linear(100*2+11,200)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(200, 128)
        self.out = nn.Linear(128, 11)

    def forward(self, input, hidden):
        input =torch.stack([input.squeeze(0) for i in range(hidden.shape[1])],dim=1)
        input = self.attn(torch.cat([input,hidden],2))
        attn_weights = F.softmax(input, dim=2)
        attn_applied = torch.sum(attn_weights * hidden,1)
        # print(attn_applied.shape)
        output,hi = self.gru(attn_applied.unsqueeze(1))
        output=self.out(output.squeeze(1))
        return output, hi

encoder_net = EncoderRNN()
decoder_net = AttnDecoderRNN()
encoder_optimizer = torch.optim.Adam(encoder_net.parameters())
decoder_optimizer = torch.optim.Adam(decoder_net.parameters())
criterion = nn.CrossEntropyLoss()
epochs=10
for i in range(epochs):
    encoder_net.train() 
    decoder_net.train()
    for _,(x,y) in enumerate(datas):
        encoder_output, encoder_hidden = encoder_net(x.float())
        decoder_input = torch.zeros(1,encoder_output.shape[0],11)
        decoder_hidden = encoder_output
        loss=0
        for di in range(10):
            decoder_input, _ = decoder_net(
                decoder_input, decoder_hidden)
            loss += criterion(decoder_input, y[:,di])
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
        print('episode{} finished,loss = {}'.format(i,loss.item()))


EXAMPLES = ['3 May 1979', '5 April 09', '21th of August 2016', 'Tue 10 Jul 2007', 'Saturday May 9 2018', 'March 3 2001', 'March 3rd 2001', '1 March 2001']
ex=[]
for example in EXAMPLES:
    source = torch.tensor(string_to_int(example, 30, human_vocab))
    source = torch.stack(list(map(lambda x: F.one_hot(x, num_classes=len(human_vocab)), source)))
    ex.append(source)
ex = torch.stack(ex)
res=[]
with torch.no_grad():
    encoder_output, encoder_hidden = encoder_net(ex.float())
    decoder_input = torch.zeros(1,encoder_output.shape[0],11)
    decoder_hidden = encoder_output
    for di in range(10):
        decoder_input, _ = decoder_net(decoder_input, decoder_hidden)
        res.append(decoder_input.argmax(dim=1).numpy())
res = np.stack(res,1)
for index,k in enumerate(res):
    output = [inv_machine_vocab[int(i)] for i in k]
    print("source:", EXAMPLES[index])
    print("output:", ''.join(output))