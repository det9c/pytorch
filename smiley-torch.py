import torch
import torch.nn as nn
import torch.utils.data as data
import time
import matplotlib.pyplot as plt
import math
import sys
import scipy
import numpy as np
import random
import os
from skimage import io
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import progressbar

def get_anchor(iside_len,ibottom_len,iguess_up,iguess_across,npixels):
    i=0    
    while i<100000: #infinite loop until stopping condition is satisfied
          iend=iguess_up+iside_len
          if(iend > npixels):
              iguess_up=iguess_up-1
          else:
              break
    i=0
    while i<100000: #infinite loop until stopping condition is satisfied
          iend=iguess_across+ibottom_len
          if(iend > npixels):
              iguess_across=iguess_across-1
          else:
              break

    return iguess_up,iguess_across      
          

def get_anchor_sad(iside_len,ibottom_len,iguess_up,iguess_across,npixels):
    i=0
    while i<100000: #infinite loop until stopping condition is satisfied
          iend=iguess_up-iside_len
          if(iend < 0):
              iguess_up=iguess_up+1
          else:
              break
    i=0
    while i<100000: #infinite loop until stopping condition is satisfied
          iend=iguess_across+ibottom_len
          if(iend > npixels):
              iguess_across=iguess_across-1
          else:
              break

    return iguess_up,iguess_across




np.random.seed(100)
npixels=32
nimages=10000
ibaseline=5
ipic_num=0
picture=np.zeros((npixels,npixels,3))
icount=0
data_set=np.zeros((2*nimages,npixels,npixels,3))
categories=np.zeros((2*nimages))
irow=-1
while ipic_num<nimages:
    iside_len=ibaseline+round(np.random.random()*10)
    ibottom_len=ibaseline+round(np.random.random()*10)
    iguess_up=round(np.random.random()*(npixels-1))
    iguess_across=round(np.random.random()*(npixels-1))
    ianchor_a,ianchor_b=get_anchor(iside_len,ibottom_len,iguess_up,iguess_across,(npixels-1))
#    print(str(iside_len)+" "+str(ibottom_len)+" "+str(ianchor_a)+" "+str(ianchor_b))
    picture[:,:,:]=0.0
    picture[:,:,0:3]=0.0
#    print(str(ianchor_a)+" "+str(iside_len))
    hue=[]
    hue=np.random.random(3)
    picture[ianchor_a:(ianchor_a+iside_len-1),ianchor_b,:]=hue
    picture[ianchor_a:(ianchor_a+iside_len-1),(ianchor_b+ibottom_len-1),:]=hue
    picture[ianchor_a+iside_len-1,ianchor_b:(ianchor_b+ibottom_len),:]=hue
    irow+=1
    data_set[irow]=picture
    categories[irow]=0
    icount+=1
#    plt.subplot(5,8,icount)
#    plt.imshow(picture)
    ipic_num+=1

ipic_num=0
while ipic_num<nimages:
    iside_len=ibaseline+round(np.random.random()*10)
    ibottom_len=ibaseline+round(np.random.random()*10)
    iguess_up=round(np.random.random()*(npixels-1))
    iguess_across=round(np.random.random()*(npixels-1))
    ianchor_a,ianchor_b=get_anchor_sad(iside_len,ibottom_len,iguess_up,iguess_across,(npixels-1))
#    print(str(iside_len)+" "+str(ibottom_len)+" "+str(ianchor_a)+" "+str(ianchor_b))
    picture[:,:,:]=0.0
    picture[:,:,0:3]=0.0
#    print(str(ianchor_a)+" "+str(iside_len))
    hue=[]
    hue=np.random.random(3)
    picture[(ianchor_a-iside_len+1):ianchor_a,ianchor_b,:]=hue
    picture[(ianchor_a-iside_len+1):ianchor_a,(ianchor_b+ibottom_len-1),:]=hue
    picture[ianchor_a-iside_len+1,ianchor_b:(ianchor_b+ibottom_len),:]=hue
    irow+=1
    data_set[irow]=picture
    categories[irow]=1
    icount+=1
#    plt.subplot(5,8,icount)
#    plt.imshow(picture)
    ipic_num+=1

#plt.show()

icount=0
i=0
while i<40:
  icount+=1
  plt.subplot(5,8,icount)
  plt.imshow(data_set[i])
  i+=1
    
#plt.show()


np.random.seed(100)
train_images, test_images, train_labels, test_labels =train_test_split(data_set,categories,test_size=.1)

ntrain=len(train_images)
scale_values=np.zeros((npixels,npixels,3,2))
i=0
while i<npixels:
    j=0
    while j<npixels:
        k=0
        while k<3:
            scale_values[i,j,k,0]=np.mean(train_images[:,i,j,k])
            scale_values[i,j,k,1]=np.std(train_images[:,i,j,k]  )                   
            train_images[:,i,j,k]=train_images[:,i,j,k]-scale_values[i,j,k,0]
            test_images[:,i,j,k]=test_images[:,i,j,k]-scale_values[i,j,k,0]
            if(scale_values[i,j,k,1] >0.0):
               train_images[:,i,j,k]=train_images[:,i,j,k]/scale_values[i,j,k,1]
               test_images[:,i,j,k]=test_images[:,i,j,k]/scale_values[i,j,k,1]
            k+=1
        j+=1
    i+=1


'''
np.random.seed(100)
#CNN
#model = keras.Sequential([keras.layers.Conv2D(input_shape=(npixels,npixels,3),filters=8,kernel_size=2,strides=(1,1),padding='valid',data_format="channels_last",use_bias=True,activation=tf.nn.relu),keras.layers.Flatten(),keras.layers.Dense(100, activation=tf.nn.relu),keras.layers.Dense(2, activation=tf.nn.softmax)])

model=keras.Sequential()
model.add(keras.layers.Conv2D(input_shape=(npixels,npixels,3),filters=8,kernel_size=2,strides=(1,1),padding='valid',data_format="channels_last",use_bias=True,activation=tf.nn.relu))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(100,activation=tf.nn.relu))
model.add(keras.layers.Dense(2,activation=tf.nn.softmax))
'''

class NeuralNet(nn.Module):

    def __init__(self,dim1,dim2):
        super(NeuralNet, self).__init__()
        self.input = nn.Linear(dim1, dim2 ) #dim1 is input size, dim2 is outputsize
        self.sig = nn.Sigmoid()
        self.fc2 = nn.Linear(dim2, 2)


    def forward(self, x):
        x = x.view(-1, self.num_flat_features(x))
        out = self.input(x)
        out = self.sig(out)
        yout = self.fc2(out)
        return yout

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



#model = NeuralNet(input_size, hidden_size, num_classes).to(device)
torch.manual_seed(7)
model=NeuralNet(npixels*npixels*3,1000)
#print(list(model.parameters()))

# Loss and optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=.01)

cutoff=.0001
xtrain_tch=torch.from_numpy(train_images).float()
ytrain_tch=torch.from_numpy(train_labels).float()
metric=nn.CrossEntropyLoss()
t=data.TensorDataset(xtrain_tch,ytrain_tch)
batch_size=100
batches=data.DataLoader(t,batch_size=batch_size,shuffle=True)
mets=np.zeros((180,1))
xtest_tch=torch.from_numpy(test_images).float()
ytest_tch=torch.from_numpy(test_labels).float()
tmp=[]
for epoch in range(0,100):
     icount=-1
     for batch in progressbar.progressbar(batches):
        pred = model(batch[0])
        loss = metric(pred, batch[1].long())
# Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        icount+=1
        mets[icount,0]=loss.item()
     eloss=np.mean(mets[:][0])
     print('loss after epoch '+str(epoch)+' is '+str(eloss))
     pred=model(xtest_tch)
     imax=len(test_images)
     i=0
     predictions=[]
     while i<imax:
       predictions.append(torch.argmax(pred[i]).item())
       i+=1
     d=confusion_matrix(predictions,test_labels)
     print("test set error is "+str(100*(d[0,1]+d[1,0])/np.sum(d)))
     tmp.append( 100*(d[0,1]+d[1,0])/np.sum(d))
     if(eloss<cutoff):
         break
     

pred=model(xtest_tch)
imax=len(test_images)
i=0
predictions=[]
while i<imax:
    predictions.append(torch.argmax(pred[i]).item())
    i+=1

print(confusion_matrix(predictions,test_labels))

'''
direction=['up','down']
i=0
while i<imax:
  if(predictions[i] != test_labels[i]):
    plt.imshow(test_images[i])
    word=str(direction[predictions[i]])+" "+str(pred[i])
    plt.xlabel(word,fontsize=12,fontweight='bold')
    plt.show()
  i+=1
'''
