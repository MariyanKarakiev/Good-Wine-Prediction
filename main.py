import numpy as np
import pandas as pd
import torch
from torch import nn

train_url = "https://ml.monov.eu/wine/train_wine_quality.csv"
test_url = "https://ml.monov.eu/wine/test_wine_quality.csv"


train_data = pd.read_csv(train_url)
test_data = pd.read_csv(test_url)


all_features = pd.concat((train_data.iloc[:, 1:], test_data.iloc[:, 1:]))
n_train = train_data.shape[0]

#all_features['quality'][n_train:] = np.where(all_features['quality']=="AVERAGE QUALITY",
 #                                  1,||all_features['quality']=="AVERAGE QUALITY"))
#all_features['quality'][n_train:] = np.where(all_features['quality']=="NORMAL QUALITY",
  #                                 2,all_features['quality'])
#all_features['quality'][n_train:] = np.where(all_features['quality']=="LOW QUALITY",
             #                      3,all_features['quality'])


numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index


all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))

all_features[numeric_features] = all_features[numeric_features].fillna(0)

all_features = pd.get_dummies(all_features,columns = ['type'])

all_features['quality'] = all_features['quality'].replace({"AVERAGE QUALITY" : 1,
                                                           'LOW QUALITY' : 2,
                                                           'HIGH QUALITY' : 3})

train_features = torch.tensor(all_features[:n_train].values,
                              dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values,
                             dtype=torch.float32)
train_labels = torch.tensor(all_features[:n_train].quality.values.reshape(-1, 1),
                            dtype=torch.float32)



class Net(nn.Module):
  def __init__(self, in_features, out_features):
    super().__init__()
    self.fc1 = nn.Linear(in_features, 128)
    self.fc2 = nn.Linear(128, 64)
    self.output = nn.Linear(64, out_features)
    self.relu = nn.ReLU()

  def forward(self, x):
    x = self.relu(self.fc1(x))
    x = self.relu(self.fc2(x))
    x = self.output(x)
    return x

loss = nn.MSELoss()
in_features = train_features.shape[1]

print("Input features", in_features)

def train(net, train_features, train_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
  train_ls = []
  train_tensor = torch.utils.data.TensorDataset(train_features, 
                                                train_labels)
  train_loader = torch.utils.data.DataLoader(train_tensor,
                                             batch_size=batch_size, shuffle=True)

  # Use Adam optimization algorithm
  optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate,
                               weight_decay=weight_decay)

  for epoch in range(num_epochs):
    for X, y in train_loader:
      output = net(X)
      optimizer.zero_grad()
      l = loss(output, y)    
      l.backward()
      optimizer.step()    
    train_ls.append(l)
    if(epoch%10==0):
      print('epoch {}, loss {}'.format(epoch, l.item()))
  return train_ls


net = Net(in_features, 1)

num_epochs = 100
learning_rate = 0.001
weight_decay = 0
batch_size = 64


import matplotlib.pyplot as plt

def train_and_pred(train_features, test_feature, train_labels, test_data,
                   num_epochs, learning_rate, weight_decay, batch_size):
    net = Net(in_features, 1)

    train_ls = train(net, train_features, train_labels,
                 num_epochs, learning_rate,
                 weight_decay, batch_size)
    
   

    print(f'train loss {float(train_ls[-1]):f}')
    # Apply the network to the test set
    preds = net(test_features).detach().numpy()

    # Reformat it to export to Kaggle
    result = test_data 
    result['quality'] = pd.Series(preds.reshape(1, -1)[0])
    submission = result['quality']
    submission.to_csv('submission.csv', index=False)
    
    
train_and_pred(train_features, test_features, train_labels, test_data,
              num_epochs, learning_rate, weight_decay, batch_size)
