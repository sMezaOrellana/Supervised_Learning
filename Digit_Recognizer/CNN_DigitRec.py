import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

device = torch.device('cpu')

class DataSet_M(Dataset):
    def __init__(self, XY):
        self.X = XY[0]
        self.y = XY[1]
        self.n_samples = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.n_samples

class MNIST_Data(Dataset):
    def __init__(self):
        matrix = np.loadtxt('data/train.csv', delimiter=',', skiprows=1, dtype=np.float32)

        y = matrix[:,0]
        x = matrix[:,1:]
        X = x/255
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        self.n_rows_train = self.X_train.shape[0]
        self.n_rows_test = self.X_test.shape[0]
    def get_test(self):
        return (torch.from_numpy(self.X_test)).reshape([self.n_rows_test,1,28,28]), torch.from_numpy(self.y_test).type(torch.LongTensor)
    
    def get_training(self):
        return torch.from_numpy(self.X_train).reshape([self.n_rows_train,1,28,28]) , torch.from_numpy(self.y_train).type(torch.LongTensor)

class CNN(nn.Module):
    def __init__(self, input_size, num_classes):
            super(CNN, self).__init__()
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1))
            self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
            self.conv2 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=(3,3), stride=(1,1), padding=(1,1))
            self.fc1 = nn.Linear(32*7*7, num_classes)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0],-1)
        x = self.fc1(x)
        return x

def check_accuracy(model,loader):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x,y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        print(f'got {num_correct} / {num_samples}, with accuracy {float(num_correct/num_samples)}')

def main():
    #input parameters
    input_size    = 784
    num_classes   = 10

    #hyper parameters
    learning_rate = 0.001
    num_epochs    = 4
    batch_size    = 5

    #load data
    dataset = MNIST_Data()
    training_dataset = DataSet_M(dataset.get_training())
    test_dataset = DataSet_M(dataset.get_test())
    dataloader_training = DataLoader(dataset=training_dataset, batch_size=batch_size,shuffle=True)
    dataloader_test = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    #init model
    model = CNN(input_size, num_classes)

    #loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    #training loop
    for epoch in range(num_epochs):
        for batch_idx, (data, targets) in  enumerate(dataloader_training):
            data = data.to(device=device)
            targets = targets.to(device=device)

            scores = model(data)
            loss = criterion(scores,targets)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
    print('Done with training...')

    #Accuracy of the model
    print('Accuracy Training Data:')
    check_accuracy(model, dataloader_training)
    print('Accuracy Testing Data:')
    check_accuracy(model, dataloader_test)

if __name__ == "__main__":
    main()