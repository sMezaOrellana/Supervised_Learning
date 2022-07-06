import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

device = torch.device('cpu')

class MNIST_Test(Dataset):
    def __init__(self, X_test,y_test):
        self.X_test = X_test
        self.y_test = y_test
        self.n_samples = self.X_test.shape[0]

    def __getitem__(self, index):
        return self.X_test[index], self.y_test[index]

    def __len__(self):
        return self.n_samples

class MNIST_Data(Dataset):
    def __init__(self):
        matrix = np.loadtxt('data/train.csv', delimiter=',', skiprows=1, dtype=np.float32)
        self.n_samples = matrix.shape[0]

        y = matrix[:,0]
        x = matrix[:,1:]
        X = x/255

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        self.n_samples = self.X_train.shape[0]
        self.X_train = torch.from_numpy(self.X_train)
        self.y_train = torch.from_numpy(self.y_train)
        self.y_train = self.y_train.type(torch.LongTensor)

    def __getitem__(self,index):
        return self.X_train[index], self.y_train[index]

    def __len__(self):
        return self.n_samples

    def get_test(self):
        return torch.from_numpy(self.X_test), torch.from_numpy(self.y_test).type(torch.LongTensor)


class NN(nn.Module):
    def __init__(self, input_size, num_classes):
            super(NN, self).__init__()
            self.fc1 = nn.Linear(input_size, 100)
            self.fc2 = nn.Linear(100, num_classes)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
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
    dataloader = DataLoader(dataset=dataset,batch_size = batch_size,shuffle=True)

    #init model
    model = NN(input_size, num_classes)

    #loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    #training loop
    for epoch in range(num_epochs):
        for batch_idx, (data, targets) in  enumerate(dataloader):
            data = data.to(device=device)
            targets = targets.to(device=device)

            scores = model(data)
            loss = criterion(scores,targets)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

    print('Done with training...')
    #testing data
    X_test, y_test = dataset.get_test()
    test_dataset = MNIST_Test(X_test, y_test)
    test_loader = DataLoader(dataset=test_dataset ,batch_size = batch_size,shuffle=True)
    check_accuracy(model, test_loader)
    check_accuracy(model, dataloader)

    #accuracy checking
if __name__ == "__main__":
    main()