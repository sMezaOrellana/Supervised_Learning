class MNIST_Test(Dataset):
    def __init__(self, XY):
        self.X_test = XY[0]
        self.y_test = XY[1]
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

    def get_test(self):
        return torch.from_numpy(self.X_test), torch.from_numpy(self.y_test).type(torch.LongTensor)
    
    def get_training(self):
        return torch.from_numpy(self.X_train) , torch.from_numpy(self.y_train).type(torch.LongTensor)

class NN(nn.Module):
    def __init__(self, input_size, num_classes):
            super(NN, self).__init__()
            self.fc1 = nn.Linear(input_size, 100)
            self.fc2 = nn.Linear(100, num_classes)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

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
    training_dataset = dataset.get_training()
    test_dataset = dataset.get_test()
    dataloader_training = DataLoader(dataset=dataset,batch_size = batch_size,shuffle=True)
    dataloader_test = DataLoader(dataset=dataset,batch_size = batch_size,shuffle=True)

    #init model
    model = NN(input_size, num_classes)

    #loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print('Hello world')

if __name__ == "__main__":
    main()