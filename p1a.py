import torch
import torchvision
import torchvision.datasets as dset
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
from PIL import Image

class Loader(Dataset):
    def __init__(self, train=True, transform=None):
        self.root = "./lfw/"
        self.transform = transform
        if train:
            self.path = "./train.txt"
        else:
            self.path = "./test.txt"

        self.data = []

        with open(self.path) as f:
            for line in f:
                self.data.append(line.split())

    def __getitem__(self,index):
        img1_path, img2_path, label = self.data[index]

        img1 = Image.open(self.root + img1_path)
        img2 = Image.open(self.root + img2_path)
        
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, torch.from_numpy(np.array([int(label)],dtype=np.float32))

    def __len__(self):
        return len(self.data)

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=(1,1), padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, stride=(2,2)),
            nn.Conv2d(64, 128, kernel_size=5, stride=(1,1), padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, stride=(2,2)),
            nn.Conv2d(128, 256, kernel_size=3, stride=(1,1), padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, stride=(2,2)),
            nn.Conv2d(256, 512, kernel_size=3, stride=(1,1), padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512)
            )
        self.fc1 = nn.Sequential(
            nn.Linear(131072, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(1024)
            )
        self.fc2 = nn.Sequential(
            nn.Linear(2048, 1),
            nn.ReLU(inplace=True)
            )
    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, img1, img2):
        output1 = self.forward_once(img1)
        output2 = self.forward_once(img2)
        out = torch.cat((output1, output2), 1)
        return self.fc2(out)

def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()

dataset = Loader(train=True, transform=transforms.Compose([transforms.Scale((128,128)),
                                                           transforms.ToTensor()]))
train_dataloader = DataLoader(dataset, shuffle=True, num_workers=8, batch_size=8)

net = SiameseNetwork().cuda()
criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr = .0005)

counter = []
loss_history = []
iteration_number= 0
m = nn.Sigmoid()

for epoch in range(0, 20):
    for i, data in enumerate(train_dataloader,0):
        img1, img2, label = data
        img1, img2, label = Variable(img1).cuda(), Variable(img2).cuda(), Variable(label).cuda()
        output = net(img1, img2)
        optimizer.zero_grad()
        loss = criterion(m(output), label)
        loss.backward()
        optimizer.step()
        if i %100 == 0 :
            print("Epoch number {}\n Current loss {}\n".format(epoch,loss.data[0]))
            iteration_number +=10
            counter.append(iteration_number)
            loss_history.append(loss.data[0])
show_plot(counter,loss_history)
