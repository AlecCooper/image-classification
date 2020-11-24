import torch as torch
import torch.nn as nn
import torch.nn.functional as func

class Net(nn.Module):


    def __init__(self):
        super(Net, self).__init__()

        # Create the layers of the encoder
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=256,kernel_size=16, stride=10)
        self.pool1 = nn.MaxPool2d((2,2))
        self.fc1 = nn.Linear(20736, 8000)
        self.fc2 = nn.Linear(8000, 4096)
        self.fc3 = nn.Linear(4096, 3)
        
    # Feedforward function
    def forward(self,x):

        x = func.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.reshape(x,[20736])
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        y = torch.sigmoid(self.fc3(x))

        return y      

    # Test function. Avoids calculation of gradients.
    def test(self, testloader, loss_func, epoch):
        self.eval()
        cross_val=0
        with torch.no_grad():
                for inputs, targets in testloader:
                    cross_val += loss_func(self.forward(inputs), torch.flatten(targets))
        return cross_val.item()