import torch
import torch.nn as nn

class Siamese(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, stride=1, kernel_size=10, padding=0),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv2=nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, stride=1, kernel_size=7, padding=0),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv3=nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, stride=1, kernel_size=4, padding=0),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv4=nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, stride=1, kernel_size=4, padding=0),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fully_layer=nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=256, out_features=300),
        )

    def forward_once(self,inp):
        inp = self.conv1(inp)
        inp = self.conv2(inp)
        inp = self.conv3(inp)
        inp = self.conv4(inp)

        inp = self.adaptive_pool(inp)

        inp = self.fully_layer(inp)

        return inp

    def forward(self,inp_1,inp_2):

        inp_1=self.forward_once(inp_1)
        inp_2=self.forward_once(inp_2)

        return inp_1, inp_2