import torch 
import torchvision.models

class Decoder(torch.nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.input_dim = int(128 * (128/16) * (128/16) / 8)
        self.layer1 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.input_dim, 512, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(512),
            torch.nn.ReLU()
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(512, 128, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(128),
            torch.nn.ReLU()
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(128, 32, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU()
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(32, 8, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(8),
            torch.nn.ReLU()
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(8, 8, kernel_size=2, stride=2, bias=False),
            torch.nn.Sigmoid()
        )
        self.layer6 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(8, 1, kernel_size=1, bias=False),
            torch.nn.Sigmoid()
        )
        
    def forward(self, x):
        x = x.view(-1, self.input_dim, 2, 2, 2) 
        x = self.layer1(x)  
        # print(x.size())
        x = self.layer2(x)
        # print(x.size()) 
        x = self.layer3(x)
        # print(x.size())
        x = self.layer4(x)
        # print(x.size())
        x = self.layer5(x)
        # print(x.size())
        x = self.layer6(x)
        # print(x.size())
        return x