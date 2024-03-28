import torch 
import torchvision.models

class Refiner(torch.nn.Module):
    def __init__(self):
        super(Refiner, self).__init__()
        
        # Layer Definition
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv3d(1, 32, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(32),
            torch.nn.LeakyReLU(.2),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv3d(32, 64, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(64),
            torch.nn.LeakyReLU(.2),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv3d(64, 128, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(128),
            torch.nn.LeakyReLU(.2),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv3d(128, 128, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(128),
            torch.nn.LeakyReLU(.2),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.Linear(8192, 2048),
            torch.nn.Dropout(0.4),
            torch.nn.ReLU()
        )
        self.layer6 = torch.nn.Sequential(
            torch.nn.Linear(2048, 8192),
            torch.nn.Dropout(0.4),
            torch.nn.ReLU()
        )
        self.layer7 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(128, 128, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(128),
            torch.nn.ReLU()
        )
        self.layer8 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(64),
            torch.nn.ReLU()
        )
        self.layer9 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.Sigmoid()
        )
        self.layer10 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(32, 1, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.Sigmoid()
        )
        
        
    def forward(self, x):
        x_64_l = x
        x_32_l = self.layer1(x)
        x_16_l = self.layer2(x_32_l)
        x_8_l = self.layer3(x_16_l)
        x_4_l = self.layer4(x_8_l)
        flatten_features = self.layer5(x_4_l.view(-1,8192))
        flatten_features = self.layer6(flatten_features)
        x_4_r = x_4_l + flatten_features.view(-1, 128, 4, 4, 4)
        x_8_r = x_8_l + self.layer7(x_4_r)
        x_16_r = x_16_l + self.layer8(x_8_r)
        x_32_r = x_32_l + self.layer9(x_16_r)
        x_64_r = (x_64_l + self.layer10(x_32_r))*0.5
        return x_64_r