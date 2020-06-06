#by Ekaterina Gurina
import torch.nn as nn

class MaskDetectionModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels = 2):
        super(MaskDetectionModel, self).__init__()
        self.block1 = ConvBlock(in_channels, hidden_channels)
        self.block2 = ConvBlock(hidden_channels, hidden_channels*2)
        self.block3 = ConvBlock(hidden_channels*2, hidden_channels*4, stride = 3)
        self.block4 = nn.Sequential(nn.Linear(2048, 1024),
                                    nn.ReLU(),
                                    nn.Linear(1024, out_channels), 
                                    nn.Softmax(dim = 1))
        
        # Initialize layers' weights
        for sequential in [self.block1, self.block2, self.block3, self.block4]:
            for layer in sequential.children():
                if isinstance(layer, (nn.Linear, nn.Conv2d)):
                    nn.init.xavier_uniform_(layer.weight)
    
    def forward(self, x):
        out = self.block1(x)
        print
        out = self.block2(out)
        out = self.block3(out)
        out = out.view(-1, 2048)
        out = self.block4(out)
        return out
		
class ConvBlock(nn.Module):
  def __init__(self, in_channels, hidden_channels, kernel_size = 3, padding = 1, stride = 1):
    super(ConvBlock, self).__init__()
    self.block = nn.Sequential(nn.Conv2d(in_channels, hidden_channels, kernel_size=kernel_size, padding= padding, stride=stride),
                                nn.ReLU(),
                               nn.MaxPool2d(2))
  def __call__(self, x):
    return self.block(x)