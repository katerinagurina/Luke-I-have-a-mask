#by Ekaterina Gurina

import torchvision.transforms as transforms
import torchvision.models as models
import torch
import cv2

import numpy as np
import torch.nn as nn

def gaussian_blur(img):
    image = np.array(img)
    image_blur = cv2.GaussianBlur(image,(65,65),10)
    new_image = image_blur
    return new_image

def MultilabelClassificator():
    my_transform = { 'train' : transforms.Compose([transforms.Resize((224,224)),
                                                   transforms.Lambda(gaussian_blur),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                                   ]),
                    'test': transforms.Compose([transforms.Resize((224,224)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
                    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=True)
    for layer, param in model.named_parameters():
        if 'layer4' not in layer:
            param.requires_grad = False

    model.fc = torch.nn.Sequential(nn.Linear(512, 32),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(32, 3),
                                     nn.LogSoftmax(dim=1))
    model.to(device)
    return model

def multilabel_classificator_v2():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(pretrained=True)
    for layer, param in model.named_parameters():
        param.requires_grad = True
    model.fc = torch.nn.Sequential(nn.Linear(2048, 32),
                                   nn.BatchNorm1d(32),
                                   nn.ReLU(),
                                   nn.Linear(32, 3),
                                   nn.Softmax(dim=1))
    model.to(device)
    return model