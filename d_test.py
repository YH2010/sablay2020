import torch
import torch.nn as nn

import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor

import gc, time, os, sys
import numpy as np

from helpers import config
from PIL import Image
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler

# Load the model
model = models.resnet18(pretrained=True)
model.load_state_dict(torch.load('output/091119154423/model_50.pth'))
model.eval()
#iter_loss = 0
#correct = 0

#for i, (inputs, labels) in enumerate(test_load):

    #inputs = Variable(inputs)
    #labels = Variable(labels)

    #if torch.cuda.is_available():
        #model = nn.DataParallel(model)
        #model.cuda()
        #inputs = inputs.cuda()
        #labels = labels.cuda()

    #optimizer.zero_grad()
    #outputs = model(inputs)
    #_, predicted = torch.max(outputs, 1)
    #loss = loss_fcn(outputs, labels)

    #iter_loss += loss.data.item()

    #correct += (predicted == labels).sum()

#test_loss = 
#sys.stdout.write("Test Loss %s, Test Accuracy %s\n" % (test_loss, test_accuracy)
#sys.stdout.flush()

transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

file_name = sys.argv[1]
image = Image.open(file_name)
image = transform(image)
# image = ToTensor()(image).unsqueeze(0)
image = Variable(image, requires_grad=True)
print(image.shape)
#image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
#print(model)
#print(model(image))
# outputs = model(image)
# _, predicted = torch.max(outputs, 1)
# print(predicted)
