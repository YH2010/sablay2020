import torch
import torch.nn as nn

import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

# import adabound

import gc, time, os, sys
import cv2
import numpy as np
import matplotlib as plt

from helpers import config
from PIL import Image
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler

sys.stdout.write("Output Folder : %s\n\n"%(str(config.TIME)))

sys.stdout.write("Preprocessing Steps : Green-filter, ROI, Histogram, Resize\n\n")

device = 1
if torch.cuda.is_available():
    torch.cuda.set_device(device)

#Define the batch size, the model, the loss function and the optimizer
batch_size = 32

# model = models.resnet50(pretrained=True)
model = models.vgg19_bn(num_classes=4)
# model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3) # for ResNet
model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1) # for VGG
# model.features[0] = nn.Conv2d(1, 96, kernel_size=7, stride=2, padding=3, bias=False) # for Densenet
class_weights = torch.FloatTensor([20.28,3.53,19.47,1.0]).cuda()
loss_fcn = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
# optimizer = adabound.AdaBound(model.parameters(), lr=1e-3, final_lr=0.01)

sys.stdout.write("Batch Size : %s\n"%(batch_size))
sys.stdout.write("Model : VGG-19-BN, untrained\n")
sys.stdout.write("Loss Function : CrossEntropyLoss(weights = (C)20.28, (D)3.53, (G)19.47, (N)1.0)\n")
sys.stdout.write("Optimizer : Adam(lr=1e-2)\n")
sys.stdout.flush()

class ImageFolderRevised(datasets.ImageFolder):
    # override the __getitem__ method that dataloader calls
    def __getitem__(self, index):
        path, target = self.imgs[index]
        orig_img = cv2.imread(path)
        g_channel = orig_img[:,:,1]
        _, thresh_img = cv2.threshold(g_channel,15,255,cv2.THRESH_BINARY)
        edged = cv2.Canny(thresh_img, 100, 255)
        contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = np.concatenate(contours)
        x,y,w,h = cv2.boundingRect(contours)
        g_channel_bound = cv2.rectangle(g_channel,(x,y,w,h),255,5)
        g_channel_crop = g_channel[y:y+h, x:x+w]
        image = cv2.equalizeHist(g_channel_crop)
        image = cv2.resize(image, (224, 224))
        image = image.reshape((1, 224, 224))
        image = torch.tensor(image, dtype=torch.float)
        # image = Image.fromarray(image)
        return image, target

dataset = ImageFolderRevised(root = os.path.sep.join([config.DATASET_DATASET_PATH,"Oct9"]))

dataset_size = len(dataset)
indices = list(range(dataset_size))
np.random.shuffle(indices)

train_split, test_split, val_split = .7, .1, .2
split1 = int(np.floor(dataset_size * train_split))
split2 = split1 + int(np.floor(dataset_size * test_split))
train_indices, test_indices, val_indices = indices[:split1], indices[split1:split2], indices[split2:]

sys.stdout.write("Train Indices\n%s\n\n"%(train_indices))
sys.stdout.write("Test Indices\n%s\n\n"%(test_indices))
sys.stdout.write("Val Indices\n%s\n\n"%(val_indices))
sys.stdout.flush()

# batch_size = 32
train_load = torch.utils.data.DataLoader(dataset = dataset,
                                         batch_size = batch_size,
                                         sampler = SubsetRandomSampler(train_indices))

test_load = torch.utils.data.DataLoader(dataset = dataset,
                                        batch_size = batch_size,
                                        sampler = SubsetRandomSampler(test_indices))

val_load = torch.utils.data.DataLoader(dataset = dataset,
                                       batch_size = batch_size,
                                       sampler = SubsetRandomSampler(val_indices))

# #Define the model, the loss function and the optimizer
# model = models.densenet161()
# loss_fcn = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
# #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

num_epochs = 1000
for epoch in range(num_epochs):

    start = time.time()

    # # # # #  T R A I N I N G  # # # # #

    #Put the network into training mode
    model.train()

    #Reset these below variables to 0 at the begining of every epoch
    correct = 0
    iter_loss = 0.0

    for i, (inputs, labels) in enumerate(train_load):

        # Convert torch tensor to Variable
        inputs = Variable(inputs)
        labels = Variable(labels)

        # If we have GPU, shift the data to GPU
        if torch.cuda.is_available():
            model.cuda()
            inputs = inputs.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()           # Clear off the gradient in (w = w - gradient)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        loss = loss_fcn(outputs, labels)
        loss.backward()                 # Backpropagation
        optimizer.step()                # Update the weights

        iter_loss += loss.data.item()   # Accumulate the loss

        # Record the correct predictions for training data
        correct += (predicted == labels).sum()

        if(epoch==0):
            sys.stdout.write("Train %s/%s, Time:%ss\n" % (i+1, len(train_load), time.time()-start))
            sys.stdout.flush()

    # Record the training loss and training accuracy
    train_loss = iter_loss / len(train_load)
    train_accuracy = 100 * correct / float(len(train_indices))


    # # # # #  V A L I D A T I O N  # # # # #

    #Put the network into evaluation/testing mode
    model.eval()

    correct = 0
    iter_loss = 0.0

    for i, (inputs, labels) in enumerate(val_load):

        inputs = Variable(inputs)
        labels = Variable(labels)

        if torch.cuda.is_available():
            model.cuda()
            inputs = inputs.cuda()
            labels = labels.cuda()

        # optimizer.zero_grad()
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        loss = loss_fcn(outputs, labels)

        iter_loss += loss.data.item()

        correct += (predicted == labels).sum()

        if(epoch==0):
            sys.stdout.write("Validation %s/%s, Time:%ss\n" % (i+1, len(val_load), time.time()-start))
            sys.stdout.flush()

    # Record the testing loss and testing accuracy
    val_loss = iter_loss / len(val_load)
    val_accuracy = 100 * correct / float(len(test_indices))
    stop = time.time()

    sys.stdout.write('Epoch {}/{}, Training Loss: {:.3f}, Training Accuracy: {:.3f}, Validation Loss: {:.3f}, Validation Accuracy: {:.3f}, Time: {}s\n'
          .format(epoch+1, num_epochs, train_loss, train_accuracy, val_loss, val_accuracy, stop-start))
    sys.stdout.flush()

    if (epoch+1) % 20 == 0:
        #Save the model
        dirPath = os.path.sep.join([config.OUTPUT_PATH, str(config.TIME)])
        if not os.path.exists(dirPath):
            os.makedirs(dirPath)
        torch.save(model.state_dict(), os.path.sep.join([dirPath, 'model_'+str(epoch+1)+'.pth']))
        sys.stdout.write("Model saved!\n")
        sys.stdout.flush()
