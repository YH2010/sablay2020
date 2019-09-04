import torch
import torch.nn as nn

import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

import gc, time, os
import numpy as np

from helpers import config
from PIL import Image
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler

#Define image transformations
# transform_left = transforms.Compose([transforms.Resize((224, 224)),
#                                     transforms.RandomHorizontalFlip(),
#                                     transforms.ToTensor(),
#                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# transform_right = transforms.Compose([transforms.Resize((224, 224)),
#                                     transforms.ToTensor(),
#                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

#Load the dataset
dataset = datasets.ImageFolder(root = os.path.sep.join([config.DATASET_DATASET_PATH, "Ken"]),
                               transform = transform)

dataset_size = len(dataset)
indices = list(range(dataset_size))
np.random.shuffle(indices)

train_split, test_split, val_split = .7, .2, .1
split1 = int(np.floor(dataset_size * train_split))
split2 = split1 + int(np.floor(dataset_size * test_split))
train_indices, test_indices, val_indices = indices[:split1], indices[split1:split2], indices[split2:]

batch_size = 15
train_load = torch.utils.data.DataLoader(dataset = dataset,
                                         batch_size = batch_size,
                                         sampler = SubsetRandomSampler(train_indices))

test_load = torch.utils.data.DataLoader(dataset = dataset,
                                        batch_size = batch_size,
                                        sampler = SubsetRandomSampler(test_indices))

val_load = torch.utils.data.DataLoader(dataset = dataset,
                                       batch_size = batch_size,
                                       sampler = SubsetRandomSampler(val_indices))

#Define the model, the loss function and the optimizer
model = models.resnet18()
loss_fcn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

#Define the lists to store the results of loss and accuracy
train_loss = []
test_loss = []
train_accuracy = []
test_accuracy = []

num_epochs = 50
for epoch in range(num_epochs):

    start = time.time()

    # # # # #  T R A I N I N G  # # # # #

    #Put the network into training mode
    model.train()
    scheduler.step()

    #Reset these below variables to 0 at the begining of every epoch
    correct = 0
    iterations = 0
    iter_loss = 0.0

    for inputs, labels in train_load:

        # Convert torch tensor to Variable
        inputs = Variable(inputs)
        labels = Variable(labels)

        # If we have GPU, shift the data to GPU
        if torch.cuda.is_available():
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
        iterations += 1

    # Record the training loss and training accuracy
    train_loss = iter_loss/iterations
    train_accuracy = (100 * correct / len(train_dataset))


    # # # # #  V A L I D A T I O N  # # # # #

    #Put the network into evaluation/testing mode
    model.eval()

    correct = 0
    iterations = 0
    iter_loss = 0.0

    for inputs, labels in val_load:

        inputs = Variable(inputs)
        labels = Variable(labels)

        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        loss = loss_fcn(outputs, labels)

        iter_loss += loss.data.item()

        correct += (predicted == labels).sum()
        iterations += 1

    # Record the testing loss and testing accuracy
    val_loss = iter_loss/iterations
    val_accuracy = (100 * correct / len(test_dataset))
    stop = time.time()

    print('Epoch {}/{}, Training Loss: {:.3f}, Training Accuracy: {:.3f}, Validation Loss: {:.3f}, Validation Accuracy: {:.3f}, Time: {}s'
          .format(epoch+1, num_epochs, train_loss, train_accuracy, val_loss, val_accuracy, stop-start))

#Save the model
dirPath = os.path.sep.join([config.OUTPUT_PATH, str(config.TIME)])
if not os.path.exists(dirPath):
	os.makedirs(dirPath)
torch.save(model.state_dict(), os.path.sep.join([dirPath, 'model.pth']))

#Load the model
# model.load_state_dict(torch.load('model.pth'))