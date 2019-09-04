import torch
import torch.nn as nn

import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

import gc, time, os

from helpers import config
from PIL import Image
from torch.autograd import Variable

#Define image transformations
transform_left = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

transform_right = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

#Load the dataset
train_dataset = datasets.ImageFolder(root = os.path.sep.join([config.DATASET_PATH, config.TRAIN]),
                                     transform = transform_left)

test_dataset = datasets.ImageFolder(root = os.path.sep.join([config.DATASET_PATH, config.TEST]),
                                    transform = transform_left)

#Divide dataset into batches
batch_size = 10
train_load = torch.utils.data.DataLoader(dataset = train_dataset,
                                         batch_size = batch_size,
                                         shuffle = True)

test_load = torch.utils.data.DataLoader(dataset = test_dataset,
                                        batch_size = batch_size,
                                        shuffle = False)

#Define the model, the loss function and the optimizer
model = models.resnet18()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

#Define the lists to store the results of loss and accuracy
train_loss = []
test_loss = []
train_accuracy = []
test_accuracy = []

num_epochs = 50
for epoch in range(num_epochs):

    start = time.time()

    # # # # #  T R A I N I N G  # # # # #

    #Reset these below variables to 0 at the begining of every epoch
    correct = 0
    iterations = 0
    iter_loss = 0.0

    #Put the network into training mode
    model.train()

    for i, (inputs, labels) in enumerate(train_load):

        # Convert torch tensor to Variable
        inputs = Variable(inputs)
        labels = Variable(labels)

        # If we have GPU, shift the data to GPU
        CUDA = torch.cuda.is_available()
        if CUDA:
            inputs = inputs.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()           # Clear off the gradient in (w = w - gradient)
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        iter_loss += loss.data.item()   # Accumulate the loss
        loss.backward()                 # Backpropagation
        optimizer.step()                # Update the weights

        # Record the correct predictions for training data
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum()
        iterations += 1

    # Record the training loss and training accuracy
    train_loss.append(iter_loss/iterations)
    train_accuracy.append((100 * correct / len(train_dataset)))


    # # # # #  T E S T I N G  # # # # #

    correct = 0
    iterations = 0
    iter_loss = 0.0

    #Put the network into evaluation/testing mode
    model.eval()

    for i, (inputs, labels) in enumerate(test_load):

        inputs = Variable(inputs)
        labels = Variable(labels)

        CUDA = torch.cuda.is_available()
        if CUDA:
            inputs = inputs.cuda()
            labels = labels.cuda()

        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        iter_loss += loss.data.item()

        # Record the correct predictions for testing data
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum()
        iterations += 1

    # Record the testing loss and testing accuracy
    test_loss.append(loss/iterations)
    test_accuracy.append((100 * correct / len(test_dataset)))
    stop = time.time()

    print('Epoch {}/{}, Training Loss: {:.3f}, Training Accuracy: {:.3f}, Testing Loss: {:.3f}, Testing Acc: {:.3f}, Time: {}s'
          .format(epoch+1, num_epochs, train_loss[-1], train_accuracy[-1], test_loss[-1], test_accuracy[-1], stop-start))

    gc.collect()

#Save the model
dirPath = os.path.sep.join([config.OUTPUT_PATH, str(config.TIME)])
if not os.path.exists(dirPath):
	os.makedirs(dirPath)
torch.save(model.state_dict(), os.path.sep.join([dirPath, 'model.pth']))

#Load the model
# model.load_state_dict(torch.load('model.pth'))