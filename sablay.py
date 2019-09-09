import torch
import torch.nn as nn

import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

import gc, time, os, sys
import numpy as np

from helpers import config
from PIL import Image
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler

sys.stdout.write("Batch Size : 32\n")
sys.stdout.write("Model : VGG-19-BN\n")
sys.stdout.write("Loss Function : CrossEntropyLoss()\n")
sys.stdout.write("Optimizer : SGD()\n")
sys.stdout.write("Learning Rate : 0.01\n\n")

#Define the batch size, the model, the loss function and the optimizer
batch_size = 32
model = models.vgg19_bn()
loss_fcn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
device = 0

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

train_split, test_split, val_split = .6, .2, .2
split1 = int(np.floor(dataset_size * train_split))
split2 = split1 + int(np.floor(dataset_size * test_split))
train_indices, test_indices, val_indices = indices[:split1], indices[split1:split2], indices[split2:]

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

# num_epochs = 1000
# for epoch in range(num_epochs):

#     start = time.time()

#     # # # # #  T R A I N I N G  # # # # #

#     #Put the network into training mode
#     model.train()

#     #Reset these below variables to 0 at the begining of every epoch
#     correct = 0
#     iter_loss = 0.0

#     for i, (inputs, labels) in enumerate(train_load):

#         # Convert torch tensor to Variable
#         inputs = Variable(inputs)
#         labels = Variable(labels)

#         # If we have GPU, shift the data to GPU
#         if torch.cuda.is_available():
#             #model = nn.DataParallel(model)
#             torch.cuda.set_device(1)
#             model.cuda()
#             inputs = inputs.cuda()
#             labels = labels.cuda()

#         optimizer.zero_grad()           # Clear off the gradient in (w = w - gradient)
#         outputs = model(inputs)
#         _, predicted = torch.max(outputs, 1)
#         loss = loss_fcn(outputs, labels)
#         loss.backward()                 # Backpropagation
#         optimizer.step()                # Update the weights

#         iter_loss += loss.data.item()   # Accumulate the loss

#         # Record the correct predictions for training data
#         correct += (predicted == labels).sum()

#         #sys.stdout.write("Train %s/%s, Time:%ss\n" % (i+1, len(train_load), time.time()-start))
#         #sys.stdout.flush()

#     # Record the training loss and training accuracy
#     train_loss = iter_loss / len(train_load)
#     train_accuracy = 100 * correct / len(train_indices)


#     # # # # #  V A L I D A T I O N  # # # # #

#     #Put the network into evaluation/testing mode
#     model.eval()

#     correct = 0
#     iter_loss = 0.0

#     for i, (inputs, labels) in enumerate(val_load):

#         inputs = Variable(inputs)
#         labels = Variable(labels)

#         if torch.cuda.is_available():
#             #model = nn.DataParallel(model)
#             model.cuda()
#             inputs = inputs.cuda()
#             labels = labels.cuda()

#         optimizer.zero_grad()
#         outputs = model(inputs)
#         _, predicted = torch.max(outputs, 1)
#         loss = loss_fcn(outputs, labels)

#         iter_loss += loss.data.item()

#         correct += (predicted == labels).sum()

#         #sys.stdout.write("Validation %s/%s, Time:%ss\n" % (i+1, len(val_load), time.time()-start))
#         #sys.stdout.flush()

#     # Record the testing loss and testing accuracy
#     val_loss = iter_loss / len(val_load)
#     val_accuracy = 100 * correct / len(test_indices)
#     stop = time.time()

#     sys.stdout.write('Epoch {}/{}, Training Loss: {:.3f}, Training Accuracy: {:.3f}, Validation Loss: {:.3f}, Validation Accuracy: {:.3f}, Time: {}s\n'
#           .format(epoch+1, num_epochs, train_loss, train_accuracy, val_loss, val_accuracy, stop-start))
#     sys.stdout.flush()

#     if (epoch+1) % 50 == 0:
#         #Save the model
#         dirPath = os.path.sep.join([config.OUTPUT_PATH, str(config.TIME)])
#         if not os.path.exists(dirPath):
#             os.makedirs(dirPath)
#         torch.save(model.state_dict(), os.path.sep.join([dirPath, 'model_'+str(epoch+1)+'.pth']))

# Load the model
model = model.load_state_dict(torch.load('output/090719230802/model_200.pth'))

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

file_name = sys.argv[1]
image = Image.open(file_name)
image = transform(image).float()
image = Variable(image, requires_grad=True)
image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet

print(model(image))