
import torch
import torch.nn as nn

import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

#import adabound

import gc, time, os, sys
import cv2
import numpy as np
import matplotlib as plt

from helpers import config
from PIL import Image
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler

#Define the batch size, the model, the loss function and the optimizer
batch_size = 32
device = 1
if torch.cuda.is_available():
    torch.cuda.set_device(device)

class ImageFolderRevised(datasets.ImageFolder):
    # override the __getitem__ method that dataloader calls
    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = cv2.imread(path)
        img = img[:,:,1]
        img = cv2.equalizeHist(img)
        img = cv2.resize(img, (224, 224))
        img = img.reshape((1, 224, 224))
        img = torch.tensor(img, dtype=torch.float)
        # img = Image.fromarray(img)
        return img, target

dataset = ImageFolderRevised(root = os.path.sep.join([config.DATASET_DATASET_PATH,"Oct9"]))

test_indices = [1318, 4273, 5190, 5184, 2329, 1953, 4248, 3182, 429, 716, 2085, 4661, 3257, 4850, 2241, 648, 4669, 2308, 2854, 2516, 2194, 1624, 1121, 4382, 1354, 885, 1556, 215, 3699, 4420, 629, 4860, 1333, 210, 5483, 530, 2984, 2444, 4970, 737, 187, 3630, 4551, 3736, 2220, 1442, 4633, 3560, 256, 3603, 1714, 5324, 3050, 5406, 3824, 1392, 3696, 2916, 4142, 4374, 4817, 3453, 3417, 4460, 117, 2264, 2639, 888, 4283, 1622, 1408, 3281, 5045, 4341, 2108, 1239, 1636, 1617, 1402, 681, 3650, 2135, 2370, 1558, 809, 273, 103, 5245, 3531, 440, 814, 4607, 176, 5260, 4871, 4779, 4655, 1589, 2578, 1660, 107, 428, 3781, 719, 3241, 592, 624, 808, 3014, 1045, 104, 4287, 1673, 2125, 1241, 4405, 5257, 466, 1262, 349, 2171, 2740, 137, 507, 1369, 1225, 596, 5052, 264, 609, 3739, 1111, 1422, 3847, 434, 655, 1551, 3669, 5152, 2843, 4973, 3563, 611, 4577, 5206, 2594, 1264, 694, 5464, 4013, 5436, 5143, 432, 3526, 209, 454, 359, 3959, 1155, 5299, 1024, 1452, 604, 5093, 3588, 5481, 4533, 1365, 4683, 4608, 3883, 573, 4403, 3098, 540, 5105, 186, 199, 4489, 61, 587, 4253, 4141, 4107, 1214, 2219, 2587, 3159, 5049, 1474, 1227, 2668, 470, 5394, 4712, 5520, 3287, 1907, 4070, 921, 3002, 4921, 906, 1255, 4994, 3036, 4709, 926, 449, 2350, 5318, 3408, 2620, 1849, 227, 1410, 749, 1602, 305, 4109, 5233, 3436, 2387, 450, 1798, 3134, 442, 2761, 1697, 5229, 1772, 96, 2768, 416, 2253, 3763, 3084, 5439, 5005, 5321, 3586, 5387, 4314, 4522, 3672, 499, 3518, 965, 1646, 4096, 5226, 5092, 4694, 4928, 4786, 2004, 1088, 3143, 211, 5300, 2938, 338, 271, 3871, 3857, 905, 1792, 3865, 752, 657, 4055, 280, 4751, 1944, 5345, 1250, 2927, 4770, 775, 124, 80, 3677, 5251, 4331, 32, 1366, 1618, 5107, 4294, 3007, 284, 838, 5338, 1931, 3591, 4309, 1512, 3103, 3165, 1807, 1666, 4476, 3850, 3970, 5133, 3044, 4902, 5088, 3861, 3919, 293, 1903, 1058, 251, 4519, 1898, 5476, 315, 4907, 4445, 4528, 3189, 2749, 3078, 3944, 2806, 2250, 175, 741, 3212, 1299, 2337, 1097, 4044, 1382, 2742, 2905, 3567, 1640, 4217, 2394, 4175, 1105, 2866, 2203, 682, 17, 3081, 4510, 28, 4513, 2297, 2031, 1829, 4366, 601, 2887, 942, 3823, 4494, 2206, 3844, 4327, 4124, 1943, 2377, 2022, 1605, 2334, 4425, 1034, 4442, 4478, 2969, 2832, 3884, 5151, 4012, 1561, 4815, 2027, 3378, 3358, 2013, 345, 1913, 5511, 801, 1534, 3790, 3887, 1985, 2611, 542, 4502, 2821, 1533, 4361, 4440, 3745, 4235, 3788, 2432, 207, 1150, 1400, 1191, 3728, 2788, 1745, 991, 4985, 1335, 1522, 2075, 5408, 1507, 1179, 5499, 3579, 4600, 714, 2209, 4243, 1642, 907, 2186, 1824, 3595, 4177, 2321, 3062, 2273, 258, 3158, 1020, 3749, 5201, 1963, 1519, 4898, 4816, 2970, 1627, 2447, 3701, 3550, 468, 4394, 5378, 4318, 2813, 3345, 3100, 2983, 1075, 2002, 1154, 2939, 2513, 1872, 2487, 873, 992, 1835, 4104, 3573, 1889, 302, 3702, 1127, 3930, 3297, 4029, 1683, 821, 366, 4132, 2699, 867, 2925, 1517, 3377, 3808, 3520, 3331, 5504, 3615, 1186, 2477, 4212, 1696, 1912, 2816, 1848, 4934, 2076, 790, 1000, 1388, 249, 1498, 5505, 1203, 39, 1223, 3581, 4666, 2016, 323, 2951, 2348, 4241, 2293, 2420, 2919, 5181, 625, 2079, 3812, 2430, 3698, 3530, 2836, 4257, 4471, 1125, 2893, 1986, 1968, 122, 2800, 5013, 4061, 4043, 2933, 613, 1302, 523, 382, 4719, 369, 2335, 3693, 618, 811, 4288, 4234, 1079, 5525, 3984, 5477, 5168, 2824, 7, 4881, 842, 595, 3344, 1601, 2228, 40, 1826, 3933, 3571, 3004, 4979, 2789, 2398, 536, 5302, 482, 5174, 1876, 2809, 445, 158, 2679, 490, 165, 206, 3934, 3261, 3480, 3830, 4673, 3478, 2940, 2023, 1348, 4159, 1135, 929, 5480, 561, 218, 4393, 1817, 2654, 1764, 3299, 1314, 299, 3363, 4539, 1210, 2386, 2426, 3507, 5424, 2630, 5467, 3315, 2607, 3278, 394, 1401, 3976, 1236, 4828, 1441, 2474, 5222, 3330, 2385, 1679, 3125, 4239, 3471, 4647, 1665, 3027, 3874, 4944, 3366, 1983, 1471, 5003, 1827, 1821, 2974, 3488, 2903, 24, 4454, 961, 2962, 5419, 2235, 5495, 4413, 4971, 2448, 4865, 1162, 4434, 4057, 2237, 1446, 4152, 2529, 527, 1819, 2105, 615, 3756, 1955, 1822, 818, 5028, 3313, 4778, 3882, 5194, 4933, 82, 4161, 5232, 4908, 2593, 1606, 2842, 2057, 823, 3398, 71, 4112, 4475, 5304, 5084, 4582, 3070, 915, 1838, 2453, 1658, 1414, 1416, 4780, 3703, 3015, 2160, 3599, 1747, 2091, 4941, 4629, 2525, 4204, 3252, 3671, 5091, 2707, 4242, 1596, 4668, 754, 539, 4649, 2698, 1394, 5006, 3396, 2909, 4810, 850, 4798, 4284, 1287, 3061, 5149, 5487, 2422, 4114, 119, 3593, 4077, 1472, 556, 3524, 115, 2966, 2561, 951, 4916, 425, 2600, 1539, 205, 1146, 5347, 1005, 457, 2409, 4812, 1750, 4103, 4952, 407, 2612, 4198, 5488, 5486, 1025, 972, 3777, 1790, 2214, 4377, 1922, 2783, 4308, 1164, 1102, 2062, 426, 488, 4597, 4083, 4532, 950, 4027, 1575, 664, 5148, 1663, 1100, 872, 25, 1899, 4657, 5108, 229, 1661, 2608, 1306, 2247, 1324, 4634, 3626, 3108, 4252, 4491, 3880, 4923, 5399, 4129, 1509, 5012, 4957, 1487, 2421, 2989, 3935, 3923, 822, 5295, 1756, 1563, 2726, 4614, 2482, 5090, 545, 4134, 1014, 4115, 3572, 4783, 2060, 5046, 4433, 3811, 3787, 2371, 1964, 3298, 1555, 2674, 5198, 2660, 931, 3651, 644, 2504, 2635, 5263, 168, 1893, 839, 3205, 1195, 2495, 4024, 5451, 3051, 2624, 2248, 4749, 1145, 2375, 2535, 4788, 2383, 1273, 413, 4878, 2803, 4873, 3333, 4756, 5513, 417, 3614, 4737, 4505, 3762, 3987, 2766, 2327, 949, 3751, 360, 2628, 553, 116, 62, 1581, 2544, 220, 1173, 4844, 2616, 1109, 4562, 1884, 4586, 3323, 4956, 3450, 4155, 2025, 1087, 1018, 586, 2865, 3760, 3219, 4110, 479, 4444, 2636, 5388, 2212, 4764, 638, 4304, 2551, 4158, 3993, 2553, 2914, 4938, 4001, 1608, 1577, 102, 4691, 321, 1056, 5212, 4069, 1043, 1853, 4058, 2610, 3441, 5320, 4866, 1327, 3983, 1120, 261, 4717, 3719, 347, 5209, 1942, 1722, 3028, 833, 1265, 1086, 1249, 5101, 1196, 506, 1710, 5225, 2280, 4180, 3445, 465, 4429, 874, 4627, 114, 5253, 1803, 2649, 453, 45, 2330, 5370, 2524, 3778, 5004, 3594, 589, 1892, 824, 2072, 3718, 2275, 656, 1395, 908, 1769, 3846, 1886, 5472, 4280, 3967, 331, 1961, 1233, 2717, 1732, 668, 4726, 1455, 4369, 3334, 5400, 989, 3803, 4948, 837, 1768, 4752, 5469, 2339, 2142, 4542, 1833, 969, 4852, 4765, 2037, 2204, 5065, 3764, 825, 4305, 3045, 3602, 4761, 3635, 3422, 2571, 1518, 5126, 5350, 3973, 3440, 1344, 118, 376, 150, 2278, 4446, 2753, 2835, 283, 1993, 4047, 1457, 5322, 3504, 3376, 3892, 1919, 1258, 3765, 170, 3386, 2162, 3367, 3131, 2760, 3821, 3809, 1758, 3807, 861, 2823, 2687, 2287, 4514, 3348, 5172, 3908, 2093, 2256, 4791, 1894, 1989, 2897, 769, 5353, 2449, 3577, 343, 3235, 2519, 3431, 2161, 232, 2669, 4515, 4076, 437, 943, 631, 4742, 3774, 372, 3898, 672, 4424, 3990, 322, 1245, 3831, 2696, 5191, 1449]

test_load = torch.utils.data.DataLoader(dataset = dataset,
                                        batch_size = batch_size,
                                        sampler = SubsetRandomSampler(test_indices))

# # # # #  T E S T I N G  # # # # #
# model = models.resnet50(pretrained=True)
# model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
model = models.vgg19_bn(num_classes=4)
model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1) # for VGG
class_weights = torch.FloatTensor([20.28,3.53,19.47,1.0]).cuda()
loss_fcn = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

model_name = input()
model.load_state_dict(torch.load('output/102019063735/'+model_name))
# model.load_state_dict(torch.load('output/100919232206/model_150.pth'))

#Put the network into evaluation/testing mode
model.eval()

correct = 0
iter_loss = 0.0

confusion_matrix = torch.zeros(len(dataset.classes), len(dataset.classes))

for i, (inputs, labels) in enumerate(test_load):

    inputs = Variable(inputs)
    labels = Variable(labels)

    if torch.cuda.is_available():
        model.cuda()
        inputs = inputs.cuda()
        labels = labels.cuda()

    optimizer.zero_grad()
    outputs = model(inputs)
    _, predicted = torch.max(outputs, 1)
    loss = loss_fcn(outputs, labels)

    iter_loss += loss.data.item()

    correct += (predicted == labels).sum()

    # sys.stdout.write(str(i))
    # sys.stdout.flush()

    for i in range(len(predicted)):
        confusion_matrix[int(labels[i])][int(predicted[i])] += 1

# Record the testing loss and testing accuracy
test_loss = iter_loss / len(test_load)
test_accuracy = 100 * correct / len(test_indices)

print(dataset.classes)
print(confusion_matrix)

sys.stdout.write('Testing Loss: {:.3f}, Testing Accuracy: {:.3f}\n'
        .format(test_loss, test_accuracy))
sys.stdout.flush()
