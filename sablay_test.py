
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

test_indices = [2071, 2475, 4627, 3205, 4594, 632, 104, 4708, 139, 4868, 1503, 4703, 2046, 4250, 5239, 3822, 3394, 4280, 963, 723, 1172, 4199, 3830, 3509, 3274, 495, 2545, 4252, 995, 3075, 4733, 2762, 3811, 4097, 3884, 232, 1673, 2804, 2957, 4660, 4388, 3156, 44, 1323, 3882, 808, 5211, 311, 2313, 154, 4911, 2505, 3453, 4613, 1935, 55, 294, 4537, 3507, 1177, 4884, 719, 4173, 3404, 1939, 4554, 110, 531, 3647, 470, 3675, 391, 3167, 890, 3896, 4332, 3473, 2589, 2862, 691, 3102, 1513, 4152, 187, 4410, 1422, 5013, 3354, 3098, 4118, 152, 815, 627, 4772, 1992, 158, 1421, 1247, 1514, 3648, 4432, 4203, 5243, 774, 3850, 1014, 178, 1626, 4538, 793, 4623, 3249, 3633, 4133, 1073, 4322, 1431, 4456, 2098, 883, 4748, 3629, 821, 2305, 1678, 1556, 4855, 5413, 5151, 4010, 5378, 527, 4318, 611, 1049, 4815, 1959, 5086, 2795, 1366, 4842, 1315, 552, 5029, 1125, 1572, 3962, 3723, 2693, 4697, 2468, 1575, 1361, 2659, 1400, 2332, 5330, 2787, 3296, 2243, 2790, 2489, 4494, 1779, 1909, 3939, 3816, 1210, 5416, 2838, 3950, 5291, 2018, 3752, 4323, 3121, 1552, 1158, 2637, 457, 171, 2476, 2540, 1094, 733, 1165, 4615, 2214, 1251, 712, 3356, 2363, 5379, 3745, 5101, 4390, 2011, 142, 5055, 730, 1613, 3456, 3684, 4029, 5509, 2898, 4959, 3708, 1475, 3192, 3358, 583, 117, 54, 4755, 1813, 4027, 5307, 4608, 2065, 1328, 362, 1065, 4539, 4774, 5266, 3944, 3729, 833, 1826, 4805, 1873, 2039, 4344, 1059, 2837, 2988, 809, 39, 628, 339, 1386, 1748, 1506, 1677, 1897, 3874, 4747, 2694, 683, 4822, 4085, 3042, 4401, 5449, 1467, 2166, 3031, 3848, 2494, 60, 3308, 5123, 1458, 429, 2002, 3207, 275, 2268, 153, 3108, 4645, 4576, 3023, 2581, 1128, 3139, 2092, 327, 4888, 316, 1273, 2266, 473, 4644, 2709, 270, 1720, 4586, 1739, 1502, 1870, 4744, 1978, 3504, 862, 775, 4244, 571, 623, 3471, 1500, 1760, 5301, 4945, 5360, 446, 0, 3493, 2352, 1200, 258, 4511, 102, 3184, 3360, 5474, 3760, 3103, 364, 5066, 77, 5065, 3995, 5369, 1586, 261, 1879, 3899, 269, 4454, 4462, 2042, 2842, 2575, 5071, 4440, 868, 4044, 2221, 3923, 1277, 2005, 5257, 4808, 3964, 416, 2354, 223, 4384, 199, 2641, 2972, 4600, 4235, 464, 1409, 1962, 5040, 5150, 2728, 486, 3256, 2997, 4386, 1806, 647, 2568, 109, 1055, 3217, 5117, 3635, 2048, 2796, 4094, 2840, 1448, 5420, 3476, 4184, 1562, 3609, 1837, 3104, 126, 978, 1563, 1491, 3724, 4198, 2721, 4512, 3532, 874, 2404, 3954, 2410, 567, 3142, 3465, 2205, 4585, 4025, 4672, 5485, 2913, 4683, 957, 912, 1398, 2119, 1423, 1560, 1570, 1905, 4342, 3902, 4171, 4153, 5080, 1949, 1298, 4792, 5288, 3089, 3689, 1060, 397, 1074, 3250, 1347, 4825, 1308, 1532, 1114, 2590, 3432, 4517, 3481, 915, 3768, 3852, 3844, 1669, 645, 3904, 2628, 1456, 3073, 4790, 2534, 2843, 2772, 4299, 2291, 3595, 941, 4803, 2331, 1900, 2261, 4249, 1209, 3227, 842, 1857, 221, 578, 4934, 2654, 501, 532, 767, 1349, 3137, 785, 2626, 2546, 5176, 4392, 639, 1601, 4651, 4212, 1858, 4465, 3843, 2147, 4798, 375, 4579, 1662, 3508, 3270, 2664, 562, 1479, 2120, 4026, 502, 3545, 1611, 1643, 4910, 3003, 5433, 5348, 5202, 2345, 724, 3762, 3974, 428, 4916, 5312, 3935, 2778, 2582, 1404, 959, 1219, 4071, 2716, 4341, 4269, 4234, 4686, 1928, 2574, 3029, 1869, 3533, 2370, 5295, 209, 2108, 3208, 3065, 4515, 2382, 4524, 3581, 4054, 1689, 5368, 5005, 2631, 90, 1396, 3298, 2182, 2883, 3248, 677, 384, 3105, 543, 3698, 2791, 1719, 1443, 3325, 3138, 4426, 3709, 2835, 4346, 2739, 607, 322, 4021, 4835, 40, 3049, 5411, 3475, 4580, 5246, 2715, 236, 2902, 1124, 2274, 3482, 2684, 1203, 3823, 643, 961, 3100, 4926, 3487, 4009, 840, 5496, 2189, 3696, 2371, 3498, 4165, 4176, 3583, 1888, 1051, 5350, 4189, 2032, 1494, 2891, 367, 3472, 5476, 1653, 3014, 2971, 4778, 1612, 313, 1593, 2857, 1753, 3985, 5033, 3658, 816, 4385, 1437, 1790, 2, 4828, 5264, 4233, 1002, 2813, 2013, 2277, 2030, 2150, 4217, 1916, 5404, 3161, 4941, 2947, 3170, 2769, 2689, 5056, 4372, 2876, 4592, 67, 5328, 1984, 3542, 3034, 2069, 1650, 3866, 5166, 1816, 2077, 4038, 4857, 1755, 1453, 1090, 1490, 198, 1484, 227, 3328, 3146, 1457, 4839, 134, 4712, 5297, 1394, 160, 3496, 1435, 3348, 2861, 967, 742, 1865, 1373, 2915, 3637, 1175, 3955, 3083, 377, 1195, 1190, 3603, 4434, 3640, 3602, 1432, 4954, 4806, 4662, 3246, 5260, 820, 5311, 3989, 735, 1894, 5527, 1044, 699, 4082, 1588, 2914, 4470, 3555, 5346, 1156, 4510, 1933, 3333, 1820, 4684, 2104, 2895, 2060, 3134, 1584, 1917, 2286, 5226, 4785, 1171, 3004, 3808, 2998, 5388, 296, 1621, 2308, 4986, 21, 505, 3380, 3320, 2131, 1317, 3815, 3749, 2407, 1829, 2695, 165, 2679, 934, 3662, 3946, 3656, 5045, 4658, 53, 5402, 630, 1215, 1153, 3438, 5493, 443, 4905, 4416, 415, 790, 1915, 5136, 3620, 4992, 4545, 1245, 594, 5067, 2035, 2753, 22, 489, 1236, 4956, 2910, 4353, 222, 2225, 3466, 952, 2586, 1875, 1906, 2961, 82, 2072, 331, 3615, 2144, 3644, 4913, 1256, 4364, 1564, 3116, 3087, 453, 2815, 3372, 609, 2364, 3999, 1625, 1199, 456, 3740, 4328, 3033, 539, 4543, 2781, 3835, 1093, 5363, 2542, 5289, 3280, 2630, 3136, 4463, 1690, 1574, 2674, 1469, 2209, 5457, 1979, 5222, 3265, 3726, 1742, 2240, 850, 2731, 3833, 2412, 1252, 2629, 2562, 4578, 727, 1699, 321, 646, 374, 3959, 3021, 668, 247, 5483, 1228, 3502, 2571, 3242, 4370, 1327, 4300, 765, 1120, 4367, 3598, 3806, 4907, 1882, 574, 2362, 2230, 3091, 1821, 2528, 2017, 2031, 1083, 2074, 553, 3927, 427, 1357, 2206, 1998, 4944, 4991, 155, 557, 4903, 3953, 4754, 4718, 2846, 150, 935, 1185, 99, 1362, 761, 4716, 2929, 2359, 2720, 1391, 2073, 1111, 4947, 1295, 5183, 1023, 3743, 3606, 2917, 2663, 484, 4534, 1397, 3670, 2954, 1434, 4210, 1220, 2897, 4215, 5107, 1100, 3388, 5057, 5390, 3054, 1602, 3513, 4058, 129, 3052, 4230, 2980, 1948, 1708, 4867, 4277, 4045, 4893, 3540, 4043, 3157, 181, 5400, 2423, 1427, 2106, 5178, 744, 4123, 2219, 973, 2501, 1229, 5353, 2647, 2171, 3639, 138, 4490, 229, 2585, 4568, 120, 1718, 5533, 1878, 1173, 3159, 2493, 5081, 354, 3738, 2517, 4136, 3294, 687, 3129, 1684, 3677, 23, 4740, 38, 1968, 522, 3730, 3735, 4571, 3239, 3417, 4605, 5242, 917, 4999, 1703, 4267, 1011, 3854, 1477, 4756, 984, 3952, 2788, 4377, 4765, 2602, 586, 4656, 1006, 1946, 3020, 2126, 4474, 988, 4104, 3253, 3607, 1224, 2122, 2900, 5344, 5008, 5073, 312, 2142, 4516, 4455, 1622, 4529, 3289, 4942, 5218, 4724, 2963, 558, 1976, 5021, 4321, 3164, 3355, 2688, 1744, 3390, 2022, 1874, 3455, 1793, 4083, 4477, 414, 519, 540, 3226, 3938, 2220, 257, 1964, 663, 831, 629, 3303, 1108, 1543, 786, 147, 5025, 4046, 4007, 5477, 1270, 3180, 872, 939, 396, 168, 4188, 3342, 5339, 418, 4105, 4997, 4814, 4745, 4013, 3235, 30, 4919, 4448, 421, 2310, 4689]
# test_indices = [7, 10, 18, 11]

test_load = torch.utils.data.DataLoader(dataset = dataset,
                                        batch_size = batch_size,
                                        sampler = SubsetRandomSampler(test_indices))

# # # # #  T E S T I N G  # # # # #
# model = models.resnet50(pretrained=True)
# model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
model = models.vgg19_bn(pretrained=True)
model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1) # for VGG
loss_fcn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# model_name = input()
# model.load_state_dict(torch.load('output/101119143441/'+model_name))
model.load_state_dict(torch.load('output/100919232206/model_150.pth'))

#Put the network into evaluation/testing mode
model.eval()

correct = 0
iter_loss = 0.0

confusion_matrix = torch.zeros(len(dataset.classes), len(dataset.classes))

for i, (inputs, labels) in enumerate(test_load):

    print(i)

    inputs = Variable(inputs)
    labels = Variable(labels)

    if torch.cuda.is_available():
        torch.cuda.set_device(device)
        model.cuda()
        inputs = inputs.cuda()
        labels = labels.cuda()

    optimizer.zero_grad()
    outputs = model(inputs)
    _, predicted = torch.max(outputs, 1)
    loss = loss_fcn(outputs, labels)

    iter_loss += loss.data.item()

    correct += (predicted == labels).sum()

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
