
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

# test_indices = [693, 2364, 1973, 2283, 5416, 183, 4788, 5377, 5387, 4732, 5211, 875, 2638, 4214, 489, 4358, 4721, 4009, 4196, 1569, 5522, 3340, 1260, 1032, 3099, 4855, 617, 5259, 4976, 4913, 4736, 2166, 3192, 4198, 3398, 3584, 2056, 4154, 725, 1613, 2078, 1669, 2376, 1169, 4395, 2165, 1592, 1532, 2768, 1871, 2269, 496, 844, 4809, 4498, 4955, 2411, 1369, 3774, 1069, 2718, 4000, 4969, 3148, 4709, 5325, 4099, 3833, 2956, 3607, 303, 2196, 3813, 2349, 5095, 4915, 4107, 772, 1876, 503, 951, 887, 1034, 2463, 1213, 4957, 4465, 974, 1630, 2913, 362, 4474, 1804, 3593, 768, 2995, 1918, 4820, 4756, 2493, 1113, 4909, 5279, 4294, 5341, 3836, 3931, 1872, 1969, 3310, 5056, 999, 4831, 2356, 2436, 3125, 5372, 995, 3303, 3039, 3672, 5502, 1929, 233, 33, 5183, 451, 466, 4243, 1139, 1145, 2500, 3390, 3957, 4433, 2610, 1503, 1059, 3387, 3708, 4666, 1657, 143, 3666, 4959, 754, 3403, 587, 2664, 4996, 515, 21, 929, 765, 914, 1819, 2974, 3901, 2852, 1927, 518, 1400, 3734, 346, 1231, 4654, 1402, 2862, 1777, 3743, 5009, 2919, 4306, 1481, 4424, 3392, 4096, 5464, 4554, 4834, 1595, 2655, 1964, 2685, 1148, 564, 1724, 2091, 3444, 4896, 1897, 3270, 5427, 5074, 2155, 2705, 699, 236, 3682, 5077, 3998, 3644, 5340, 2015, 5052, 2035, 123, 377, 4632, 4033, 1983, 487, 2519, 3312, 5001, 4412, 3434, 5144, 270, 3439, 1295, 4219, 3250, 4121, 1058, 1365, 1550, 583, 5292, 4836, 3029, 4239, 2839, 2525, 253, 1676, 2711, 4271, 5229, 2222, 3158, 5230, 5436, 4797, 742, 2743, 5509, 4534, 955, 1654, 4015, 5251, 2669, 3856, 5504, 4937, 1330, 3566, 3154, 1981, 674, 399, 1253, 864, 525, 3196, 2871, 2624, 2450, 1891, 2318, 2512, 4252, 4463, 4414, 944, 2116, 2629, 5154, 1751, 4949, 4127, 4509, 4383, 2164, 3445, 3114, 4718, 5020, 5142, 1055, 2470, 2527, 449, 1698, 508, 3418, 5116, 5519, 3920, 2814, 2809, 2590, 2566, 2005, 392, 2492, 3316, 4680, 4339, 766, 5446, 2405, 5469, 268, 2854, 1974, 1449, 2609, 3623, 1631, 2163, 4488, 273, 3885, 146, 2390, 2672, 3506, 2967, 2467, 1563, 3698, 668, 1834, 1482, 3412, 2958, 3872, 1758, 4484, 3787, 3659, 3003, 1601, 2783, 3232, 4747, 5399, 1588, 4940, 5263, 4300, 4277, 4026, 266, 4215, 4544, 3058, 1672, 1053, 1027, 291, 443, 535, 365, 3996, 4805, 613, 1293, 2236, 2, 5376, 3564, 5021, 2391, 3089, 862, 5076, 2381, 3923, 5406, 5417, 2883, 4796, 3939, 350, 495, 2808, 3536, 4515, 5517, 4057, 3663, 3959, 2589, 4042, 1459, 1609, 5398, 963, 2568, 3247, 5094, 1902, 4233, 945, 5238, 4356, 3729, 1202, 124, 1735, 2305, 2652, 1674, 2114, 1616, 1487, 1219, 4656, 1838, 50, 5176, 1063, 4004, 499, 1833, 2464, 3465, 3129, 2337, 3282, 671, 1399, 4783, 3760, 4459, 3646, 5, 3419, 221, 2692, 1221, 706, 1995, 2345, 214, 306, 830, 3354, 4866, 1508, 5156, 4559, 1753, 1828, 1768, 1270, 905, 2237, 4167, 2934, 4371, 808, 44, 3205, 3873, 2028, 4990, 5370, 3088, 3936, 536, 993, 2648, 1021, 375, 2487, 335, 3473, 2354, 3135, 1346, 5133, 2635, 1733, 4190, 4980, 4011, 507, 3517, 975, 1412, 690, 5007, 309, 1545, 5357, 558, 1568, 2887, 1463, 2954, 3334, 3619, 835, 3902, 1647, 3199, 2479, 4092, 4847, 2202, 4516, 1378, 3661, 2573, 996, 4593, 2775, 5287, 1880, 1172, 5275, 4851, 1111, 2453, 3253, 6, 5421, 4376, 4806, 1318, 1529, 519, 3381, 2534, 3023, 1084, 647, 4345, 5151, 745, 4082, 3115, 1691, 4261, 2966, 5453, 954, 4071, 485, 3882, 1306, 5480, 192, 4975, 3067, 3502, 588, 4303, 1862, 5068, 3030, 2764, 705, 4646, 4260, 3967, 3700, 5511, 3386, 162, 4291, 2804, 467, 1809, 1243, 5140, 596, 1760, 34, 5536, 4067, 4019, 2438, 2595, 5297, 3846, 3331, 1932, 4530, 136, 1283, 1078, 1690, 4427, 4972, 1469, 4281, 4030, 1433, 4024, 1585, 3892, 5459, 937, 5299, 4550, 358, 2547, 5018, 71, 5411, 326, 2583, 2182, 4890, 3309, 3830, 4001, 1783, 258, 2219, 2603, 5022, 5210, 1597, 4415, 2259, 3636, 3772, 5285, 4921, 1831, 4791, 4920, 2322, 1257, 576, 630, 4887, 3848, 1991, 3660, 4566, 4438, 275, 4441, 490, 2611, 3879, 2938, 817, 928, 849, 1581, 2235, 547, 3556, 4894, 5193, 2761, 3326, 8, 1709, 1161, 769, 1694, 2560, 1750, 3420, 4982, 99, 5272, 2842, 1678, 213, 1422, 2691, 4249, 4984, 3545, 3428, 2781, 3363, 5040, 4069, 1744, 3990, 2815, 3200, 2802, 2948, 1661, 1755, 5508, 3330, 3162, 3021, 1008, 2239, 3271, 747, 2977, 5368, 677, 1385, 3678, 5498, 4005, 2695, 1953, 4625, 2600, 1007, 5389, 5434, 1209, 2799, 2825, 2386, 1795, 169, 4616, 757, 5205, 5187, 3263, 2939, 1612, 5207, 74, 5227, 612, 1417, 1370, 4089, 1960, 2916, 2850, 5443, 1778, 1701, 3258, 4686, 958, 1386, 2455, 4870, 3685, 206, 5438, 1947, 4611, 4822, 4240, 4782, 204, 2727, 856, 1425, 180, 322, 4861, 1552, 3527, 1564, 746, 3941, 3878, 889, 4768, 3262, 695, 1992, 2594, 440, 728, 340, 5042, 1479, 3295, 3375, 2627, 4337, 2369, 5169, 3768, 4978, 2151, 3874, 526, 2803, 682, 969, 4017, 4653, 3220, 1544, 1704, 4814, 1512, 1822, 3548, 2870, 1603, 1996, 2502, 215, 1329, 2865, 676, 3293, 5225, 4266, 3492, 5439, 4248, 4586, 2698, 1162, 2746, 1434, 3161, 3580, 3513, 1979, 3604, 373, 1480, 5256, 4945, 3045, 1886, 3582, 5520, 4507, 2037, 2507, 989, 4408, 959, 2994, 998, 1395, 3315, 3016, 151, 1068, 3160, 1342, 724, 3327, 1279, 1854, 3771, 3219, 199, 3735, 5348, 2291, 2087, 3356, 3394, 3203, 4966, 3945, 1586, 5488, 592, 4922, 3691, 378, 210, 2523, 2257, 3820, 1017, 293, 3715, 4657, 5489, 1655, 2274, 1775, 298, 3533, 5130, 4776, 1517, 967, 3924, 2472, 436, 867, 3005, 3153, 2431, 3324, 557, 4377, 1946, 940, 1104, 2949, 4829, 4925, 2968, 1489, 5423, 2143, 2608, 3180, 1687, 3292, 4560, 4663, 5493, 3223, 2591, 4116, 5105, 528, 3287, 4443, 4713, 3912, 666, 2703, 4944, 3272, 4179, 5113, 1764, 2396, 3522, 763, 4081, 2398, 4838, 1621, 3714, 347, 4437, 4643, 2084, 3254, 2426, 2280, 4981, 2342, 2666, 574, 5310, 3204, 2115, 200, 4061, 404, 4382, 1153, 1250, 3485, 286, 3910, 5420, 1245, 331, 4314, 1131, 791, 3918, 3572, 3960, 3552, 3472, 5108, 1827, 1912, 1065, 159, 4037, 2759, 1810, 458, 3986, 1101, 4147, 1373, 315, 3010, 5429, 1952, 344, 4637, 2081, 2824, 4589, 1020, 946, 1305, 1535, 222, 4491, 3104, 2146, 770, 627, 1442, 3078, 4704, 5380, 1858, 1802, 1658, 3243, 4297, 59, 473, 3122, 731, 4145, 455, 4886, 3780, 1246, 1151, 3116, 1623, 949, 4902, 4615, 26, 687, 4800, 1135, 4580, 4457, 3475, 3140, 1560, 1794, 787, 98, 1659, 1335, 5121, 546, 1028, 1273, 4974, 3456, 1082, 721, 727, 85, 4360, 3516, 3625, 5066, 36, 831, 2109, 3831, 5226, 1428, 5030, 116, 2447, 401, 771, 1097, 4630, 4160, 2293, 360, 3118, 1483, 1185, 2025, 4617, 3790, 388, 1590, 1573, 1136, 2483, 4548, 2247, 953, 4881, 225, 3501, 3810, 154, 1584, 1922, 3963, 2120, 1695, 134, 5264, 1149, 2397, 49]
test_indices = [7, 10, 18, 11]

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
model.load_state_dict(torch.load('output/100919232206/'+model_name))

#Put the network into evaluation/testing mode
model.eval()

correct = 0
iter_loss = 0.0

confusion_matrix = torch.zeros(len(dataset.classes), len(dataset.classes))

for i, (inputs, labels) in enumerate(test_load):

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
        confusion_matrix[int(predicted[i])][int(labels[i])] += 1

# Record the testing loss and testing accuracy
test_loss = iter_loss / len(test_load)
test_accuracy = 100 * correct / len(test_indices)

print(dataset.classes)
print(confusion_matrix)

sys.stdout.write('Testing Loss: {:.3f}, Testing Accuracy: {:.3f}\n'
        .format(test_loss, test_accuracy))
sys.stdout.flush()
