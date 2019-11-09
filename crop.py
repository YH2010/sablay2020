import os
import sys

import cv2
import imutils
import numpy as np

def resizeToHeight(image):
    return imutils.resize(image, height=500)

classes = ['A','C','D','G','H','M','N']
for c in classes:

    print(c)
    dirPath = os.path.sep.join(['dataset', 'classes_cropped', c])
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)

    files = os.listdir(os.path.sep.join(['dataset', 'classes', c]))
    for f in files:
        try:
            orig_img = cv2.imread(os.path.sep.join(['dataset', 'classes_cropped', c, f]))
            g_channel = orig_img[:,:,1]
            _, thresh_img = cv2.threshold(g_channel,15,255,cv2.THRESH_BINARY)

            edged = cv2.Canny(thresh_img, 100, 255)
            contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = np.concatenate(contours)
            x,y,w,h = cv2.boundingRect(contours)

            # g_channel_bound = cv2.rectangle(g_channel,(x,y,w,h),255,5)
            # g_channel_bound = resize(g_channel_bound.get())
            # g_channel = cv2.copyMakeBorder(g_channel, 0, 0, 0, 5, cv2.BORDER_CONSTANT, value=255)

            g_channel_crop = g_channel[y:y+h, x:x+w]

            # output = np.hstack((g_channel_bound, resize(thresh_img), resize(g_channel_crop)))

            # cv2.imshow('',img3)
            # key = cv2.waitKey(0)
            # if key == ord('/'):
            #     sys.stdout.write(f)
            #     input()

            cv2.imwrite(os.path.sep.join(['dataset', 'classes_cropped', c, f]),resizeToHeight(g_channel_crop))

        except Exception as e:
            print(f, e)


# orig_img = cv2.imread('dataset/classes/C/2126_left.jpg')
# g_channel = orig_img[:,:,1]
# _, thresh_img = cv2.threshold(g_channel,15,255,cv2.THRESH_BINARY)

# edged = cv2.Canny(thresh_img, 100, 255)
# contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# contours = np.concatenate(contours)
# x,y,w,h = cv2.boundingRect(contours)

# # edged = cv2.Canny(thresh_img, 100, 255)
# # contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# # contours = np.concatenate(contours)
# # (x2,y2,w2,h2) = cv2.boundingRect(contours)

# # if(w1*h1 >= w2*h2):
# #     x,y,w,h = x1,y1,w1,h1
# # else:
# #     x,y,w,h = x2,y2,w2,h2

# g_channel_bound = cv2.rectangle(g_channel,(x,y,w,h),255,5)
# g_channel_bound = resize(g_channel_bound.get())
# # g_channel = cv2.copyMakeBorder(g_channel, 0, 0, 0, 5, cv2.BORDER_CONSTANT, value=255)

# g_channel_crop = g_channel[y:y+h, x:x+w]

# output = np.hstack((g_channel_bound, resize(thresh_img), resize(g_channel_crop)))

# cv2.imshow('',output)
# key = cv2.waitKey(0)
# if key == ord('/'):
#     sys.stdout.write(f)
#     input()
