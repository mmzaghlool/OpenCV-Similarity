import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
import glob

####### get time at start of the execution
time1 = time.time()

####### list of all resources
imgsComp = [
    '1LE.jpg',
    '50_1.jpg',
    '100_old.jpeg',
    '100_cap1.jpg',
    '100_cap2.jpg',
    # '100_cap3.jpg',
    # '100_1.jpeg',
    # '100_2.jpg',
    '100_3.jpg',
    # 'x.png',
]

# get dim. of image

####### SIFT
sift = cv2.xfeatures2d.SURF_create()
# sift = cv2.ORB_create(nfeatures=50000)

for i in imgsComp:
    ######## Read image
    comp = cv2.imread(i)
    imgRead = cv2.imread('100_cap3.jpg')

    ##### get dim. of image
    heightComp, widthComp, channelsy = comp.shape
    heightRead, widthRead, channelsx = imgRead.shape

    
    if widthRead > widthComp: 
        resizeWidth = widthComp
        resizeHeight = heightComp
    else: 
        resizeWidth = widthRead
        resizeHeight = heightRead

    print(resizeWidth)
    print(resizeHeight)


    ######### dimensions of resizing (Width, height)
    dim = (resizeWidth, resizeHeight)

    ######### resize image
    # imgRead = cv2.resize(imgRead, dim, interpolation = cv2.INTER_AREA)
    # comp = cv2.resize(comp, dim, interpolation = cv2.INTER_AREA)

    ##### detect all keypoints and descriptors 
    kp1, des1 = sift.detectAndCompute(comp, None)
    kp2, des2 = sift.detectAndCompute(imgRead, None)

    ##### Brute Force Matching
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key = lambda x: x.distance)
    matching_result = cv2.drawMatches(comp, kp1, imgRead, kp2, matches[:50], None, flags=2)
    

    # index_params = dict(algorithm=0, trees=5)
    # search_params = dict()
    # flann = cv2.FlannBasedMatcher(index_params, search_params)
    # matches = flann.knnMatch(des1, des2, k=2)

    ##### comp result
    lenKP1 = len(kp1)
    lenKP2 = len(kp2)
    lenMatches = len(matches)

    # number_keypoints = 0
    # if lenKP1 <= lenKP2:
    #     number_keypoints = lenKP1
    # else:
    #     number_keypoints = lenKP2
    
    percentage = "{0:.2f}".format(lenMatches / lenKP2 * 100)


    index = 0
    mat = [1,2,3,4,5,6,7,8,9,10]

    # for m in matches[:10]:
    #     rounded = "{0:.2f}".format(m.distance)
    #     mat[index] = rounded
    #     index += 1
    s = 0
    # for m in matches[:20]:
    #     s += m.distance


    print("With:{0}, KP1:{1}, KP2:{2}, Match:{3}, %:{4}, {5}".format(i, lenKP1, lenKP2, lenMatches, percentage, s / 20))

    ###### dimensions of resizing (Width, height)
    # dim = (1450, 450)
    ###### resize image
    # matching_result = cv2.resize(matching_result, dim, interpolation = cv2.INTER_AREA)
    cv2.imshow("Matching result", cv2.resize(matching_result, None, fx=0.2, fy=0.2))
time2 = time.time()

print(time2 - time1)






######### dimensions of resizing (Width, height)
# dim = (650, 325)
# resize image
# imgRead = cv2.resize(imgRead, dim, interpolation = cv2.INTER_AREA)
# imgComp = cv2.resize(imgComp, dim, interpolation = cv2.INTER_AREA)

####### SIFT Example
# sift = cv2.xfeatures2d.SIFT_create()
# kp1, des1 = sift.detectAndCompute(imgRead, None)
# kp2, des2 = sift.detectAndCompute(imgComp, None)


# ##### Brute Force Matching
# bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
# matches = bf.match(des1, des2)
# matches = sorted(matches, key = lambda x:x.distance)
# matching_result = cv2.drawMatches(imgRead, kp1, imgComp, kp2, matches[:50], None, flags=2)

###### Print num of matches
# print(len(kp1))
# print(len(kp2))
# print(len(matches))
# print(len(matches) / len(kp1))

# for m in matches[:10]:
#     print(m.distance)


# dimensions of resizing (Width, height)
# dim = (1300, 650)
# resize image
# matching_result = cv2.resize(matching_result, dim, interpolation = cv2.INTER_AREA)

# cv2.imshow("imgRead", imgRead)
# cv2.imshow("ImgComp", imgComp)
# cv2.imshow("Matching result", matching_result)


##### Show Image like in editor
# plt.imshow(imgRead, cmap = 'gray', interpolation = 'bicubic')
# plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
# plt.show()

# plt.imshow(imgComp, cmap = 'gray', interpolation = 'bicubic')
# plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
# plt.show()


# wait for keystroke
k = cv2.waitKey(0)
cv2.destroyAllWindows()
