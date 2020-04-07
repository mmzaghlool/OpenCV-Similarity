import cv2
import numpy as np
from matplotlib import pyplot as plt

######## Read images
img1 = cv2.imread('100_2.jpg', 0)
img2 = cv2.imread('100_2.jpg', 0)



# dimensions of resizing (Width, height)
dim = (650, 325)
# resize image
img1 = cv2.resize(img1, dim, interpolation = cv2.INTER_AREA)
img2 = cv2.resize(img2, dim, interpolation = cv2.INTER_AREA)

####### SURF Example
surf = cv2.xfeatures2d.SURF_create()
kp1, des1 = surf.detectAndCompute(img1, None)
kp2, des2 = surf.detectAndCompute(img2, None)


##### Brute Force Matching
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key = lambda x:x.distance)
matching_result = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], None, flags=2)


###### Print num of matches
print(len(matches))

for m in matches[:10]:
    print(m.distance)


# dimensions of resizing (Width, height)
dim = (1300, 650)
# resize image
matching_result = cv2.resize(matching_result, dim, interpolation = cv2.INTER_AREA)

# cv2.imshow("Img1", img1)
# cv2.imshow("Img2", img2)
cv2.imshow("Matching result", matching_result)


##### Show Image like in editor
# plt.imshow(img1, cmap = 'gray', interpolation = 'bicubic')
# plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
# plt.show()

# plt.imshow(img2, cmap = 'gray', interpolation = 'bicubic')
# plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
# plt.show()


# wait for keystroke
k = cv2.waitKey(0)
cv2.destroyAllWindows()
