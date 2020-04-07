import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
import glob

####### get time at start of the execution
time1 = time.time()

####### list of all resources
imgsComp = [
    # {"path": '1LE.jpg', "val": 1},
    {"path": '5.jpeg', "val": 5},
    {"path": '5_2.jpg', "val": 5},
    {"path": '20.jpg', "val": 20},
    {"path": '20_2.jpg', "val": 20},
    {"path": '50.jpeg', "val": 50},
    {"path": '50_2.jpg', "val": 50},
    {"path": '100_2.jpg', "val": 100},
    {"path": '100_3.jpg', "val": 100},
    {"path": '100.jpg', "val": 100},
]

####### list of all reading list
imgsRead = [
    '5_2.jpg',
    '20.jpg',
    '20_2.jpg',
    '50_1.jpg',
    '50_2.jpg',
    '50_3.jpg',
    '100_4.jpg',
    '100_5.jpeg',
    '100_6.jpg',
    '100_cap1.jpg',
    '100_cap2.jpg',
    '100_cap3.jpg',
    '100_old.jpeg',
]

class Final():
    result = []
    s = 0
    
    def compute(percent, distanceAverage, val, path):

        if distanceAverage < 0.075:        
            Final.result.append({"percent": percent, "distanceAverage": distanceAverage, "val": val, "path": path})

 
    def crop(imagePath):
        # Reading image 
        processImg2 = cv2.imread(imagePath) 
        
        # Reading same image in another variable and  
        # converting to gray scale. 
        processImg = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE) 

        # Converting image to a binary image  
        # (black and white only image). 
        _,threshold = cv2.threshold(processImg, 110, 255, cv2.THRESH_BINARY) 
       
        # Detecting shapes in image by selecting region  
        # with same colors or intensity. 
        image, contours, hierarchy=cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)                 
        
        # Find object with the biggest bounding box
        mx = (0,0,0,0)      # biggest bounding box so far
        mx_area = 0
        arr = []
        for cont in contours:
            x,y,w,h = cv2.boundingRect(cont)
            area = cv2.contourArea(cont) 
            arr.append({'x': x, 'y': y, 'w': w, 'h': h, 'area': area})

        arr = sorted(arr, key = lambda x: -x['area'])
        # heightRead, widthRead, channelsx = processImg2.shape
        
        x = arr[0]['x']
        y = arr[0]['y']
        w = arr[0]['w']
        h = arr[0]['h']
        ###### Output to files
        ###### crop image
        roi=processImg2[y:y+h,x:x+w]
        ###### invert colors from BRG to RGB ... matplot uses RGB ... cv2 uses BRG
        # roi = roi[:,:,::-1]
        ##### show image using matplot 
        # plt.imshow(roi)
        # plt.show()
        return roi
            

    def removeResult(path):
        if len(Final.result) > 0:
            newResult = sorted(Final.result, key = lambda x: -x["percent"])
            # print(newResult)
            res = newResult[0]["val"]
            print("{0}: {1}".format(path, res))
            Final.s += res
            print("Total: {0}".format(Final.s))
            
        else:
            print('no val added')

        Final.result=[]

    
    def compare(imgRead, kpRead, desRead, path):

        for i in imgsComp:
            ######## Read image
            comp = cv2.imread("images/original/" + i["path"])

            ##### detect all keypoints and descriptors 
            kpComp, desComp = sift.detectAndCompute(comp, None)

            ##### Brute Force Matching
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            matches = bf.match(desComp, desRead)
            matches = sorted(matches, key = lambda x: x.distance)
            
            ##### comp result
            lenKPComp = len(kpComp)
            lenKPRead = len(kpRead)
            lenMatches = len(matches)


            percent = lenMatches / lenKPRead * 100

            x = 0
            for m in matches[:30]:
                x += m.distance

            distanceAverage = x / 30

            Final.compute(percent, distanceAverage, i["val"], i["path"]);
        Final.removeResult(path)


        
    def start():

        for image in imgsRead:
            ######## Read image images/comp/5_2.JPG
            path = "images/comp/" + image
            imgRead = Final.crop(path)

            ##### detect all keypoints and descriptors 
            kpRead, desRead = sift.detectAndCompute(imgRead, None)

            ###### compare this image to determine the value
            Final.compare(imgRead, kpRead, desRead, path)

####### SIFT
sift = cv2.xfeatures2d.SURF_create()
# sift = cv2.ORB_create(nfeatures=50000)

Final.start()


time2 = time.time()

print("time: {0}".format(time2 - time1))

# wait for keystroke
k = cv2.waitKey(0)
cv2.destroyAllWindows()
