import cv2 as cv 
import numpy as np
from skimage.feature import hog

type='head'
img = cv.imread('images/thai_head/sukhothai_buddha_1300_copper_LACMA.jpeg')
print(img.shape)
#cropping about the centre
targetaspectratio = 3/3
height,width,color = img.shape
currentaspectratio = width/ height
if currentaspectratio > targetaspectratio:
    #we are cutting the width 
    centre = (int(width/2),int(height/2))
    right = centre[0] + int(height*targetaspectratio/2)
    left = centre[0] - int(height*targetaspectratio/2)
    img=img[:,left:right]
elif currentaspectratio < 280/650:
    #we are adding to the width
    newWidth = height*3/4
    img = cv.copyMakeBorder(img,top=0,bottom=0,left=round((newWidth-width)/2),right=round((newWidth-width)/2),borderType=cv.BORDER_CONSTANT,value=[0,0,0])
    #print(img.shape)
else:
    #we are cutting the height 
    centre = (int(width/2),int(height/2))
    top = centre[1] - int(width/targetaspectratio/2)
    bottom = centre[1] + int(width/targetaspectratio/2)
    img=img[top:bottom,:]
#print(img.shape)
#downscaling
width = 300
scale  = width/img.shape[1]
height = int(img.shape[0] * scale)
resized = cv.resize(img,(width,height),interpolation=cv.INTER_AREA)
print(resized.shape)
#Convert to greyscale
greyResized = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
cv.imshow('resizedimg',greyResized)
#adjust contrast with clahe
#--clip: The threshold for contrast limiting. You’ll typically want to leave this value in the range of 2-5. If you set the value too large, then effectively, what you’re doing is maximizing local contrast, which will, in turn, maximize noise (which is the opposite of what you want). Instead, try to keep this value as low as possible.
#--tile: The tile grid size for CLAHE. Conceptually, what we are doing here is dividing our input image into tile x tile cells and then applying histogram equalization to each cell (with the additional bells and whistles that CLAHE provides).
clahe = cv.createCLAHE(clipLimit=2,tileGridSize=(3,3))
greyResized = clahe.apply(greyResized)
print(greyResized.shape)
cv.imshow('clahe',greyResized)
if type == 'body':
#adjust sharpness
    kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    greyResizedSharp = cv.filter2D(src=greyResized,ddepth=-1,kernel=kernel)
elif type == 'head':
    #adjust edge blur
    img_blur = cv.blur(greyResized,(4,4))
    cv.imshow('gaussianblur',img_blur)
    #cv.waitKey(0)
    #Canny edge detection
    greyResizedCanny = cv.Canny(img_blur,threshold1=255/3,threshold2=255)
    greyResizedCanny = cv.bilateralFilter(greyResizedCanny,4,100,100)
    #calculate sift keypoints  
    sift = cv.SIFT_create()
    kp = sift.detect(greyResizedCanny,None)
    # print(kp[0])
    # print(len(kp))
    des = sift.compute(greyResizedCanny,kp)
    #print(des[0][0])
    #print(len(des[1]))
    #create hog 
    

#cv.imwrite('test_input/output.jpg',img_blur)
cv.imshow('TEST',greyResizedCanny)
img_1 = cv.drawKeypoints(greyResizedCanny,kp,greyResizedCanny)
cv.imshow('keypoints',img_1)
cv.waitKey(0)
