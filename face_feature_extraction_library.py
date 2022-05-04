import cv2 as cv 
import pandas as pd
import numpy as np
import pickle

def resizeImage(path,widthHeight=3/4,targetHeight=300):
    # this returns a smaller image in greyscale in 4:3 ratio
    img = cv.imread(path)
    targetaspectratio = widthHeight
    height,width,color = img.shape
    #greyscale
    imgGrey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #adjust contrast with clahe if the image is super low res     
    if width < targetHeight:
        #--clip: The threshold for contrast limiting. You’ll typically want to leave this value in the range of 2-5. If you set the value too large, then effectively, what you’re doing is maximizing local contrast, which will, in turn, maximize noise (which is the opposite of what you want). Instead, try to keep this value as low as possible.
        #--tile: The tile grid size for CLAHE. Conceptually, what we are doing here is dividing our input image into tile x tile cells and then applying histogram equalization to each cell (with the additional bells and whistles that CLAHE provides).
        clahe = cv.createCLAHE(clipLimit=2,tileGridSize=(5,5))
        imgContrast= clahe.apply(imgGrey)    
    else:
        imgContrast = imgGrey
    print('initial shape '+str(imgContrast.shape))

    # correct the aspect ratio but works when current image is bigger than the target size
    height,width =imgContrast.shape
    currentaspectratio = width/height
    #cropping about the centre
    if currentaspectratio > targetaspectratio:
        #we are cutting the width 
        centre = (int(width/2),int(height/2))
        right = centre[0] + int(height*targetaspectratio/2)
        left = centre[0] - int(height*targetaspectratio/2)
        imgContrast=imgContrast[:,left:right]
    elif currentaspectratio < targetaspectratio:
        #we are adding to the width
        newWidth = height*1/widthHeight
        imgContrast = cv.copyMakeBorder(imgContrast,top=0,bottom=0,left=round((newWidth-width)/2),right=round((newWidth-width)/2),borderType=cv.BORDER_CONSTANT,value=[0,0,0])
        #print(img.shape)
    else:
        #we are cutting the height 
        centre = (int(width/2),int(height/2))
        top = centre[1] - int(width/targetaspectratio/2)
        bottom = centre[1] + int(width/targetaspectratio/2)
        imgContrast=imgContrast[top:bottom,:]
    #print(img.shape)
    #downscaling
    height = targetHeight
    scale  = height/imgContrast.shape[0]
    width = int(imgContrast.shape[1] * scale)
    resized = cv.resize(imgContrast,(width,height),interpolation=cv.INTER_AREA)
    print('final shape '+str(resized.shape)+'\n')
    return resized

def imageDescriptorSift(image,image_id,mask,kpimgpath,format):
    # return kp and descriptors in dataframe
    # kp object has several attributes, they are self explanotry, 
    # response is the magnitude of the quality of the kp detectAndCompute removes the bottom 3%
    # the descriptor is a matrix that contains the info of size, angle of kp, creating a fingerprint
    #calculate sift keypoints  and  descriptors 
    print(image_id,mask)
    sift = cv.SIFT_create()
    kp,des = sift.detectAndCompute(image,None)
    df = pd.DataFrame(columns=['id','image_name','mask','group','coord_x','coord_y','kp_response','angle','size','descriptor_matrix'])
    # draw the keypoints image
    tempKeypointImage = cv.drawKeypoints(image,kp,image)
    cv.imwrite(kpimgpath+'sift_'+image_id+'_'+mask+format,tempKeypointImage)
    #print(kp[0].response)
    #print(type(kp))
    #print(des[0])
    #print(len(des))
    #print(len(kp))
    counter=0
    for keypoint in kp:
        if image_id[0] == 's':
            df.loc[counter,'group'] = 0
        elif image_id[0] == 't':
            df.loc[counter,'group'] = 2
        else:
            df.loc[counter,'group'] = 1
        df.loc[counter,'id'] = image_id+'_'+str(counter)
        df.loc[counter,'image_name'] = image_id
        df.loc[counter,'mask'] = mask
        df.loc[counter,'coord_x'] = keypoint.pt[0]
        df.loc[counter,'coord_y'] = keypoint.pt[1]
        df.loc[counter,'kp_response'] = keypoint.response
        df.loc[counter,'size'] = keypoint.size
        df.loc[counter,'angle'] = keypoint.angle
        df.loc[counter,'descriptor_matrix'] = des[counter]
        #print(len(des[counter]))
        counter +=1
    print(str(df.shape)+'\n')
    df.set_index('id')
    return df
   
def imgMasks(imgpath,maskpath):
    #retrieve the masked part of the image
    mask=cv.imread(maskpath)
    mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
    img=cv.imread(imgpath)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    targetSize = img.shape
    initialSize = mask.shape
    print(mask.shape)
    print(img.shape)
    #check height
    if targetSize != initialSize:
        mask = cv.resize(mask,(targetSize[1],targetSize[0]))
        output = cv.bitwise_and(img,mask)
        print(mask.shape)
    else:
        output = cv.bitwise_and(img,mask)
        print(mask.shape)
    print(img.shape)
    print('\n')
    return output

def adjustBrightness(img,mean=255/2):
    # increase and decrease pixel intensity
    # we return a tuple one brighter the other darker 
    initialMean = np.mean(img)
    ratio = mean/initialMean
    newImg = img*ratio
    return newImg

def sharpenBlur(img,kernelGaussian=(3,3),kernelSharpFactor = 5):
    imgBlur = cv.GaussianBlur(img,kernelGaussian,0)
    kernel = np.array([[0, -1, 0],[-1, int(kernelSharpFactor),-1],[0, -1, 0]])
    imgSharp = cv.filter2D(img,ddepth=-1,kernel=kernel)
    
    return imgBlur , imgSharp

path = 'images/thai_head/'
format='.jpeg'
outputPath = 'cropped_images/thai_head/'
# GROUP 0 sukhothaiImgLs= ['sukhothai_buddha_1300_copper_LACMA','sukhothai_buddha_1400_bronze_met','sukhothai_buddha_1300_copper_art_gallery_NSW','sukhothai_buddha_1400_bronze_the_walters_art','sukhothai_buddha_1300_copper_LACMA','suk_ayutthaya_buddha_1500_copper_LACMA_cropped']
# GROUP 1 ayutthayaImgLs=['ayutthaya_1300_sandstone_LACMA','ayutthaya_buddha_1347_bronze_NGAustralia','ayutthaya_buddha_1400_1500_bronze_met','ayutthaya_1700_bronze_freer_art']
# GROUP 2 TEST = ['test_1400_1500_bronze_central_thai_LACMA']
# read from pickle file for name list
with open('imageNameLsShort.pkl', 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    imgNameLsShort = pickle.load(f)
print(len(imgNameLsShort))
#img = editImage(path+'ayutthaya_buddha_1347_bronze_NGAustralia'+format)
#cv.imwrite('cropped_images/thai_head/ayutthaya_buddha_1347_bronze_NGAustralia.jpg',img)
#here we resize the images, and apply the 
import os
# delete previous set training and test of the last run. 
filelist = [ f for f in os.listdir(outputPath) if f.endswith(".jpeg") ]
for f in filelist:
    os.remove(os.path.join(outputPath, f))

print('resize images')
imgDfLs = []
imgLs=[]
for i in imgNameLsShort:
    fileName = path+i+format
    print(fileName)
    img = resizeImage(fileName)
    imgLs.append(img)
    if i[0] != 't':
        #brighterImg = adjustBrightness(img,mean=255/2)
        #brightImgName = i + '_standard'
        #cv.imwrite(outputPath+brightImgName+format,brighterImg)
        cv.imwrite(outputPath+i+format,img)
        # imgBlur, imgSharp = sharpenBlur(img,(7,7),5)
        # imgBlurName = i + '_blur'
        # imgSharpName= i + '_sharp'
        # cv.imwrite(outputPath+imgBlurName+format,imgBlur)
        # cv.imwrite(outputPath+imgSharpName+format,imgSharp)
    else:
        #brighterImg = adjustBrightness(img,mean=255/2)
        #brightImgName = i
        #cv.imwrite(outputPath+brightImgName+format,brighterImg)
        cv.imwrite(outputPath+i+format,img)
    #cv.imshow('test',editImage(fileName))
    #cv.waitKey(0)

#get list of all images for feature extraction incuding our darker and brighter images. 
path='cropped_images/thai_head/'
import os
imageLs = os.listdir(path)
imageLsOutput = []
for imgStr in imageLs:
    imgLs = list(imgStr)
    #print(imgLs)
    index = imgLs.index('.')
    #print(index)
    name =imgLs[0:index]
    nameStr = ''.join(name)
    print(nameStr)
    imageLsOutput.append(nameStr)
import pickle
with open('imageNameLs.pkl', 'wb') as f:
    pickle.dump(imageLsOutput,f)

# read from pickle file for name list
with open('imageNameLs.pkl', 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    imgNameLs = pickle.load(f)
print(len(imgNameLs))

maskpath='images/thai_head_mask/'
maskLs=['ears_mask','hair_mask','eyebrow_nose_mask','mouth_mask']
maskedImgDict = {}
print('mask images')
for mask in maskLs:
    dict ={}
    for img in imgNameLs:
        print(mask,img)
        maskedimg = imgMasks(outputPath+img+format,maskpath+mask+format)
        dict[img] = maskedimg
        #cv.imshow(img,maskedimg)
        #cv.imwrite(maskedimgPath+mask+img+format,maskedimg)
    maskedImgDict[mask] = dict
#cv.waitKey(0)

print('calculate sift')
# #computing the sift descriptors and the assigning keypoint id and mask
kpimagepath='output_images/thai_head/'
siftdfLs=[]
for mask in maskLs:
    tempMask = maskedImgDict[mask]
    for img in imgNameLs:
        tempImg = tempMask[img]
        print(tempImg.shape)        
        tempDf = imageDescriptorSift(tempImg,img,mask,kpimagepath,format)
        siftdfLs.append(tempDf)

#concat the dfLs
combinedSiftDf = pd.concat(siftdfLs)
print()
print('combined sift key points')
print(combinedSiftDf.shape)
combinedSiftDf.reset_index(inplace=True)
combinedSiftDf.to_csv('all_sift_descr.csv')
combinedSiftDf.to_pickle('siftCombinedDf.pkl')