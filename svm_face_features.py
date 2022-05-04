#%%
import cv2 as cv 
import pandas as pd
import numpy as np
from sklearn.svm import SVC

# GROUP 0 sukhothaiImgLs= ['sukhothai_buddha_1300_copper_LACMA','sukhothai_buddha_1400_bronze_met','sukhothai_buddha_1300_copper_art_gallery_NSW','sukhothai_buddha_1400_bronze_the_walters_art','sukhothai_buddha_1300_copper_LACMA','suk_ayutthaya_buddha_1500_copper_LACMA_cropped']
# GROUP 1 ayutthayaImgLs=['ayutthaya_1300_sandstone_LACMA','ayutthaya_buddha_1347_bronze_NGAustralia','ayutthaya_buddha_1400_1500_bronze_met','ayutthaya_1700_bronze_freer_art']
# GROUP 2 testImg = ['test_1400_1500_bronze_central_thai_LACMA','test_1400_1500_bronze_LACMA']
#hogDf = pd.read_pickle('hogCombinedDf.pkl')
siftDf = pd.read_pickle('siftCombinedDf.pkl')
print(siftDf.shape)
print()

def siftKeypointCentroids(siftDf,mask,bin=200):
    # by combining the sift keypoints descriptors into kmeans clusters 
    # each cluster has a centroid 
    # Each mask has its own centroid set for matching with test keypoint descriptor
    # returns the centroids for group 0 and group 1 in dictionary
    # group 0 sukhothai group 1 ayutthaya
    print(mask)
    #prepare training data into the sukhothai GROUP 0 and ayutthaya GROUP 1
    print('sukhothai')
    t = siftDf.loc[siftDf['mask']==mask]
    trainingDfSukhothai = t.loc[t['group'] == 0]
    t = (trainingDfSukhothai['descriptor_matrix'])
    trainingSetSukothai=[]
    for i in t:
        #print(len(i))
        trainingSetSukothai.append(list(i))
    trainingSetSukothai = np.array(trainingSetSukothai)
    print(trainingSetSukothai.shape)
    print('')

    print('ayutthaya')
    trainingDfAyutthaya = siftDf.loc[siftDf['group'] == 1]
    t = trainingDfAyutthaya['descriptor_matrix']
    trainingSetAyutthaya = []
    for i in t:
        #print(len(i))
        trainingSetAyutthaya.append(list(i))
    trainingSetAyutthaya = np.array(trainingSetAyutthaya)
    print(trainingSetAyutthaya.shape)
    print('')
    
    from sklearn.cluster import KMeans
    #break up the selected descriptors into 'bins' 
    if trainingSetAyutthaya.shape[0] < bin:
        clusters=trainingSetAyutthaya.shape[0]
        print(mask +' ayutthaya has less than '+str(bin) + ' only '+str(clusters)+' used')
    else:
        clusters = bin 
    sukhothaiCentriods = KMeans(clusters).fit(trainingSetAyutthaya).cluster_centers_
    if trainingSetSukothai.shape[0] < bin:
        clusters=trainingSetSukothai.shape[0]
        print(mask +' sukhothai has less than '+str(bin) + ' only '+str(clusters)+' used')
    else:
        clusters = bin 
    ayutthayaCentroids = KMeans(clusters).fit(trainingSetSukothai).cluster_centers_
    print(sukhothaiCentriods.shape)
    print(ayutthayaCentroids.shape)
    print('kmeans for training set done\n')

    return {'sukhothai':sukhothaiCentriods,'ayutthaya':ayutthayaCentroids}

def siftDistHist(siftDf,mask,centroidsDict,bins):
    #preparing the all keypoints values to be compared to the centroids 
    print('preparing all values for'+mask)
    #select mask sift keypoints
    testDf=siftDf.loc[siftDf['mask']==mask]
    #print(testDf)
    imgNames = list(set(list(testDf['image_name'])))
    #print(imgNames)
    imgKeypoints={}
    for img in imgNames:
        t = testDf.loc[testDf['image_name']==img]
        t = t['descriptor_matrix']
        print(img)
        print(t.shape)
        testSetX = []
        for i in t:
            #print(len(i))
            testSetX.append(list(i))
        testSetX = np.array(testSetX)
        print(testSetX.shape)
        imgKeypoints[img] = testSetX
    #building comparison histogram for one mask 
    #There are 200 centroid rows for each mask, we are using euclidean distances to find the centroid row with 
    #minimum distance to each of the test keypoints
    #dictinary contains the values for the 200 by 1 histogram for each of the test images 
    # this histogram is used in the svc
    # the histograms are returned in a dataframe
    # number of histograms is number of images * number of masks
    print('forming comparison histogram for Sukhothai centroids')
    sukhothaiCentriods = centroidsDict['sukhothai']
    sukhothaiMatchHistDf = pd.DataFrame(columns=['index','image_id','mask','group','histogram'])
    counter =0 
    for img in imgNames:
        print(img)
        #get the keypoint array
        testSet = imgKeypoints[img]
        print(testSet.shape)
        from sklearn.metrics import pairwise_distances_argmin_min
        centroidRowMatches,distances = pairwise_distances_argmin_min(testSet,sukhothaiCentriods,metric='hamming')
        histogram= np.zeros(bins)
        for i in centroidRowMatches:
            histogram[i] += 1
        print('histogram '+str(histogram.shape))
        #print(histogram[100])
        if np.sum(histogram) == testSet.shape[0]:
            print('the histogram values matches the number of test image keypoints')
        else:
            print('oi mate the histogram values sum do not match number of test image keypoints')
        #populate dataframe
        sukhothaiMatchHistDf.loc[counter,'index'] = img+'_'+mask
        sukhothaiMatchHistDf.loc[counter,'image_id'] = img
        sukhothaiMatchHistDf.loc[counter,'mask'] = mask 
        sukhothaiMatchHistDf.loc[counter,'histogram'] = histogram
        if img[0] == 't':
            sukhothaiMatchHistDf.loc[counter,'group'] = 2
        elif img[0] == 's':
            sukhothaiMatchHistDf.loc[counter,'group']=0
        else:
            sukhothaiMatchHistDf.loc[counter,'group']=1
        counter+=1
    print(sukhothaiMatchHistDf.shape)
    print()
    print('forming comparison histogram for ayutthaya centroids')
    ayutthayaCentriods = centroidsDict['ayutthaya']
    ayutthayaMatchHistDf = pd.DataFrame(columns=['index','image_id','mask','group','histogram'])
    counter =0 
    for img in imgNames:
        print(img)
        #get the keypoint array
        testSet = imgKeypoints[img]
        print(testSet.shape)
        from sklearn.metrics import pairwise_distances_argmin_min
        centroidRowMatches,distances = pairwise_distances_argmin_min(testSet,ayutthayaCentriods)
        histogram= np.zeros(bins)
        for i in centroidRowMatches:
            histogram[i] += 1
        print('histogram '+str(histogram.shape))
        #print(histogram[100])
        if np.sum(histogram) == testSet.shape[0]:
            print('the histogram values matches the number of test image keypoints')
        else:
            print('oi mate the histogram values sum do not match number of test image keypoints')
        #populate dataframe
        ayutthayaMatchHistDf.loc[counter,'index'] = img+'_'+mask
        ayutthayaMatchHistDf.loc[counter,'image_id'] = img
        ayutthayaMatchHistDf.loc[counter,'mask'] = mask 
        ayutthayaMatchHistDf.loc[counter,'histogram'] = histogram
        if img[0] == 't':
            ayutthayaMatchHistDf.loc[counter,'group'] = 2
        elif img[0] == 's':
            ayutthayaMatchHistDf.loc[counter,'group']=0
        else:
            ayutthayaMatchHistDf.loc[counter,'group']=1
        counter+=1
    print(ayutthayaMatchHistDf.shape)
    print()
    return sukhothaiMatchHistDf,ayutthayaMatchHistDf

def inverseDocFrequency(sukhothaiHistDf,ayutthayaHistDf):
    # using the sklearn tfidftransformer function to created a vector transformer
    # the vector transformer is weighted vector that for each of the 200 long sift histograms
    # the term frequency and inverse document frequency concept 
    # instead of only using term frequency which is the siftDistHist 
    # we reduce the weightage of histograms values, centroids that are common across the test images 
    # so we are using the centriods that are unique to each of the test images 
    # we are combining both ayutthaya and sukhothai histograms. 
    # each histogram now has twice the number of keywords. 
    # combined the histograms for both categories to form 400 long histograms
    mergedHistDf = sukhothaiHistDf.merge(ayutthayaHistDf,on='index')
    print('mergedDf')
    print(mergedHistDf.shape)
    print(list(mergedHistDf.columns))
    mergedHistDf.to_csv('combinedHistDump.csv')
    finalHistDf = pd.DataFrame(columns=['index','image_id','mask','group','histogram'])
    for i in range(mergedHistDf.shape[0]):
        sukhothaiHist = mergedHistDf.loc[i,'histogram_x']
        ayutthayaHist = mergedHistDf.loc[i,'histogram_y']
        combinedHist = np.concatenate((sukhothaiHist,ayutthayaHist),axis=None)
        #print(combinedHist.shape)
        finalHistDf.loc[i,'index'] = mergedHistDf.loc[i,'index']
        finalHistDf.loc[i,'image_id'] = mergedHistDf.loc[i,'image_id_x']
        finalHistDf.loc[i,'mask'] = mergedHistDf.loc[i,'mask_x']
        finalHistDf.loc[i,'group'] = mergedHistDf.loc[i,'group_x']
        finalHistDf.loc[i,'histogram'] = combinedHist
    print()
    print('final cleaned df')
    print(finalHistDf.shape) 
    print('new hist shape')
    print(finalHistDf.loc[0,'histogram'].shape)
    print()
    # conduct tfidf to get weights in a 400 long vector
    # retrieve histograms in a 400 by number of imaged masks value as numpy array 
    temp = []
    histogramSeries = finalHistDf['histogram']
    for histogram in histogramSeries:
        histogram = np.array(histogram)
        #print(histogram.shape)
        temp.append(histogram) 
    temp = np.array(temp)
    print(np.max(temp),np.mean(temp),np.median(temp))
    #standardise
    temp = (temp - np.mean(temp))/np.std(temp)
    print(np.max(temp),np.mean(temp),np.median(temp))
    temp = np.cbrt(temp)
    print(np.max(temp),np.mean(temp),np.median(temp))
    print(temp.shape)
    print()
    # apply tfidfTransformer 
    from sklearn.feature_extraction.text import TfidfTransformer
    weightedVector = TfidfTransformer().fit(temp).idf_
    print(weightedVector.shape)

    # multiply the weights to the histograms
    for i in range(finalHistDf.shape[0]):
        finalHistDf.loc[i,'histogram'] = finalHistDf.loc[i,'histogram']*weightedVector
    #print(temp.shape)
    # return the final Hist with the longer weighted histograms
    return finalHistDf

def siftSvcimput(siftDf,maskLs,bins):
    # return a combined dataframe of all the histograms
    # each image * each mask = number of histograms
    sukhothaiCombinedHistLs =[]
    ayutthayaCombinedHistLs=[]
    for mask in maskLs:
        centroidsDict = siftKeypointCentroids(siftDf,mask,bins)
        sukhothaiHistDf,ayutthayaHistDf = siftDistHist(siftDf,mask,centroidsDict,bins)
        sukhothaiCombinedHistLs.append(sukhothaiHistDf)
        ayutthayaCombinedHistLs.append(ayutthayaHistDf)
    #concat them dataframes
    print('concat sift histograms')
    sukhothaiHistDf = pd.concat(sukhothaiCombinedHistLs)
    ayutthayaHistDf = pd.concat(ayutthayaCombinedHistLs)
    #reset index so when using loc or iloc, we can correctly get the correctly indexed row
    sukhothaiHistDf.reset_index(inplace=True)
    ayutthayaHistDf.reset_index(inplace=True)
    print('sukhothai '+str(sukhothaiHistDf.shape))
    print('ayutthaya '+str(ayutthayaHistDf.shape))
    return sukhothaiHistDf, ayutthayaHistDf

def siftSvc(histDf):
    # appending all the masks together
    # one image has now bins*masks array 
    masksCombined = pd.DataFrame(columns=['image_id','group','histogram','mask_order'])
    imageNameLs = list(set(list(histDf['image_id'])))
    indexOg =0 
    for imageName in imageNameLs:
        print(imageName)
        currentImgDf = histDf.loc[histDf['image_id'] == imageName]
        t = currentImgDf['histogram']
        #print(t.iloc[0].shape[0])
        nBins=t.iloc[0].shape[0]
        maskorder = list(currentImgDf['mask'])
        print(maskorder)
        tLs = np.zeros(nBins*len(maskorder),dtype='float')
        #print(tLs.shape)
        counter = 0
        for i in t:
            for j in i:
                tLs[counter] = j
                counter +=1
        print(tLs.shape)
        masksCombined.loc[indexOg,'image_id'] = imageName
        masksCombined.loc[indexOg,'histogram'] = tLs
        masksCombined.loc[indexOg,'mask_order'] = maskorder
        if imageName[0] == 's':
            masksCombined.loc[indexOg,'group'] = 0
        elif imageName[0] == 't':
            masksCombined.loc[indexOg,'group'] = 2
        else:
            masksCombined.loc[indexOg,'group'] = 1
        indexOg += 1 
    print('training set df shape')
    print(masksCombined.shape)
    masksCombined.to_csv('trainingX.csv')
    # training set 
    trainingDf = masksCombined.loc[masksCombined['group']!=2]
    trainingDfX = list(trainingDf['histogram'])
    trainingX = []
    for i in trainingDfX:
        trainingX.append(i)
    trainingX = np.array(trainingX)
    np.save('trainingX',trainingX)
    trainingY = np.array(trainingDf['group'])
    trainingY = trainingY.astype('int')
    print('training set')
    print(trainingX.shape)
    #print(type(trainingX[0]))
    print(trainingY)
    
    testDf = masksCombined.loc[masksCombined['group'] == 2].reset_index()
    testDfX = list(testDf['histogram'])
    testX = []
    for i in testDfX:
        testX.append(i)
    testX = np.array(testX)
    #svc model
    svc = SVC()
    model = svc.fit(trainingX,trainingY)
    prediction = model.predict(testX)
    testDf['prediction'] = prediction
    #print(prediction)
    return testDf

def writingTestImgClass(outputPath,inputPath,predictionDf):
    import os
    # delete the results of the last run. 
    filelist = [ f for f in os.listdir(outputPath) if f.endswith(".jpeg") ]
    for f in filelist:
        os.remove(os.path.join(outputPath, f))
    for row in np.arange(0,predictionDf.shape[0]):
        #print(row)
        imageFileName = predictionDf.loc[row,'image_id']
        #print(imageFileName)
        imgFile=cv.imread(inputPath+imageFileName+'.jpeg')
        classifier = predictionDf.loc[row,'prediction']
        print(inputPath+imageFileName+'.jpeg'+'  '+str(classifier))
        cv.imwrite(outputPath+imageFileName+"_group_"+str(classifier)+'.jpeg',imgFile)
    return 


# maskLs=['eyebrow_nose_mask','mouth_mask']
# compare2sukhothai, compare2ayutthaya =  siftSvcimput(siftDf,maskLs,200)
# compare2sukhothai.to_csv('siftHistCompare2Sukhothai.csv')
# compare2ayutthaya.to_csv('siftHistCompare2Ayutthaya.csv')
# compare2sukhothai.to_pickle('siftHistCompare2Sukhothai.pkl')
# compare2ayutthaya.to_pickle('siftHistCompare2Ayutthaya.pkl')

compare2sukhothai = pd.read_pickle('siftHistCompare2Sukhothai.pkl')
compare2ayutthaya = pd.read_pickle('siftHistCompare2Ayutthaya.pkl')


combinedWieghtedHist = inverseDocFrequency(compare2sukhothai,compare2ayutthaya)
predictionDf = siftSvc(combinedWieghtedHist)
predictionDf.to_csv('siftPrediction.csv')

writingTestImgClass('output_images/test_classified/','images/thai_head/',predictionDf)


# %%
