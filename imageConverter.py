import os 
import cv2 as cv
import numpy as np
# convert raw images 
path='images/raw_thai_head/'
imageLs = os.listdir(path)
#print(imageLs)
for imgStr in imageLs:
    imgLs = list(imgStr)
    #print(imgLs)
    index = imgLs.index('.')
    #print(index)
    formatLs =imgLs[index+1:]
    format = ''.join(formatLs)
    print(format)
    if format != 'jpeg':
        readimg = np.array(cv.imread(path+imgStr))
        outputFileName= ''.join(imgLs[:index])+'.jpeg'
        imgOutput = cv.imwrite(path+outputFileName,readimg)

#get list of all images for feature extraction
path='images/thai_head/'
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
with open('imageNameLsShort.pkl', 'wb') as f:
    pickle.dump(imageLsOutput,f)

with open('imageNameLsShort.pkl', 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    imgNameLs = pickle.load(f)
print(len(imgNameLs))

test ={





  'ayutthaya_1300_u_thong_bronze_NSW':'https://www.artgallery.nsw.gov.au/collection/works/341.1985/',
  'ayuthuya_early_1400_bronze_walters':'https://art.thewalters.org/detail/37922/seated-buddha-inmaravijaya/', 
  'ayutthaya_1200_sandstone_victoria_albert':'https://collections.vam.ac.uk/item/O82514/head-of-buddha-sculpture-unknown/', 
  'ayutthaya_1300_1400_bronze_walters':'https://art.thewalters.org/detail/1403/buddha-in-royal-attire/', 
  'ayutthaya_1300_1500_bronze_victoria_albert':'https://collections.vam.ac.uk/item/O145297/sculpture-unknown/',        
  'ayutthaya_1300_sandstone_LACMA':'https://collections.lacma.org/node/246637',
  'ayutthaya_1300_1400_sandstone':'https://art.thewalters.org/detail/1453/head-of-the-buddha/', 
  'ayutthaya_1600_1650_bronze_walters':'https://art.thewalters.org/detail/37300/', 
  'ayutthaya_1600_crowned_bronze_victoria_albert':'https://collections.vam.ac.uk/item/O82511/sculpture-head-of-the-unknown/', 
  'ayutthaya_1700_tin_brass_walters':'https://art.thewalters.org/detail/1403/buddha-in-royal-attire/', 
  'ayutthaya_1700_hollow_cast_bronze_crowned_walters':'https://art.thewalters.org/detail/12624/',
  'ayutthaya_buddha_1347_bronze_NGAustralia':'https://searchthecollection.nga.gov.au/object?keyword=ayutthaya&searchIn=artistOrCulture&searchIn=title&searchIn=medium&uniqueId=103747', 
  'ayutthaya_1600_1625_bronze_walters':'https://art.thewalters.org/detail/23222/buddha-at-the-moment-of-victory-2/', 
  'ayutthaya_u_thong_1300_1400_bronze_walters':'https://art.thewalters.org/detail/29705/seated-buddha-in-maravijaya-13/', 
  'ayutthaya_1500_copper_LACMA_cropped':'https://collections.lacma.org/node/178446', 
  'ayutthaya_1700_bronze_smithsonian':'https://asia.si.edu/object/F1909.51/',
  'ayutthaya_1700_bronze_smithsonian_2':'https://asia.si.edu/object/F1909.49/',



  'sukhothai_1300_1400_bronze_british_museum':'https://www.britishmuseum.org/collection/object/A_1880-1002',
  'sukhothai_1550_1600_bronze_walters':'https://art.thewalters.org/detail/15634/seated-buddha-in-maravijaya-6/',
  'sukhothai_1550_bronze_walters':'https://art.thewalters.org/detail/39948/head-of-the-buddha-17/', 
  'sukhothai_1584_metal_ACM':'https://www.roots.gov.sg/Collection-Landing/listing/1245570', 
  'sukhothai_1584_metal_ACM_personal':'taken on ACM trip', 
  'sukhothai_1600_1700_bronze_ACM':'https://www.roots.gov.sg/Collection-Landing/listing/1120177', 
  'sukhothai_1600_1700_bronze_ACM_personal':'taken on ACM trip', 
  'sukhothai_1650_1700_bronze_walters':'https://art.thewalters.org/detail/16172/', 
  'sukhothai_1600_bronze_walters_2':'https://art.thewalters.org/detail/36196/', 
  'sukhothai_1650_bronze_phitsanulok_walters':'https://art.thewalters.org/detail/37881/buddha-at-the-moment-of-victory-4/', 
  'sukhothai_ayutthaya_1400_1500_victoria_albert':'https://collections.vam.ac.uk/item/O81140/figure-unknown/', 
  'sukhothai_ayutthaya_1400_1500_walters':'https://art.thewalters.org/detail/8655/seated-buddha-in-maravijaya-2/',
  'sukhothai_ayutthaya_1500_1600_copper_standing_ACM':'https://www.roots.gov.sg/Collection-Landing/listing/1251569', 
  'sukhothai_buddha_1300_copper_art_gallery_NSW':'https://www.artgallery.nsw.gov.au/collection/works/EV1.1963/', 
  'sukhothai_buddha_1300_copper_LACMA':'https://collections.lacma.org/node/209129', 
  'sukhothai_walking_1400_bronze_walters':'https://art.thewalters.org/detail/36064/walking-buddha/', 

  'unknown_1400_1500_bronze_central_thai_LACMA':'https://collections.lacma.org/node/175997',
  'unknown_1400_1500_bronze_LACMA':'https://collections.lacma.org/node/183248', 
  'unknown_1300_bronze_cantor':'https://cantorcollections.stanford.edu/objects-1/thumbnails?records=80&query=mfs%20all%20%22thailand%20buddha%22&sort=0', 
  'unknown_1500_1600_stone_met':'https://www.metmuseum.org/art/collection/search/38457?searchField=All&amp;sortBy=Relevance&amp;ft=thailand+buddha&amp;offset=0&amp;rpp=80&amp;pos=40', 
  'unknown_1500_bronze_cantor_2':'https://cantorcollections.stanford.edu/objects-1/thumbnails?records=80&query=mfs%20all%20%22thailand%20buddha%22&sort=0',
  'unknown_1500_bronze_cantor_3':'https://cantorcollections.stanford.edu/objects-1/thumbnails?records=80&query=mfs%20all%20%22thailand%20buddha%22&sort=0', 
  'unknown_1500_bronze_met':'https://www.metmuseum.org/art/collection/search/39149?searchField=All&amp;sortBy=Relevance&amp;ft=thailand+buddha&amp;offset=0&amp;rpp=80&amp;pos=33',
  'unknown_1700_bronze_cantor':'https://cantorcollections.stanford.edu/objects-1/thumbnails?records=80&query=mfs%20all%20%22thailand%20buddha%22&sort=0', 
  'unknown_buddha_1400_1500_bronze_met':'https://www.metmuseum.org/art/collection/search/39112?searchField=All&amp;sortBy=Relevance&amp;ft=thailand+buddha&amp;offset=0&amp;rpp=80&amp;pos=8', 
  'unknown_buddha_1400_bronze_met':'https://www.metmuseum.org/art/collection/search/63532?searchField=All&amp;sortBy=Relevance&amp;ft=thailand+buddha&amp;offset=0&amp;rpp=20&amp;pos=2', 
  'unknown_1400_copper_lacma':'https://collections.lacma.org/node/228013',
  'unknown_1600_bronze_walters':'https://art.thewalters.org/detail/17687/seated-buddha-in-maravijaya-8/'}

print(len(test))
testKeysSorted = sorted(test.keys(), key=lambda x:x.lower())
test1 ={}
for i in testKeysSorted:
    test1[i] = test[i]
    print(i)
