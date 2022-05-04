# CSSLA DH assignment
The aim is to build a classifier to separate Sukhothai-style and Ayutthaya-style Buddha heads.

## Resource Folder Breakdown
```directory
|dh assignment
|_ imageConverter.py - Creates List objects for the images to be used from directory
|_ face_feature_extraction_library.py - Sift feature extraction
|_ svm_face_features.py - Bag of Words and support vector classifier
|_ main.py
|_ *.pkl (for serialised DataFrame and List objects that are imported into the python scripts)
|_ *.csv (these can be safely ignored as they are used for DataFrame dumps and debugging)
|_ siftPrediction.csv (contains the predictions for the test image set)
|
|_images 
|__ raw_thai_head (this is are the training and test set images before they are converted and not ready for processing)
|__ thai_head (contains images ready for processing)
|__ thai_head_mask (contains the 4 masks, for mouth, eyes and nose, hair, ears)
|
|_cropped_images
|__ thai_head (contains the processed test and training image sets)
|
|_ output_images
|__ thai_head (output masked images with the sift keypoints)
|__ test_classified (images with group labels 1 Ayutthaya, 0 Sukhothai)
```


## To run project
run file main.py 

# Report 
## Problem statement
Using computer vision and machine learning to separate the religious sculpture styles of Northern (Sukhothai) and Southern (Ayutthaya) Medieval Thailand. Images set are already images from these two periods and will be classified into one or the other.

## Image Set Overview
As we have 17 Ayutthaya images and 16 Sukhothai images, we have selected 5 Ayutthaya and 3 Sukhothai images to be part of the test group. We have 12 unknown images in the test image set. So we have 12 images of each classification in the training set and a total of 21 images in the test set.
In the ```thai_head```  folder the test image set has labelled with the prefix ```test```. Ayutthaya and Sukhothai images prefixes follow correspondingly, with ```ayutthaya``` and ```sukhothai```

## Searching for the training set
The challenge here is that art styles of particular societies flow with the movement of peoples and the ever changing borders. The period we are concerned with is 13th Century to the early 18th Century before the unification of the modern Thai State. There are two medieval Thai kingdoms that I am focusing on. Sukhothai considered to be a northern province, and the southern Ayutthaya. There is a third, kingdom of Lanna. It is important, the art of the Lanna Kingdom, should be treated as a separate category, this feudal state centred around Chang Mai, remained sovereign or at least a vassal state until the late 1700s. So we will not be considering artefacts that are labelled as Lanna.

It is common to find strong similarities between artefacts that are classified as Ayutthayan and Sukhothaiin period.  This is interesting as it poses a problem that where the two period styles share a significant number of facial features. The most common is the introduction of the rounder jawlines in the later Ayutthayan images following the Sukhothai styles and the curved arching eyebrows. In building the sample, I used the period classification done by the museum. Some sculptures that are identified as Ayutthaya period, but they are geographically from the northern cities like Sukhothai and Phitsanulok. For these sculptures I have labelled them as Sukhothai. Some sculptures that are from the Ayutthayan period but have been identified as having a Sukhothai style also receive the same treatment. 

|<img src=images\raw_thai_head\sukhothai_1500_bronze_walters_art.jpeg alt=sukhothaiImage width =200px>|<img src=images\raw_thai_head\ayutthaya_1500_copper_LACMA_cropped.jpeg alt=ayutthayaImage width =200px>|
|:--:|:--:|
|*1500 Sukhothai Example*|*1500 Ayutthaya Example*|

This overlap of styles poses the question of what truly differentiates the styles. This is the main motivation of the project to try and quantify the differences in the two styles.

### So what really is the Northern and Southern style?
It is likely the mixing between the Ayutthaya and Sukhothai kingdoms and court culture gradually allowed for significant overlaps in terms of the religious sculpture styles between the northern and later southern style. This poses a challenge in order to identify Sukhothai and late era Ayutthaya religious imagery. 

The Sukhothai kingdom that had its heyday in the late 14th century. Although it disappeared as a state it has a longer cultural impact of its southern neighbours. The monk order from Sukhothai have some roots in Sri Lanka, this authenticity lent the order authority. (p37) There are stories about the religious awakening brought about by returning monks form Sri Lanka. There was significant royal patronage of the monastic order. (p39) This religious zeal is seen with many bronze Buddha images in their distinctive style. This lent a very strong religious culture that allowed it to influence its neighbours, in this project's case, in terms of religious imagery. 

This far reaching influence of the northern Sukhothai style can be traced through the history of the two kingdoms, that eventually merged under an Ayutthaya-based but Sukhothaiin dominant aristocracy in 1569. (p 74- 78, Baker & Pasuk) 

In the early 1300s, we find a mercantile Ayutthaya port city. There is less prominence to religious culture in this period. Thus less monuments and religious imagery unlike its northern neighbour. The style of imagery is the U Thong style after the founding Ayutthayan king. (p57-58, Baker & Pasuk) This style is very differnt from the northern and mid-to-late Ayutthayan period images. The sculpture follows earlier Ankor characteristics, with the square head and jaw. There is strong strait brow. (p82, Baker & Pasuk) 

|<img src=images\raw_thai_head\ayutthaya_buddha_1347_bronze_NGAustralia.jpeg alt=sukhothaiImage width =200px>|<img src=images\raw_thai_head\ayutthaya_1700_bronze_smithsonian_2.jpeg alt=ayutthayaImage height =160>|
|:--:|:--:|
|*1300 Ayutthaya Example*|*1700 Ayutthaya Example*|

The 15th century brought intermarrying of the royal Sukhothai and Ayutthaya families. In the 16th century through intense court intrigue, saw a political change with a Suphanburi--Sukhothai family coming to dominate the Ayutthayan court. A series of bad wars with the invading Pagan left Ayutthaya in weak state, and the northern allies who did not partake in the war, swooped in to occupy and rule the city after the much of the old Ayutthayan elite were exiled. (p74-78, Baker & Pasuk) 

In this period we see increase in the prominence of religion in the south. There are now more copies of the Sihing Buddha and the Sukhothai walking Buddha. There is strong patronage of the religious imagery and temples. As the old U Thong style is pushed out, a compromise between the Sukhothai and U Thong styles appears. We see the oval face of Sukhothai, with is arching eyebrows, but there is element of 'robustness' that is lacking in the older Sukhothai imagery. (p81-82, Baker & Pasuk) 

And this 'robustness' would need to be quantified for the support vector machine to do its magic. This is done using the scale invariant feature transform from the opencv python library for feature extraction. 

### Challenges faced with the training set
The biggest issue with the training set, is the lack of quality images. In the search, we found a problem with the provenance of some artefacts. In such cases the donation lacks a clear or reputable acquisition of the artefact, and this affects the ability to correctly identify the period or manufacturing city. 

>Smithsonian Museum has a 4 Buddha heads donated by a private collector who said they are from Angkor Wat but they are clearly of Thai origin. The museum has identified them to be 1700 Ayuttayan period. 

|<img src=images\raw_thai_head\ayutthaya_1700_bronze_smithsonian_2.jpeg alt=sukhothaiImage height =200>|<img src=images\raw_thai_head\ayutthaya_1700_bronze_smithsonian.jpeg alt=ayutthayaImage height =200>|
|:--:|:--:|
|*Troublesome images*|

These private donations often have complicated histories in how it came into private ownership origins which means that they are usually left unlabelled, so we cannot concretely determine their period or style. This contributes to the small size of the training set and test set. These unclassified artefacts form the second motivation for pursuing this project, adding to the body of knowledge by interpreting these unclassified artefacts.

The images available may have different lighting set ups, those taken with spotlight lighting may create the illusion of a sharper edges along the shadows. This can lead to erroneous sift key points. 
Some images have high compression noise and this also affects the effectiveness of the sift algorithm to identify key points. 

In the end any result of a small sample will inherently be less meaningful. 

## Feature extraction
### SIFT
The scale invariant feature transform is useful in finding features even with different levels of image compression. One alternative that was considered was histogram of gradients. However, the histogram of gradients struggles with the different compression noise levels and results in inconsistent histogram sizes, which is not ideal for comparison. Each sift key points are described with 128 long histogram. This sift descriptor histogram describes the vectoring of the magnitude of the pixel brightness around the key point. 

### Masking images
There are four masks are for the different regions of the face. They are for *hair*, *mouth*, *eyebrows and nose*, *ears*. We apply the 4 masks to create 4 separate images representing regions of the image and extract sift key points from each area. The choice of masking the images is so that we can prevent matching of key points erroneously. So that a hair key point will not be compared to a mouth key point, making the successful sift key point matches more meaningful. We are only using the *mouth* and *eyes and nose* masks. We are ignoring the ear and hair masks. Some of the sculptures have broken ears and we do not want to differentiate sculptures based on damage incurred. We do not want to use the hair mask because some of the images have crowns, this is not a clear identifier of the style, so we do not want to use it the differentiate the images by the headwear.

### Bag of Words method

- First let us consider the process for the Ayutthaya classification. 
- This method identifies the features of the masked training image, as our *vocabulary*.
- We combine the training images so we have two composite images, one for the *mouth* and one for the *eyes and nose*. These composite images contain all the sift key points for each of the masks.
- Using k-means clustering we build a set of n clusters of key points based on their sift descriptors, 
  - n is by default 200
  - these are the words or key descriptors for each masked region of the face
  - we take the centre of each cluster, known as centroids
- By default for each mask of the training set, we get a 200 long *vocabulary*. 
- In total this training set has a 400 long *vocabulary*

- Using all the image sets, we compare each image to the *vocabulary* by calculating each sift key points' *hamming* distance to each word in the *vocabulary*.
  - *Hamming* algorithm is chosen as it can handle the vector aspect of the sift key point descriptor and its high dimensionality which Euclidean distances struggle with. 
  - The word that has the minimum distance to each of the masked test image sift key points is recorded.
- Each masked image returns a 400 long histogram which marks the word frequency.
- This histogram acts as a descriptor of our image in comparison to the composite training set image.
- Do note this is done for both the training and test image set.
- This is what is used in the support vector classifier.
  
Output of the bag of words method
- We have training and test images which have by default a 400 long histogram descriptor showing word frequency from the Ayutthaya training set
- We repeat the process with the Sukhothai images as well. 
- So each image has 800 long histogram descriptor.
- This histogram behaves like a vector, so the order of the masks and the Ayutthaya and Sukhothai *vocabulary* should be the consistent.

### Term Frequency - Inverse Document Frequency 
Now we have the test image descriptors. The histogram contains the term frequency in each test image. 
However some terms may be common across the different test images. To improve the quality of the support vector classifier, we want to reduce the influence of common terms in the support vector classifier and increase the influence of distinct terms.
Using the Sklearn Tfidf function, we can input the images' histogram descriptors and return a 800 long weighted vector to apply to the test X data.
The multiplication of this vector increases the magnitude of the distinct features in the descriptor histogram.

## Support Vector Classification
A Gaussian kernel was used in the SVC. As discussed above, each image has a 800 long weighted histogram to describe it. Looking at our trainingX raw values.

Values|max    |min|mean  |median|
|:--:|:--:|:--:|:--:|:--:|
|Raw|651.451|0  |1.3441|0     |
|Standard|51.114|-2.368|-0.09037|
|Cube root|3.7112|-0.024395|-0.44876|

We see that majority of the values are 0 and the standard deviation for larger values is large, so after standardising, we apply a cube root function to reduce the magnitude of the large values, so the max values are less extreme, allowing for more meaningful results in our classifier.

As we have 17 Ayutthaya images and 16 Sukhothai images, we have selected 5 Ayutthaya and 3 Sukhothai images to be part of the test group. We have 12 unknown images in the test image set. So we have 12 images of each classification in the training set and a total of 21 images in the test set.

Currently the classification misclassifies 2 out of 4 of the Sukhothai images and 1 out of 5 of the Ayutthaya images.

This is likely due to the above discussed challenges of the small and varied lighting and noise in this project's data set. 

## Understanding the classification
|<img src=images/thai_head/test_sukhothai_walking_1400_bronze_walters.jpeg width =200px>|<img src=images/thai_head/test_sukhothai_ayutthaya_1500_1600_copper_standing_ACM.jpeg width=200px>|<img src=images/thai_head/test_sukhothai_buddha_1300_copper_LACMA.jpeg width =200px>| <img src=images/thai_head/test_sukhothai_1650_bronze_phitsanulok_walters.jpeg width=200px>|
|:--:|:--:|:--:|:--:|
|*Correct Sukhothai Classification*|*Correct Sukhothai Classification*|*Wrong Sukhothai Classification*|*Wrong Sukhothai Classification*|

|<img src=images/thai_head/test_ayutthaya_1300_u_thong_bronze_NSW.jpeg width =200px>|<img src=images/thai_head/test_ayutthaya_1500_copper_LACMA_cropped.jpeg width=200px>|<img src=images/thai_head/test_ayutthaya_1700_hollow_cast_bronze_crowned_walters.jpeg width =200px>|<img src=images/thai_head/test_ayutthaya_u_thong_1300_1400_bronze_walters.jpeg width=200px>|<img src=images/thai_head/test_ayutthaya_1600_1625_bronze_walters.jpeg width=200px>|
|:--:|:--:|:--:|:--:|:---:|
|*Correct Ayutthaya Classification*|*Correct Ayutthaya Classification*|*Correct Ayutthaya Classification*|*Correct Ayutthaya Classification*|*Wrong Ayutthaya Classification*|

Let us compare the sift key points on the different masks we are using. 

### Mouth
|<img src=output_images\thai_head\sift_test_sukhothai_walking_1400_bronze_walters_mouth_mask.jpeg width =200px>|<img src=output_images\thai_head\sift_test_sukhothai_ayutthaya_1500_1600_copper_standing_ACM_mouth_mask.jpeg width=200px>|<img src=output_images/thai_head/sift_test_sukhothai_buddha_1300_copper_LACMA_mouth_mask.jpeg width =200px>| <img src=output_images\thai_head\sift_test_sukhothai_1650_bronze_phitsanulok_walters_mouth_mask.jpeg width=200px>|
|:--:|:--:|:--:|:--:|
|*Correct Sukhothai Classification*|*Correct Sukhothai Classification*|*Wrong Sukhothai Classification*|*Wrong Sukhothai Classification*|

|<img src=output_images/thai_head/sift_test_ayutthaya_1300_u_thong_bronze_NSW_mouth_mask.jpeg width =200px>|<img src=output_images/thai_head/sift_test_ayutthaya_1500_copper_LACMA_cropped_mouth_mask.jpeg width=200px>|<img src=output_images//thai_head/sift_test_ayutthaya_1700_hollow_cast_bronze_crowned_walters_mouth_mask.jpeg width =200px>|<img src=output_images/thai_head/sift_test_ayutthaya_u_thong_1300_1400_bronze_walters_mouth_mask.jpeg width=200px>|<img src=output_images//thai_head/sift_test_ayutthaya_1600_1625_bronze_walters_mouth_mask.jpeg width=200px>|
|:--:|:--:|:--:|:--:|:---:|
|*Correct Ayutthaya Classification*|*Correct Ayutthaya Classification*|*Correct Ayutthaya Classification*|*Correct Ayutthaya Classification*|*Wrong Ayutthaya Classification*|

For the mouth mask, it appears that the sharp jawline of the Ayutthayan sculptures cause a shadow to cast directly under the chin creating a line of key points at the edge of this shadow. It appears that the those classified as Ayutthayan have a sharp line below the chin. 

### Eyebrows and nose
|<img src=output_images\thai_head\sift_test_sukhothai_walking_1400_bronze_walters_eyebrow_nose_mask.jpeg width =200px>|<img src=output_images\thai_head\sift_test_sukhothai_ayutthaya_1500_1600_copper_standing_ACM_eyebrow_nose_mask.jpeg width=200px>|<img src=output_images/thai_head/sift_test_sukhothai_buddha_1300_copper_LACMA_eyebrow_nose_mask.jpeg width =200px>| <img src=output_images\thai_head\sift_test_sukhothai_1650_bronze_phitsanulok_walters_eyebrow_nose_mask.jpeg width=200px>|
|:--:|:--:|:--:|:--:|
|*Correct Sukhothai Classification*|*Correct Sukhothai Classification*|*Wrong Sukhothai Classification*|*Wrong Sukhothai Classification*|

|<img src=output_images/thai_head/sift_test_ayutthaya_1300_u_thong_bronze_NSW_eyebrow_nose_mask.jpeg width =200px>|<img src=output_images/thai_head/sift_test_ayutthaya_1500_copper_LACMA_cropped_eyebrow_nose_mask.jpeg width=200px>|<img src=output_images//thai_head/sift_test_ayutthaya_1700_hollow_cast_bronze_crowned_walters_eyebrow_nose_mask.jpeg width =200px>|<img src=output_images/thai_head/sift_test_ayutthaya_u_thong_1300_1400_bronze_walters_eyebrow_nose_mask.jpeg width=200px>|<img src=output_images//thai_head/sift_test_ayutthaya_1600_1625_bronze_walters_eyebrow_nose_mask.jpeg width=200px>|
|:--:|:--:|:--:|:--:|:---:|
|*Correct Ayutthaya Classification*|*Correct Ayutthaya Classification*|*Correct Ayutthaya Classification*|*Correct Ayutthaya Classification*|*Wrong Ayutthaya Classification*|

For the eyebrows and nose, there is strong similarity between the two styles. From our sift key point detection it is not obvious qualitatively how the two classifications differ.
Perhaps more ideally the images could be in profile or three-quarters to better capture the face structure in the jawline and cheek bones.



__________________________
## Resources 
Opencv, Pandas, Numpy and Sklearn documentation

Baker, C., & Phongpaichit, P. (2017). A History of Ayutthaya: Siam in the Early Modern World. Cambridge University Press.
### Resources
- NY Met Museum
- LACMA
- Cantor Arts Center 
- Victoria Albert Museum
- Walters Art Gallery
- British Museum
- ACM Singapore
- Arts_Culture_Google using screenshots

### Itemised list
```python 
  
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
  'ayutthaya_1700_bronze_smithsonian':'https://asia.si.edu/object/F1909.49/',



  
  'sukhothai_1300_1400_bronze_british_museum':'https://www.britishmuseum.org/collection/object/A_1880-1002',
  'sukhothai_1550_1600_bronze_walters':'https://art.thewalters.org/detail/15634/seated-buddha-in-maravijaya-6/',
  'sukhothai_1550_bronze_walters_art':'https://art.thewalters.org/detail/39948/head-of-the-buddha-17/', 
  'sukhothai_1584_metal_ACM':'https://www.roots.gov.sg/Collection-Landing/listing/1245570', 
  'sukhothai_1584_metal_ACM_personal':'taken on acm trip', 
  'sukhothai_1650_1700_bronze_walters':'https://art.thewalters.org/detail/16172/', 
  'sukhothai_1600_1700_bronze_ACM':'https://www.roots.gov.sg/Collection-Landing/listing/1120177', 
  'sukhothai_1600_1700_bronze_ACM_personal':'taken on acm trip', 
  'sukhothai_1600_bronze_walters_2':'https://art.thewalters.org/detail/36196/', 
  'sukhothai_1650_bronze_phitsanulok':'https://art.thewalters.org/detail/37881/buddha-at-the-moment-of-victory-4/', 
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
  'unknown_1600_bronze_walters':'https://art.thewalters.org/detail/17687/seated-buddha-in-maravijaya-8/'
}

```