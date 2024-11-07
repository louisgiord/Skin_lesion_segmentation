import numpy as np

#Compute DICE to compare the similarity of two masks
#Comparison between our pipeline and the reference, between the segmentation method and the ground-truth segmentation

#Definition of the Jaccard Index (JI)
#Provides the similarity between two sets 
#Computed as the size of the intersection divided by the size of the size of the union 
#of the  segmentation mask

def JI (mask1,mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    ji = np.sum(intersection)/np.sum(union)
    return ji   

def dice(mask1,mask2):
    ji = JI(mask1,mask2)
    DICE = (2*ji)/(1+ji)
    return DICE

