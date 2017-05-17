
# coding: utf-8

# In[1]:

# Imports and defines
import numpy as np
import sys
import time

import cv2
import os
import glob
import opengm
import psutil
import datetime

import matplotlib as mpl

import CMT

from PIL import Image
from PySide import QtCore, QtGui

import VideoTexturesUtils as vtu
import GraphWithValues as gwv


DICT_SEQUENCE_NAME = 'semantic_sequence_name'
DICT_BBOXES = 'bboxes'
DICT_FOOTPRINTS = 'footprints' ## same as bboxes but it indicates the footprint of the sprite on the ground plane
DICT_BBOX_ROTATIONS = 'bbox_rotations'
DICT_BBOX_CENTERS = 'bbox_centers'
DICT_FRAMES_LOCATIONS = 'frame_locs'
DICT_MASK_LOCATION = 'frame_masks_location'
DICT_ICON_TOP_LEFT = "icon_top_left"
DICT_ICON_FRAME_KEY = "icon_frame_key"
DICT_ICON_SIZE = "icon_size"
DICT_REPRESENTATIVE_COLOR = 'representative_color'
DICT_FRAME_SEMANTICS = "semantics_per_frame"
DICT_SEMANTICS_NAMES = "semantics_names"
DICT_NUM_SEMANTICS = "number_of_semantic_classes"
DICT_PATCHES_LOCATION = "sequence_preloaded_patches_location"
DICT_TRANSITION_COSTS_LOCATION = "sequence_precomputed_transition_costs_location"
# DICT_FRAME_COMPATIBILITY_LABELS = 'compatibiliy_labels_per_frame'
DICT_LABELLED_FRAMES = 'labelled_frames' ## includes the frames labelled for the semantic labels (the first [DICT_FRAME_SEMANTICS].shape[1])
DICT_NUM_EXTRA_FRAMES = 'num_extra_frames' ## same len as DICT_LABELLED_FRAMES
DICT_CONFLICTING_SEQUENCES = 'conflicting_sequences'
DICT_DISTANCE_MATRIX_LOCATION = 'sequence_precomputed_distance_matrix_location' ## for label propagation
DICT_SEQUENCE_LOCATION = "sequence_location"

TL_IDX = 0
TR_IDX = 1
BR_IDX = 2
BL_IDX = 3
GRAPH_MAX_COST = 10000000.0

HANDLE_MOVE_SEGMENT = 0
HANDLE_ROTATE_BBOX = 1
MOVE_SEGMENT_HANDLE_SIZE = 20
ROTATE_BBOX_HANDLE_SIZE = 10

## used for enlarging bbox used to decide size of patch around it (percentage)
PATCH_BORDER = 0.4


# In[ ]:

## compute euclidean distance assuming f is an array where each row is a flattened image (1xN array, N=W*H*Channels)
## euclidean distance defined as the length of the the displacement vector:
## len(q-p) = sqrt(len(q)^2+len(p)^2 - 2*dot(p, q)) where p and q are two images in vector format and 1xN size
def ssd(f) :
    ## gives sum over squared intensity values for each image
    ff = np.sum(f*f, axis=1)
    ## first term is sum between each possible combination of frames
    ## second term is the the dot product between each frame as in the formula above
    d = np.reshape(ff, [len(ff),1])+ff.T - 2*np.dot(f, f.T)
    return d

def ssd2(f1, f2) :
    ## gives sum over squared intensity values for each image
    ff1 = np.sum(f1*f1, axis=1)
    ff2 = np.sum(f2*f2, axis=1)
#     print ff1.shape
#     print ff2.shape
    ## first term is sum between each possible combination of frames
    ## second term is the the dot product between each frame as in the formula above
#     print "askdfh", np.repeat(np.reshape(ff1, [len(ff1),1]), len(ff2), axis=1).shape, np.repeat(np.reshape(ff2, [1, len(ff2)]), len(ff1), axis=0).shape
    d = np.repeat(np.reshape(ff1, [len(ff1),1]), len(ff2), axis=1)+np.repeat(np.reshape(ff2, [1, len(ff2)]), len(ff1), axis=0) - 2*np.dot(f1, f2.T)
    return d


# In[3]:

def multivariateNormal(data, mean, var, normalized = True) :
    if (data.shape[0] != mean.shape[0] or np.any(data.shape[0] != np.array(var.shape)) 
        or len(var.shape) != 2 or var.shape[0] != var.shape[1]) :
        raise Exception("Data shapes don't agree data(" + np.string_(data.shape) + ") mean(" + np.string_(mean.shape) + 
                        ") var(" + np.string_(var.shape) + ")")
        
    D = float(data.shape[0])
    n = (1/(np.power(2.0*np.pi, D/2.0)*np.sqrt(np.linalg.det(var))))
    if normalized :
        p = n*np.exp(-0.5*np.sum(np.dot((data-mean).T, np.linalg.inv(var))*(data-mean).T, axis=-1))
    else :
        p = np.exp(-0.5*np.sum(np.dot((data-mean).T, np.linalg.inv(var))*(data-mean).T, axis=-1))
        
    return p

def minusLogMultivariateNormal(data, mean, var, normalized = True) :
    if (data.shape[0] != mean.shape[0] or np.any(data.shape[0] != np.array(var.shape)) 
        or len(var.shape) != 2 or var.shape[0] != var.shape[1]) :
        raise Exception("Data shapes don't agree data(" + np.string_(data.shape) + ") mean(" + np.string_(mean.shape) + 
                        ") var(" + np.string_(var.shape) + ")")
    
    D = float(data.shape[0])
    n = -0.5*np.log(np.linalg.det(var))-(D/2.0)*np.log(2.0*np.pi)
    if normalized :
        p = n -0.5*np.sum(np.dot((data-mean).T, np.linalg.inv(var))*(data-mean).T, axis=-1)
    else :
        p = -0.5*np.sum(np.dot((data-mean).T, np.linalg.inv(var))*(data-mean).T, axis=-1)
        
    return -p

def vectorisedMinusLogMultiNormal(dataPoints, means, var, normalized = True) :
    if (dataPoints.shape[1] != means.shape[1] or np.any(dataPoints.shape[1] != np.array(var.shape)) 
        or len(var.shape) != 2 or var.shape[0] != var.shape[1]) :
        raise Exception("Data shapes don't agree data(" + np.string_(dataPoints.shape) + ") mean(" + np.string_(means.shape) + 
                        ") var(" + np.string_(var.shape) + ")")
    
    D = float(dataPoints.shape[1])
    n = -0.5*np.log(np.linalg.det(var))-(D/2.0)*np.log(2.0*np.pi)
    
    ## this does 0.5*dot(dot(data-mean, varInv), data-mean)
    varInv = np.linalg.inv(var)
    dataMinusMean = dataPoints-means
    
    ps = []
    for i in xrange(int(D)) :
        ps.append(np.sum((dataMinusMean)*varInv[:, i], axis=-1))
    
    ps = np.array(ps).T
    
    ps = -0.5*np.sum(ps*(dataMinusMean), axis=-1)
    
    if normalized :
        return n-ps
    else :
        return -ps
    
def getGridPairIndices(width, height) :
## deal with pixels that have East and South neighbours i.e. all of them apart from last column and last row
    pairIdxs = np.zeros(((width*height-(width+height-1))*2, 2), dtype=int)
## each column contains idxs [0, h-2]
    idxs = np.arange(0, height-1, dtype=int).reshape((height-1, 1)).repeat(width-1, axis=-1)
## each column contains idxs [0, h-2]+h*i where i is the column index 
## (i.e. now I have indices of all nodes in the grid apart from last col and row)
    idxs += (np.arange(0, width-1)*height).reshape((1, width-1)).repeat(height-1, axis=0)
    # figure(); imshow(idxs)
## now flatten idxs and repeat once so that I have the idx for each node that has E and S neighbours twice
    idxs = np.ndarray.flatten(idxs.T).repeat(2)
## idxs for each "left" node (that is connected to the edge) are the ones just computed
    pairIdxs[:, 0] = idxs
## idxs for each "right" node are to the E and S so need to sum "left" idx to height and to 1
# print np.ndarray.flatten(np.array([[patchSize[0]], [1]]).repeat(np.prod(patchSize)-(np.sum(patchSize)-1), axis=-1).T)
    pairIdxs[:, 1] = idxs + np.ndarray.flatten(np.array([[height], [1]]).repeat(width*height-(width+height-1), axis=-1).T)
    
## deal with pixels that have only East neighbours
## get "left" nodes
    leftNodes = np.arange(height-1, width*height-1, height)
## now connect "left" nodes to the nodes to their East (i.e. sum to height) and add them to the list of pair indices
    pairIdxs = np.concatenate((pairIdxs, np.array([leftNodes, leftNodes+height]).T), axis=0)
    
## deal with pixels that have only South neighbours
## get "top" nodes
    topNodes = np.arange(width*height-height, width*height-1)
## now connect "to" nodes to the nodes to their South (i.e. sum to 1) and add them to the list of pair indices
    pairIdxs = np.concatenate((pairIdxs, np.array([topNodes, topNodes+1]).T), axis=0)
    
    return pairIdxs

def getGraphcutOnOverlap(patchA, patchB, patchAPixels, patchBPixels, multiplier, unaryPriorPatchA, unaryPriorPatchB,
                         patchAGradX = None, patchAGradY = None, patchBGradX = None, patchBGradY = None) :
    """Computes pixel labels using graphcut given two same size patches
    
        \t  patchA           : patch A
        \t  patchB           : patch B
        \t  patchAPixels     : pixels that are definitely to be taken from patch A
        \t  patchBPixels     : pixels that are definitely to be taken from patch B
        \t  multiplier       : sigma multiplier for rgb space normal
        \t  unaryPriorPatchA : prior cost ditribution for patchA labels
        \t  unaryPriorPatchB : prior cost ditribution for patchB labels
           
        return: reshapedLabels = labels for each pixel"""
    
    t = time.time()
    if np.all(patchA.shape != patchB.shape) :
        raise Exception("The two specified patches have different shape so graph cannot be built")
        
    if patchA.dtype != np.float64 or patchB.dtype != np.float64 :
        raise Exception("The two specified patches are not of type float64! Check there is no overflow when computing costs")
    
    h, width = patchA.shape[0:2]
    maxCost = 10000000.0#np.sys.float_info.max
    
    s = time.time()
    ## build graph
    numLabels = 2
    numNodes = h*width+numLabels
    gm = opengm.gm(np.ones(numNodes,dtype=opengm.label_type)*numLabels)
    
    ## Last 2 nodes are patch A and B respectively
    idxPatchANode = numNodes - 2
    idxPatchBNode = numNodes - 1
    
        
    ## get unary functions
    unaries = np.zeros((numNodes,numLabels))
    
    ## fix label for nodes representing patch A and B to have label 0 and 1 respectively
    unaries[idxPatchANode, :] = [0.0, maxCost]
    unaries[idxPatchBNode, :] = [maxCost, 0.0]
    
    ## set unaries based on the priors given for both patches
    unaries[0:h*width, 0] = unaryPriorPatchA
    unaries[0:h*width, 1] = unaryPriorPatchB
    
    # add functions
    fids = gm.addFunctions(unaries)
    # add first order factors
    gm.addFactors(fids, np.arange(0, numNodes, 1))
    
    
    ## get factor indices for the overlap grid of pixels
    stmp = time.time()
#     pairIndices = np.array(opengm.secondOrderGridVis(width,h,True))
    pairIndices = getGridPairIndices(width, h)
#     print "pairIndices took", time.time()-stmp, "seconds"
#     sys.stdout.flush()
    ## get pairwise functions for those nodes
#     pairwise = np.zeros(len(pairIndices))
#     for pair, i in zip(pairIndices, np.arange(len(pairIndices))) :
#         sPix = np.array([int(np.mod(pair[0],h)), int(pair[0]/h)])
#         tPix = np.array([int(np.mod(pair[1],h)), int(pair[1]/h)])
        
# #         pairwise[i] = norm(patchA[sPix[0], sPix[1], :] - patchB[sPix[0], sPix[1], :])
# #         pairwise[i] += norm(patchA[tPix[0], tPix[1], :] - patchB[tPix[0], tPix[1], :])

#         pairwise[i] = minusLogMultivariateNormal(patchA[sPix[0], sPix[1], :].reshape((3, 1)), patchB[sPix[0], sPix[1], :].reshape((3, 1)), np.eye(3)*multiplier, False)
#         pairwise[i] += minusLogMultivariateNormal(patchA[tPix[0], tPix[1], :].reshape((3, 1)), patchB[tPix[0], tPix[1], :].reshape((3, 1)), np.eye(3)*multiplier, False)
        
#         fid = gm.addFunction(np.array([[0.0, pairwise[i]],[pairwise[i], 0.0]]))
#         gm.addFactor(fid, pair)
        
    sPixs = np.array([np.mod(pairIndices[:, 0],h), pairIndices[:, 0]/h], dtype=int).T
    tPixs = np.array([np.mod(pairIndices[:, 1],h), pairIndices[:, 1]/h], dtype=int).T
    
    pairwise = vectorisedMinusLogMultiNormal(patchA[sPixs[:, 0], sPixs[:, 1], :], patchB[sPixs[:, 0], sPixs[:, 1], :], np.eye(3)*multiplier, False)
    pairwise += vectorisedMinusLogMultiNormal(patchA[tPixs[:, 0], tPixs[:, 1], :], patchB[tPixs[:, 0], tPixs[:, 1], :], np.eye(3)*multiplier, False)
#     print np.min(pairwise), np.max(pairwise), pairwise
    if False and patchAGradX != None and patchAGradY != None and patchBGradX != None and patchBGradY != None :
#         pairwise /= ((vectorisedMinusLogMultiNormal(patchAGradX[sPixs[:, 0], sPixs[:, 1], :], np.zeros_like(patchAGradX[sPixs[:, 0], sPixs[:, 1], :]), np.eye(3)*multiplier, False)+
#                      vectorisedMinusLogMultiNormal(patchAGradX[tPixs[:, 0], tPixs[:, 1], :], np.zeros_like(patchAGradX[tPixs[:, 0], tPixs[:, 1], :]), np.eye(3)*multiplier, False)+
#                      vectorisedMinusLogMultiNormal(patchBGradX[sPixs[:, 0], sPixs[:, 1], :], np.zeros_like(patchBGradX[sPixs[:, 0], sPixs[:, 1], :]), np.eye(3)*multiplier, False)+
#                      vectorisedMinusLogMultiNormal(patchBGradX[sPixs[:, 0], sPixs[:, 1], :], np.zeros_like(patchBGradX[tPixs[:, 0], tPixs[:, 1], :]), np.eye(3)*multiplier, False)+
#                      vectorisedMinusLogMultiNormal(patchAGradY[sPixs[:, 0], sPixs[:, 1], :], np.zeros_like(patchAGradY[sPixs[:, 0], sPixs[:, 1], :]), np.eye(3)*multiplier, False)+
#                      vectorisedMinusLogMultiNormal(patchAGradY[tPixs[:, 0], tPixs[:, 1], :], np.zeros_like(patchAGradY[tPixs[:, 0], tPixs[:, 1], :]), np.eye(3)*multiplier, False)+
#                      vectorisedMinusLogMultiNormal(patchBGradY[sPixs[:, 0], sPixs[:, 1], :], np.zeros_like(patchBGradY[sPixs[:, 0], sPixs[:, 1], :]), np.eye(3)*multiplier, False)+
#                      vectorisedMinusLogMultiNormal(patchBGradY[sPixs[:, 0], sPixs[:, 1], :], np.zeros_like(patchBGradY[tPixs[:, 0], tPixs[:, 1], :]), np.eye(3)*multiplier, False))/1000.0+0.00001)
        denominator = (np.sqrt(np.sum(patchAGradX[sPixs[:, 0], sPixs[:, 1], :]**2, axis=-1))+
                     np.sqrt(np.sum(patchAGradX[tPixs[:, 0], tPixs[:, 1], :]**2, axis=-1))+
                     np.sqrt(np.sum(patchBGradX[sPixs[:, 0], sPixs[:, 1], :]**2, axis=-1))+
                     np.sqrt(np.sum(patchBGradX[sPixs[:, 0], sPixs[:, 1], :]**2, axis=-1))+
                     np.sqrt(np.sum(patchAGradY[sPixs[:, 0], sPixs[:, 1], :]**2, axis=-1))+
                     np.sqrt(np.sum(patchAGradY[tPixs[:, 0], tPixs[:, 1], :]**2, axis=-1))+
                     np.sqrt(np.sum(patchBGradY[sPixs[:, 0], sPixs[:, 1], :]**2, axis=-1))+
                     np.sqrt(np.sum(patchBGradY[sPixs[:, 0], sPixs[:, 1], :]**2, axis=-1)))
    
        pairwise /= ((np.max(denominator) - denominator)+0.000001)
    
#     print np.min(pairwise), np.max(pairwise), pairwise
    fids = gm.addFunctions(np.array([[0.0, 1.0],[1.0, 0.0]]).reshape((1, 2, 2)).repeat(len(pairwise), axis=0)*
                           pairwise.reshape((len(pairwise), 1, 1)).repeat(2, axis=1).repeat(2, axis=2))
    
    gm.addFactors(fids, pairIndices)
            
    
    # add function used for connecting the patch variables
    fid = gm.addFunction(np.array([[0.0, maxCost],[maxCost, 0.0]]))
    
    # connect patch A to definite patch A pixels
    if len(patchAPixels) > 0 :
        patchAFactors = np.hstack((patchAPixels.reshape((len(patchAPixels), 1)), np.ones((len(patchAPixels), 1), dtype=np.uint)*idxPatchANode))
        gm.addFactors(fid, patchAFactors)
    
    # connect patch B to definite patch B pixels
    if len(patchBPixels) > 0 :
        patchBFactors = np.hstack((patchBPixels.reshape((len(patchBPixels), 1)), np.ones((len(patchBPixels), 1), dtype=np.uint)*idxPatchBNode))
        gm.addFactors(fid, patchBFactors)
    
#     print "graph setup", time.time() - t
    t = time.time()
#     print "graph setup took", time.time()-s, "seconds"
#     sys.stdout.flush()
    s = time.time()
    graphCut = opengm.inference.GraphCut(gm=gm)
    graphCut.infer()
#     print "graph inference took", time.time()-s, "seconds"
#     sys.stdout.flush()
    
    labels = np.array(graphCut.arg(), dtype=int)
    
    reshapedLabels = np.reshape(np.copy(labels[0:-numLabels]), patchA.shape[0:2], 'F')
    
#     print "solving", time.time() - t
    t = time.time()
#     print gm
#     print gm.evaluate(labels)
    
    return reshapedLabels, unaries, pairwise, gm


# In[4]:

def getForegroundPatch(trackedSequence, frameKey, frameWidth, frameHeight) :
    """Computes foreground patch based on its bbox
    
        \t  trackedSequence : dictionary containing relevant data for the tracked sequence we want a foreground patch of
        \t  frameKey        : the key of the frame the foreground patch should be taken from
        \t  frameWidth      : width of original image
        \t  frameHeight     : height of original image
           
        return: fgPatch, offset, patchSize,
                [left, top, bottom, right] : array of booleans telling whether the expanded bbox touches the corresponding border of the image"""
    
    ## get the bbox for the current frame, make it larger and find the rectangular patch to work with
    ## boundaries of the patch [min, max]
    
    ## returns foreground patch based on bbox and returns it along with the offset [x, y] and it's size [rows, cols]
    
    ## make bbox bigger
    largeBBox = trackedSequence[DICT_BBOXES][frameKey].T
    ## move to origin
    largeBBox = np.dot(np.array([[-trackedSequence[DICT_BBOX_CENTERS][frameKey][0], 1.0, 0.0], 
                                 [-trackedSequence[DICT_BBOX_CENTERS][frameKey][1], 0.0, 1.0]]), 
                        np.vstack((np.ones((1, largeBBox.shape[1])), largeBBox)))
    ## make bigger
    largeBBox = np.dot(np.array([[0.0, 1.0 + PATCH_BORDER, 0.0], 
                                 [0.0, 0.0, 1.0 + PATCH_BORDER]]), 
                        np.vstack((np.ones((1, largeBBox.shape[1])), largeBBox)))
    ## move back tooriginal center
    largeBBox = np.dot(np.array([[trackedSequence[DICT_BBOX_CENTERS][frameKey][0], 1.0, 0.0], 
                                 [trackedSequence[DICT_BBOX_CENTERS][frameKey][1], 0.0, 1.0]]), 
                        np.vstack((np.ones((1, largeBBox.shape[1])), largeBBox)))
    
    xBounds = np.zeros(2); yBounds = np.zeros(2)
    
    ## make sure xBounds are in between 0 and width and yBounds are in between 0 and height
    xBounds[0] = np.max((0, np.min(largeBBox[0, :])))
    xBounds[1] = np.min((frameWidth, np.max(largeBBox[0, :])))
    yBounds[0] = np.max((0, np.min(largeBBox[1, :])))
    yBounds[1] = np.min((frameHeight, np.max(largeBBox[1, :])))
    
    offset = np.array([np.round(np.array([xBounds[0], yBounds[0]]))], dtype=int).T # [x, y]
    patchSize = np.array(np.round(np.array([yBounds[1]-yBounds[0], xBounds[1]-xBounds[0]])), dtype=int) # [rows, cols]
    
    fgPatch = np.array(Image.open(trackedSequence[DICT_FRAMES_LOCATIONS][frameKey]))[offset[1]:offset[1]+patchSize[0], offset[0]:offset[0]+patchSize[1], :]
    
    return fgPatch, offset, patchSize, [np.min((largeBBox)[0, :]) > 0.0 ,
                                        np.min((largeBBox)[1, :]) > 0.0 ,
                                        np.max((largeBBox)[1, :]) < frameHeight,
                                        np.max((largeBBox)[0, :]) < frameWidth]


def getPatchPriors(bgPatch, fgPatch, offset, patchSize, trackedSequence, frameKey, framesLoc, allXs, allYs, prevFrameKey = None, prevFrameAlphaLoc = "",
                   prevMaskImportance = 0.8, prevMaskDilate = 13, prevMaskBlurSize = 31, prevMaskBlurSigma = 2.5,
                   diffPatchImportance = 0.015, diffPatchMultiplier = 1000.0, useOpticalFlow = True, useDiffPatch = False) :
    """Computes priors for background and foreground patches
    
        \t  bgPatch             : background patch
        \t  fgPatch             : foreground patch
        \t  offset              : [x, y] position of patches in the coordinate system of the original images
        \t  patchSize           : num of [rows, cols] per patches
        \t  trackedSequence     : dictionary containing relevant data for the tracked sequence need the patch priors for
        \t  frameKey            : the key of the frame the foreground patch should be taken from
        \t  prevFrameKey        : the key of the previous frame
        \t  prevFrameAlphaLoc   : location of the previous frame
        \t  prevMaskImportance  : balances the importance of the prior based on the remapped mask of the previous frame
        \t  prevMaskDilate      : amount of dilation to perform on previous frame's mask
        \t  prevMaskBlurSize    : size of the blurring kernel perfomed on previous frame's mask
        \t  prevMaskBlurSigma   : variance of the gaussian blurring perfomed on previous frame's mask
        \t  diffPatchImportance : balances the importance of the prior based on difference of patch to background
        \t  diffPatchMultiplier : multiplier that changes the scaling of the difference based cost
        \t  useOpticalFlow      : modify foreground prior by the mask of the previous frame
        \t  useDiffPatch        : modify bg prior by difference of fg to bg patch
           
        return: bgPrior, fgPrior"""
    
    ## get uniform prior for bg patch
    bgPrior = -np.log(np.ones(patchSize)/np.prod(patchSize))
    
    ## get prior for fg patch
    fgPrior = np.zeros(patchSize)
    xs = np.ndarray.flatten(np.arange(patchSize[1], dtype=float).reshape((patchSize[1], 1)).repeat(patchSize[0], axis=-1))
    ys = np.ndarray.flatten(np.arange(patchSize[0], dtype=float).reshape((1, patchSize[0])).repeat(patchSize[1], axis=0))
    data = np.vstack((xs.reshape((1, len(xs))), ys.reshape((1, len(ys)))))
    
    ## get covariance and means of prior on patch by using the bbox
    bbox = trackedSequence[DICT_BBOXES][frameKey].T
    segment1 = bbox[:, 0] - bbox[:, 1]
    segment2 = bbox[:, 1] - bbox[:, 2]
    sigmaX = np.linalg.norm(segment1)/3.7
    sigmaY = np.linalg.norm(segment2)/3.7
    
    rotRadians = trackedSequence[DICT_BBOX_ROTATIONS][frameKey]
    
    rotMat = np.array([[np.cos(rotRadians), -np.sin(rotRadians)], [np.sin(rotRadians), np.cos(rotRadians)]])
    
    means = np.reshape(trackedSequence[DICT_BBOX_CENTERS][frameKey], (2, 1)) - offset
    covs = np.dot(np.dot(rotMat.T, np.array([[sigmaX**2, 0.0], [0.0, sigmaY**2]])), rotMat)
    
    fgPrior = np.reshape(minusLogMultivariateNormal(data, means, covs, True), patchSize, order='F')
    
    ## change the fgPrior using optical flow stuff
    if useOpticalFlow and prevFrameKey != None :
        prevFrameName = trackedSequence[DICT_FRAMES_LOCATIONS][prevFrameKey].split(os.sep)[-1]
        nextFrameName = trackedSequence[DICT_FRAMES_LOCATIONS][frameKey].split(os.sep)[-1]
        
        if os.path.isfile(prevFrameAlphaLoc+prevFrameName) :
            alpha = np.array(Image.open(prevFrameAlphaLoc+prevFrameName))[:, :, -1]/255.0

            flow = cv2.calcOpticalFlowFarneback(cv2.cvtColor(np.array(Image.open(framesLoc+nextFrameName)), cv2.COLOR_RGB2GRAY), 
                                                cv2.cvtColor(np.array(Image.open(framesLoc+prevFrameName)), cv2.COLOR_RGB2GRAY), 
                                                0.5, 3, 15, 3, 5, 1.1, 0)
        
            ## remap alpha according to flow
            remappedFg = cv2.remap(alpha, flow[:, :, 0]+allXs, flow[:, :, 1]+allYs, cv2.INTER_LINEAR)
            ## get patch
            remappedFgPatch = remappedFg[offset[1]:offset[1]+patchSize[0], offset[0]:offset[0]+patchSize[1]]
            remappedFgPatch = cv2.GaussianBlur(cv2.morphologyEx(remappedFgPatch, cv2.MORPH_DILATE, 
                                                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (prevMaskDilate, prevMaskDilate))), 
                                               (prevMaskBlurSize, prevMaskBlurSize), prevMaskBlurSigma)

            fgPrior = (1.0-prevMaskImportance)*fgPrior + prevMaskImportance*(-np.log((remappedFgPatch+0.01)/np.sum(remappedFgPatch+0.01)))
    
    
    if useDiffPatch :
        ## change the background prior to give higher cost for pixels to be classified as background if the difference between bgPatch and fgPatch is high
        diffPatch = np.reshape(vectorisedMinusLogMultiNormal(fgPatch.reshape((np.prod(patchSize), 3)), 
                                                             bgPatch.reshape((np.prod(patchSize), 3)), 
                                                             np.eye(3)*diffPatchMultiplier, True), patchSize)
        bgPrior = (1.0-diffPatchImportance)*bgPrior + diffPatchImportance*diffPatch
        
    
    return bgPrior, fgPrior
    

def mergePatches(bgPatch, fgPatch, bgPrior, fgPrior, offset, patchSize, touchedBorders, scribble = None, useCenterSquare = True, useGradients = False) :
    """Computes pixel labels using graphcut given two same size patches
    
        \t  bgPatch         : background patch
        \t  fgPatch         : foreground patch
        \t  bgPrior         : background prior
        \t  fgPrior         : foreground prior
        \t  offset          : [x, y] position of patches in the coordinate system of the original images
        \t  patchSize       : num of [rows, cols] per patches
        \t  touchedBorders  : borders of the image touched by the enlarged bbox
        \t  useCenterSquare : forces square of pixels in the center of the patch to be classified as foreground
        \t  useGradients    : uses the gradient weighted pairwise cost
           
        return: reshapedLabels = labels for each pixel"""

    t = time.time()
    ## merge two overlapping patches

    h = patchSize[0]
    w = patchSize[1]
    
    patAPixs = np.empty(0, dtype=np.uint)
    patBPixs = np.empty(0, dtype=np.uint)
    
    ## force small square of size squarePadding*2 + 1 around center of patch to come from patch B (i.e. the car)
    if useCenterSquare :
        squarePadding = 6
        rows = np.ndarray.flatten(np.arange((h/2)-squarePadding, (h/2)+squarePadding+1).reshape((squarePadding*2+1, 1)).repeat(squarePadding*2+1, axis=-1))
        cols = np.ndarray.flatten(np.arange((w/2)-squarePadding, (w/2)+squarePadding+1).reshape((1, squarePadding*2+1)).repeat(squarePadding*2+1, axis=0))
        patBPixs = np.unique(np.concatenate((patBPixs, np.array(rows + cols*h, dtype=np.uint))))
    
    ## force one ring of pixels on the edge of the patch to come from patch A (i.e. the bg) (unless that column/row is intersected by the bbox)
#     if np.min((largeBBox)[0, :]) > 0.0 :
    if touchedBorders[0] :
#         print "adding left column to A"
        patAPixs = np.unique(np.concatenate((patAPixs, np.arange(0, h, dtype=np.int)[1:-1])))
    else :
#         print "adding left column to B"
        patBPixs = np.unique(np.concatenate((patBPixs, np.arange(0, h, dtype=np.uint)[1:-1])))
#     if np.min((largeBBox)[1, :]) > 0.0 :
    if touchedBorders[1] :
#         print "adding top row to A"
        patAPixs = np.unique(np.concatenate((patAPixs, np.arange(0, h*(w-1)+1, h, dtype=np.uint)[1:-1])))
    else :
#         print "adding top row to B"
        patBPixs = np.unique(np.concatenate((patBPixs, np.arange(0, h*(w-1)+1, h, dtype=np.uint)[1:-1])))
#     if np.max((largeBBox)[1, :]) < bgImage.shape[0] :
    if touchedBorders[2] :
#         print "adding bottom row to A"
        patAPixs = np.unique(np.concatenate((patAPixs, np.arange(0, h*(w-1)+1, h, dtype=np.uint)[1:-1]+h-1)))
    else :
#         print "adding bottom row to B"
        patBPixs = np.unique(np.concatenate((patBPixs, np.arange(0, h*(w-1)+1, h, dtype=np.uint)[1:-1]+h-1)))
#     if np.max((largeBBox)[0, :]) < bgImage.shape[1] :
    if touchedBorders[3] :
#         print "adding right column to A"
        patAPixs = np.unique(np.concatenate((patAPixs, np.arange(h*(w-1), h*w, dtype=np.uint)[1:-1])))
    else :
#         print "adding right column to B"
        patBPixs = np.unique(np.concatenate((patBPixs, np.arange(h*(w-1), h*w, dtype=np.uint)[1:-1])))
    
#     patBPixs = np.empty(0)

    ## deal with scribble if present
    if scribble != None :
        ## find indices of bg (blue) and fg (green) pixels in scribble
        bgPixs = np.argwhere(np.all((scribble[:, :, 0] == 0, scribble[:, :, 1] == 0, scribble[:, :, 2] == 255), axis=0))
        bgPixs[:, 0] -= offset[1]
        bgPixs[:, 1] -= offset[0]
        bgPixs = bgPixs[np.all(np.concatenate(([bgPixs[:, 0] >= 0], 
                                               [bgPixs[:, 1] >= 0], 
                                               [bgPixs[:, 0] < h], 
                                               [bgPixs[:, 1] < w])).T, axis=-1), :]
        fgPixs = np.argwhere(np.all((scribble[:, :, 0] == 0, scribble[:, :, 1] == 255, scribble[:, :, 2] == 0), axis=0))
        fgPixs[:, 0] -= offset[1]
        fgPixs[:, 1] -= offset[0]
        fgPixs = fgPixs[np.all(np.concatenate(([fgPixs[:, 0] >= 0], 
                                               [fgPixs[:, 1] >= 0], 
                                               [fgPixs[:, 0] < h], 
                                               [fgPixs[:, 1] < w])).T, axis=-1), :]
        
        

        ## for simplicity keep track of fixed pixels in a new patch-sized array
        fixedPixels = np.zeros(patchSize)
        ## get fixed pixels from other params first
        ## 1 == bg pixels (get 2d coords from 1d first)
        if len(patAPixs) > 0 :
            fixedPixels[np.array(np.mod(patAPixs, patchSize[0]), dtype=np.uint), np.array(patAPixs/patchSize[0], dtype=np.uint)] = 1
        ## 2 == fg pixels (get 2d coords from 1d first)
        if len(patAPixs) > 0 :
            fixedPixels[np.array(np.mod(patBPixs, patchSize[0]), dtype=np.uint), np.array(patBPixs/patchSize[0], dtype=np.uint)] = 2
        
        if len(bgPixs) > 0 :
            fixedPixels[bgPixs[:, 0], bgPixs[:, 1]] = 1
        if len(fgPixs) > 0 :
            fixedPixels[fgPixs[:, 0], fgPixs[:, 1]] = 2

        ## turn back to 1d indices
        patAPixs = np.argwhere(fixedPixels == 1)
        patAPixs = np.sort(patAPixs[:, 0] + patAPixs[:, 1]*patchSize[0])
        patBPixs = np.argwhere(fixedPixels == 2)
        patBPixs = np.sort(patBPixs[:, 0] + patBPixs[:, 1]*patchSize[0])
    
    patA = np.copy(bgPatch/255.0)
    patB = np.copy(fgPatch/255.0)
    
#     print "patch setup", time.time() - t
    t = time.time()
    
    if useGradients :
        sobelX = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]])

        labels, unaryCosts, pairCosts, graphModel = getGraphcutOnOverlap(patA, patB, patAPixs, patBPixs, 0.001, 
                                                           bgPrior.reshape(np.prod(patchSize), order='F'),
                                                           fgPrior.reshape(np.prod(patchSize), order='F'),
                                                           cv2.filter2D(bgPatch, cv2.CV_32F, sobelX),
                                                           cv2.filter2D(bgPatch, cv2.CV_32F, sobelX.T),
                                                           cv2.filter2D(fgPatch, cv2.CV_32F, sobelX),
                                                           cv2.filter2D(fgPatch, cv2.CV_32F, sobelX.T))
    else :
        labels, unaryCosts, pairCosts, graphModel = getGraphcutOnOverlap(patA, patB, patAPixs, patBPixs, 0.001, 
                                                           bgPrior.reshape(np.prod(patchSize), order='F'),
                                                           fgPrior.reshape(np.prod(patchSize), order='F'))
    
#     print "total solving", time.time() - t
    t = time.time()
        
    return labels


# In[5]:

def propagateLabels(distances, initPoints, numExtraPoints, verbose, sigma=0.06) :
    
    ########## initializing the labelled frames #########
    numClasses = len(initPoints)
    numFrames = distances.shape[0]
    labelledPoints = np.empty(0, dtype=np.int)
    numExamplesPerClass = []
    prevNumExamples = 0
    for i in xrange(numClasses) :
        for j in xrange(len(initPoints[i])) :
            extraPoints = numExtraPoints[i][j]
            labelledPoints = np.concatenate((labelledPoints, range(initPoints[i][j]-extraPoints/2, initPoints[i][j]+extraPoints/2+1)))
        labelledPoints = labelledPoints[np.all(np.vstack([[labelledPoints >= 0], [labelledPoints < numFrames]]), axis=0)]
        numExamplesPerClass.append(len(labelledPoints)-prevNumExamples)
        prevNumExamples = len(labelledPoints)
    numExamplesPerClass = np.array(numExamplesPerClass)
    
    ## class probabilities for labelled points
    fl = np.zeros((len(labelledPoints), numClasses))
    for i in xrange(0, numClasses) :
        fl[np.sum(numExamplesPerClass[:i]):np.sum(numExamplesPerClass[:i+1]), i] = 1

    ## order w to have labeled nodes at the top-left corner
    flatLabelled = np.ndarray.flatten(labelledPoints)
    
    if verbose :
        print "number of classes:", numClasses
        print "number of examples per class:", numExamplesPerClass
        print "number of extra points per class:", numExtraPoints
        print "all examples:", flatLabelled, "(", len(flatLabelled), ")"
    
    ######### do label propagation as zhu 2003 #########
    
    orderedDist = np.copy(distances)
    sortedFlatLabelled = flatLabelled[np.argsort(flatLabelled)]
    sortedFl = fl[np.argsort(flatLabelled), :]
    for i in xrange(0, len(sortedFlatLabelled)) :
        #shift sortedFlatLabelled[i]-th row up to i-th row and adapt remaining rows
        tmp = np.copy(orderedDist)
        orderedDist[i, :] = tmp[sortedFlatLabelled[i], :]
        orderedDist[i+1:, :] = np.vstack((tmp[i:sortedFlatLabelled[i], :], tmp[sortedFlatLabelled[i]+1:, :]))
        #shift sortedFlatLabelled[i]-th column left to i-th column and adapt remaining columns
        tmp = np.copy(orderedDist)
        orderedDist[:, i] = tmp[:, sortedFlatLabelled[i]]
        orderedDist[:, i+1:] = np.hstack((tmp[:, i:sortedFlatLabelled[i]], tmp[:, sortedFlatLabelled[i]+1:]))

    ## compute weights
    w, cumW = vtu.getProbabilities(orderedDist, sigma, None, False)

    l = len(sortedFlatLabelled)
    n = orderedDist.shape[0]
    ## compute graph laplacian
    L = np.diag(np.sum(w, axis=0)) - w

    ## propagate labels
    fu = np.dot(np.dot(-np.linalg.inv(L[l:, l:]), L[l:, 0:l]), sortedFl)

    ## use class mass normalization to normalize label probabilities
    q = np.sum(sortedFl)+1
    fu_CMN = fu*(np.ones(fu.shape)*(q/np.sum(fu)))
    
    
    ########## get label probabilities and plot ##########
    
    ## add labeled points to propagated labels (as labelProbs)
    labelProbs = np.copy(np.array(fu))
    for frame, i in zip(sortedFlatLabelled, np.arange(len(sortedFlatLabelled))) :
        labelProbs = np.vstack((labelProbs[0:frame, :], sortedFl[i, :], labelProbs[frame:, :]))

#     ## plot
#     if verbose :
#         print "final computed labels shape:", labelProbs.shape
#         fig1 = figure()
#         clrs = np.arange(0.0, 1.0+1.0/(len(initPoints)-1), 1.0/(len(initPoints)-1)).astype(np.string_) #['r', 'g', 'b', 'm', 'c', 'y', 'k', 'w']
#         stackplot(np.arange(len(labelProbs)), np.row_stack(tuple([i for i in labelProbs.T])), colors=clrs)
        
    return labelProbs


# In[6]:

DRAW_FIRST_FRAME = 'first_frame'
DRAW_LAST_FRAME = 'last_frame'
DRAW_COLOR = 'color'
LIST_SECTION_SIZE = 60
SLIDER_INDICATOR_WIDTH=4

class SemanticsSlider(QtGui.QSlider) :
    def __init__(self, orientation=QtCore.Qt.Horizontal, parent=None) :
        super(SemanticsSlider, self).__init__(orientation, parent)
        style = "QSlider::handle:horizontal { background: #cccccc; width: 0px; border-radius: 0px; } "
        style += "QSlider::groove:horizontal { background: #dddddd; } "
        self.setStyleSheet(style)
        
        self.semanticsToDraw = []
        self.numOfFrames = 1
        self.selectedSemantics = 0
        
    def setSelectedSemantics(self, selectedSemantics) :
        self.selectedSemantics = selectedSemantics
        
    def setSemanticsToDraw(self, semanticsToDraw, numOfFrames) :
        self.semanticsToDraw = semanticsToDraw
        self.numOfFrames = float(numOfFrames)
        
        desiredHeight = np.max((42, len(self.semanticsToDraw)*7))
        self.setFixedHeight(desiredHeight)
        
        self.resize(self.width(), self.height())
        self.update()
        
    def mousePressEvent(self, event) :
        if event.button() == QtCore.Qt.LeftButton :
            self.setValue(event.pos().x()*(float(self.maximum())/self.width()))
        
    def paintEvent(self, event) :
        super(SemanticsSlider, self).paintEvent(event)
        
        painter = QtGui.QPainter(self)
        
        ## draw semantics
        
        yCoord = 0.0
        for i in xrange(len(self.semanticsToDraw)) :
            col = self.semanticsToDraw[i][DRAW_COLOR]

            painter.setBrush(QtGui.QBrush(QtGui.QColor.fromRgb(col[0], col[1], col[2], 255)))
            startX =  self.semanticsToDraw[i][DRAW_FIRST_FRAME]/self.numOfFrames*self.width()
            endX =  self.semanticsToDraw[i][DRAW_LAST_FRAME]/self.numOfFrames*self.width()

            if self.selectedSemantics == i :
                painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(255 - col[0], 255 - col[1], 255 - col[2], 127), 1, 
                                              QtCore.Qt.DashLine, QtCore.Qt.SquareCap, QtCore.Qt.MiterJoin))
                painter.drawRect(startX, yCoord+0.5, endX-startX, 5)

            else :
                painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(255 - col[0], 255 - col[1], 255 - col[2], 63), 1, 
                                              QtCore.Qt.SolidLine, QtCore.Qt.SquareCap, QtCore.Qt.MiterJoin))
                painter.drawRect(startX, yCoord+0.5, endX-startX, 5)


            yCoord += 7        
        
        ## draw slider
        ## mapping slider interval to its size
        A = 0.0
        B = float(self.maximum())
        a = SLIDER_INDICATOR_WIDTH/2.0
        b = float(self.width())-SLIDER_INDICATOR_WIDTH/2.0
        if (B-A) != 0.0 :
            ## (val - A)*(b-a)/(B-A) + a
            sliderXCoord = (float(self.value()) - A)*(b-a)/(B-A) + a
        else :
            sliderXCoord = a
        painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(0, 0, 0, 255), SLIDER_INDICATOR_WIDTH,
                                          QtCore.Qt.SolidLine, QtCore.Qt.FlatCap, QtCore.Qt.MiterJoin))
        painter.drawLine(sliderXCoord, 0, sliderXCoord, self.height())
        
        painter.end()


# In[7]:

class ListDelegate(QtGui.QItemDelegate):
    
    def __init__(self, parent=None) :
        super(ListDelegate, self).__init__(parent)
        
        self.setBackgroundColor(QtGui.QColor.fromRgb(245, 245, 245))
        self.iconImage = None

    def setBackgroundColor(self, bgColor) :
        self.bgColor = bgColor
    
    def setIconImage(self, iconImage) :
        self.iconImage = np.ascontiguousarray(np.copy(iconImage))
    
    def drawDisplay(self, painter, option, rect, text):
        painter.save()
        
        colorRect = QtCore.QRect(rect.left()+rect.height(), rect.top(), rect.width()-rect.height(), rect.height())
        selectionRect = rect
        iconRect = QtCore.QRect(rect.left(), rect.top(), rect.height(), rect.height())
        
        if np.any(self.iconImage != None) :

            # draw colorRect
            painter.setBrush(QtGui.QBrush(self.bgColor))
            painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(128, 128, 128, 255), 1, 
                                              QtCore.Qt.SolidLine, QtCore.Qt.SquareCap, QtCore.Qt.MiterJoin))
            painter.drawRect(colorRect)

            ## draw iconRect
            qim = QtGui.QImage(self.iconImage.data, self.iconImage.shape[1], self.iconImage.shape[0], 
                                                           self.iconImage.strides[0], QtGui.QImage.Format_RGB888)
            painter.drawImage(iconRect, qim.scaled(iconRect.size()))

            ## draw selectionRect
            if option.state & QtGui.QStyle.State_Selected:
                painter.setBrush(QtGui.QBrush(QtGui.QColor.fromRgb(0, 0, 0, 0)))
                painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(255, 64, 64, 255), 5, 
                                                  QtCore.Qt.SolidLine, QtCore.Qt.SquareCap, QtCore.Qt.MiterJoin))
                painter.drawRect(selectionRect)

            # set text color
            painter.setPen(QtGui.QPen(QtCore.Qt.black))
            if option.state & QtGui.QStyle.State_Selected:
                painter.setFont(QtGui.QFont("Helvetica [Cronyx]", 11, QtGui.QFont.Bold))
            else :
                painter.setFont(QtGui.QFont("Helvetica [Cronyx]", 11))

            painter.drawText(colorRect, QtCore.Qt.AlignVCenter | QtCore.Qt.AlignCenter, text)
        else :
            painter.drawText(QtCore.QRect(rect.left(), rect.top(), rect.width(), rect.height()), 
                             QtCore.Qt.AlignVCenter | QtCore.Qt.AlignCenter, text)

        painter.restore()


# In[8]:

class ImageLabel(QtGui.QLabel) :
    
    def __init__(self, text, parent=None):
        super(ImageLabel, self).__init__(text, parent)
        
        self.setMouseTracking(True)
        
        self.image = None
        self.overlay = None
        self.segmentedImage = None
        self.scribbleImage = None
        self.imageOpacity = 0.5
        self.scribbleOpacity = 0.5
        self.isModeBBox = True
        
    def setImage(self, image) :
        if np.any(image != None) :
            self.image = image.copy()
        else :
            self.image = None
        self.update()

    def setOverlay(self, overlay) :
        if np.any(overlay != None) :
            self.overlay = overlay.copy()
        else :
            self.overlay = None
        self.update()
        
    def setSegmentedImage(self, segmentedImage) : 
        if np.any(segmentedImage != None) :
            self.segmentedImage = segmentedImage.copy()
        else :
            self.segmentedImage = None
        self.update()
        
    def setScribbleImage(self, scribbleImage) :
        if np.any(scribbleImage != None) :
            self.scribbleImage = scribbleImage.copy()
        else :
            self.scribbleImage = None
        self.update()
        
    def setModeBBox(self, isModeBBox) : 
        self.isModeBBox = isModeBBox
        self.update()
        
    def setImageOpacity(self, imageOpacity) : 
        self.imageOpacity = imageOpacity
        self.update()
        
    def setScribbleOpacity(self, scribbleOpacity) : 
        self.scribbleOpacity = scribbleOpacity
        self.update()
        
    def paintEvent(self, event):
        super(ImageLabel, self).paintEvent(event)
        painter = QtGui.QPainter(self)
        if np.any(self.image != None) :
            painter.drawImage(QtCore.QPoint(0, 0), self.image)
        
        if np.any(self.segmentedImage != None) :
            ## draw rect
            painter.setBrush(QtGui.QBrush(QtGui.QColor.fromRgb(0, 32, 32, 255-int(self.imageOpacity*255))))
            painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(0, 0, 0, 0)))
            painter.drawRect(QtCore.QRect(0, 0, self.image.width(), self.image.height()))
            painter.setBrush(QtGui.QBrush(QtGui.QColor.fromRgb(0, 0, 0, 0)))
            
            painter.drawImage(QtCore.QPoint(0, 0), self.segmentedImage)
        
        if np.any(self.scribbleImage != None) :
            painter.setOpacity(self.scribbleOpacity)
            painter.setCompositionMode(QtGui.QPainter.CompositionMode_Multiply)
            painter.drawImage(QtCore.QPoint(0, 0), self.scribbleImage)
            painter.setOpacity(1.0)
            painter.setCompositionMode(QtGui.QPainter.CompositionMode_SourceOver)
            
        if np.any(self.overlay != None) :
            painter.drawImage(QtCore.QPoint(0, 0), self.overlay)
            
#         if self.isModeBBox :
#             modeColor = QtGui.QColor.fromRgb(255, 0, 0)
#         else :
#             modeColor = QtGui.QColor.fromRgb(0, 0, 255)
            
#         painter.setPen(QtGui.QPen(modeColor, 10, QtCore.Qt.SolidLine, QtCore.Qt.SquareCap, QtCore.Qt.MiterJoin))
#         painter.drawRect(QtCore.QRect(0, 0, self.width(), self.height()))
            
        painter.end()
        
    def setPixmap(self, pixmap) :
        if pixmap.width() > self.width() :
            super(ImageLabel, self).setPixmap(pixmap.scaledToWidth(self.width()))
        else :
            super(ImageLabel, self).setPixmap(pixmap)
        
    def resizeEvent(self, event) :
        if np.any(self.pixmap() != None) :
            if self.pixmap().width() > self.width() :
                self.setPixmap(self.pixmap().scaledToWidth(self.width()))


# In[9]:

class IconLabel(QtGui.QLabel) :
    
    def __init__(self, text, parent=None):
        super(IconLabel, self).__init__(text, parent)
        
        self.setMouseTracking(True)
        
        self.image = None
        self.topLeft = np.array([0, 0])
        self.zoomLevel = 1.0
        
    def setImage(self, image) : 
        self.image = image.copy()
        self.topLeft = np.array([image.width()/2, image.height()/2])
        self.update()
        
    def changeZoomLevel(self, delta):
        if (self.width()*(self.zoomLevel+delta) < self.image.width()-self.topLeft[0] and
            self.height()*(self.zoomLevel+delta) < self.image.height()-self.topLeft[1] and
            self.width()*(self.zoomLevel+delta) > 10.0 and self.height()*(self.zoomLevel+delta) > 10.0) :
    
            self.zoomLevel += delta
            self.update()
        
    def changeTopLeft(self, delta):
        if (self.topLeft[0]+delta[0] > 0 and
            self.topLeft[0]+delta[0] < self.image.width()-self.width()*self.zoomLevel) :
            
            self.topLeft[0] += delta[0]
            self.update()
            
        if (self.topLeft[1]+delta[1] > 0 and
            self.topLeft[1]+delta[1] < self.image.height()-self.height()*self.zoomLevel) :
            
            self.topLeft[1] += delta[1]
            self.update()
        
    def getPixelColor(self, x, y):
        return QtGui.QColor(self.image.copy(self.topLeft[0], self.topLeft[1],self.width()*self.zoomLevel, 
                                            self.height()*self.zoomLevel).scaled(self.size()).pixel(x, y))
        
    def paintEvent(self, event):
        super(IconLabel, self).paintEvent(event)
        painter = QtGui.QPainter(self)
        
        if np.any(self.image != None) :
            painter.drawImage(QtCore.QPoint(0, 0), self.image.copy(self.topLeft[0], self.topLeft[1], 
                                                                   self.width()*self.zoomLevel, 
                                                                   self.height()*self.zoomLevel).scaled(self.size()))
        
        painter.end()
                
class NewSemanticsDialog(QtGui.QDialog):
    def __init__(self, parent=None, title="", image=None):
        super(NewSemanticsDialog, self).__init__(parent)
        
        self.prevPoint = None
        self.movingIcon = False
        self.semanticsColor = np.array([255, 255, 255])
        
        self.createGUI(image)
        
        self.setWindowTitle(title)
        
    def accept(self):        
        self.done(1)
    
    def reject(self):        
        self.done(0)
        
    def mousePressed(self, event):
        if event.button() == QtCore.Qt.LeftButton :
            self.movingIcon = True
            self.prevPoint = event.pos()
        elif event.button() == QtCore.Qt.RightButton :
            newColor = self.iconLabel.getPixelColor(event.pos().x(), event.pos().y())
            
            self.semanticsColor = np.array([newColor.red(), newColor.green(), newColor.blue()])
            self.colorButton.setStyleSheet("QPushButton {border: 1px solid black; background-color: rgb("+
                                           np.string_(self.semanticsColor[0])+", "+np.string_(self.semanticsColor[1])
                                           +", "+np.string_(self.semanticsColor[2])+");}")
                
        sys.stdout.flush()
                
    def mouseMoved(self, event):
        if self.movingIcon and np.any(self.prevPoint != None) :
            tmp = self.prevPoint-event.pos()
            self.iconLabel.changeTopLeft(np.array([tmp.x(), tmp.y()]))
            self.prevPoint = event.pos()
            
    def mouseReleased(self, event):
        if self.movingIcon and np.any(self.prevPoint != None) :
            tmp = self.prevPoint-event.pos()
            self.iconLabel.changeTopLeft(np.array([tmp.x(), tmp.y()]))
            self.prevPoint = event.pos()
            self.movingIcon = False        
        
    def wheelEvent(self, e) :
        if e.delta() < 0 :
            self.iconLabel.changeZoomLevel(-0.05)
        else :
            self.iconLabel.changeZoomLevel(0.05)
        
    def eventFilter(self, obj, event) :
        if obj == self.iconLabel and event.type() == QtCore.QEvent.Type.MouseMove :
            self.mouseMoved(event)
            return True
        elif obj == self.iconLabel and event.type() == QtCore.QEvent.Type.MouseButtonPress :
            self.mousePressed(event)
            return True
        elif obj == self.iconLabel and event.type() == QtCore.QEvent.Type.MouseButtonRelease :
            self.mouseReleased(event)
            return True
        return QtGui.QWidget.eventFilter(self, obj, event)
    
    def setSemanticsColor(self) :
        newColor = QtGui.QColorDialog.getColor(QtGui.QColor(self.semanticsColor[0],
                                                            self.semanticsColor[1],
                                                            self.semanticsColor[2]), self, "Choose Sequence Color")
        if newColor.isValid() :
            self.semanticsColor = np.array([newColor.red(), newColor.green(), newColor.blue()])
            self.colorButton.setStyleSheet("QPushButton {border: 1px solid black; background-color: rgb("+
                                           np.string_(self.semanticsColor[0])+", "+np.string_(self.semanticsColor[1])
                                           +", "+np.string_(self.semanticsColor[2])+");}")
    
    def createGUI(self, image):
        
        self.nameEdit = QtGui.QLineEdit()
        self.nameEdit.setText("sequence_name")
        
        self.iconLabel = IconLabel("Icon")
        self.iconLabel.setFixedSize(100, 100)
        self.iconLabel.setImage(image)
        self.iconLabel.installEventFilter(self)
        
        self.colorButton = QtGui.QPushButton("Choose Color")
        self.colorButton.setStyleSheet("QPushButton {border: 1px solid black; background-color: rgb(255, 255, 255);}")
        
        self.buttonBox = QtGui.QDialogButtonBox(QtGui.QDialogButtonBox.Ok | QtGui.QDialogButtonBox.Cancel);
         
        ## SIGNALS ##
        
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.colorButton.clicked.connect(self.setSemanticsColor)
        
        ## LAYOUTS ##
        
        mainLayout = QtGui.QGridLayout()
        
        mainLayout.addWidget(QtGui.QLabel("Name:"), 0, 0, 1, 1, QtCore.Qt.AlignLeft)
        mainLayout.addWidget(self.nameEdit, 0, 1, 1, 1, QtCore.Qt.AlignRight)
        mainLayout.addWidget(QtGui.QLabel("Icon:"), 1, 0, 1, 1, QtCore.Qt.AlignLeft)
        mainLayout.addWidget(self.iconLabel, 1, 1, 1, 1, QtCore.Qt.AlignRight)
        mainLayout.addWidget(self.colorButton, 2, 0, 1, 2, QtCore.Qt.AlignCenter)
        mainLayout.addWidget(self.buttonBox, 3, 0, 1, 2, QtCore.Qt.AlignCenter)
        
        self.setLayout(mainLayout)

def createNewSemantics(parent=None, title="Dialog", image= None) :
    newSemanticsDialog = NewSemanticsDialog(parent, title, image)
    exitCode = newSemanticsDialog.exec_()
    
    return (newSemanticsDialog.nameEdit.text(),
            newSemanticsDialog.iconLabel.topLeft[::-1].astype(np.int),
            int(newSemanticsDialog.iconLabel.width()*newSemanticsDialog.iconLabel.zoomLevel),
            newSemanticsDialog.semanticsColor.astype(np.int), exitCode)


# In[10]:

def makeStackPlot(values, height) :
    stackPlot = np.zeros((height, values.shape[0]))
    if values.shape[1]  == 1 :
        clrs = [0.0]
    else :
#         clrs = np.arange(0.0, 1.0+1.0/(values.shape[1]-1), 1.0/(values.shape[1]-1))
        clrs = np.arange(0.0, 1.0 + 1.0/15.0, 1.0/15.0)
        clrs = clrs[:values.shape[1]]
    for idx, ranges in enumerate(np.cumsum(np.round(values*height), axis=1).astype(int)) :
        for i, j, col in zip(np.concatenate(([0], ranges[:-1])), ranges, np.arange(len(ranges), dtype=int)) :
            stackPlot[i:j, idx] = clrs[col]
    return stackPlot

def valsToImg(vals) :
    img = np.ones((vals.shape[0], vals.shape[1], 4), np.uint8)*255
    img[:, :, :-1] = (vals*255).astype(np.uint8).reshape((vals.shape[0], vals.shape[1], 1))
    return img

class SemanticsLabel(QtGui.QLabel) :
    
    def __init__(self, text, parent=None):
        super(SemanticsLabel, self).__init__(text, parent)
        
        self.image = None
        self.labelledFrames = []
        self.currentFrame = 0
        self.frameIdxDelta = 0
        self.maxFrames = 0
        self.scaleFactor = 1.0
        
        self.PLOT_HEIGHT = 50
        self.GRAPH_EXTRA = 10
        self.MIN_GRAPH_WIDTH = 250.0
        self.MAX_GRAPH_WIDTH = 1000.0
        self.setFixedHeight(self.PLOT_HEIGHT+self.GRAPH_EXTRA+2)
        
    def setSemantics(self, semanticSequence) :
        if np.all(semanticSequence != None) and DICT_FRAMES_LOCATIONS in semanticSequence.keys() and len(semanticSequence[DICT_FRAMES_LOCATIONS].keys()) > 0 :
            self.frameIdxDelta = np.min(semanticSequence[DICT_FRAMES_LOCATIONS].keys())
            self.maxFrames = len(semanticSequence[DICT_FRAMES_LOCATIONS].keys())
            if DICT_FRAME_SEMANTICS in semanticSequence.keys() :
#                 self.semanticsPlot = np.ascontiguousarray(valsToImg(makeStackPlot(semanticSequence[DICT_FRAME_SEMANTICS], self.PLOT_HEIGHT))[:, :, [2, 1, 0, 3]])
                self.semanticsPlot = np.ascontiguousarray(mpl.cm.Set1(makeStackPlot(semanticSequence[DICT_FRAME_SEMANTICS], self.PLOT_HEIGHT), bytes=True)[:, :, [2, 1, 0, 3]])
                self.image = QtGui.QImage(self.semanticsPlot.data, self.semanticsPlot.shape[1], self.semanticsPlot.shape[0],
                                          self.semanticsPlot.strides[0], QtGui.QImage.Format_ARGB32);
                
                if self.image.width() < self.MIN_GRAPH_WIDTH :
                    self.scaleFactor = self.MIN_GRAPH_WIDTH/(self.image.width()-1)
                    self.image = self.image.scaled(self.MIN_GRAPH_WIDTH, self.image.height())
                    print "scale1", self.scaleFactor
                elif self.image.width() > self.MAX_GRAPH_WIDTH :
                    self.scaleFactor = self.MAX_GRAPH_WIDTH/(self.image.width()-1)
                    self.image = self.image.scaled(self.MAX_GRAPH_WIDTH, self.image.height())
                    print "scale2", self.scaleFactor
                else :
                    self.scaleFactor = 1.0
                    print "scale3", self.scaleFactor
                    
                if DICT_NUM_SEMANTICS in semanticSequence.keys() :
                    self.labelledFrames = semanticSequence[DICT_LABELLED_FRAMES][:semanticSequence[DICT_NUM_SEMANTICS]]
                else :
                    self.labelledFrames = []
            else :
                self.image = None
                self.labelledFrames = []
        else :
            self.image = None
            self.labelledFrames = []

        self.update()
    
    def setCurrentFrame(self, currentFrame) :
        self.currentFrame = currentFrame-self.frameIdxDelta
        self.update()
        
    def paintEvent(self, event):
        super(SemanticsLabel, self).paintEvent(event)
        painter = QtGui.QPainter(self)
        
        if np.any(self.image != None) :
            xLoc = (self.width()-self.image.width())/2
            painter.drawImage(QtCore.QPoint(xLoc, self.GRAPH_EXTRA), self.image)
            
            ## draw current frame indicator
            if self.currentFrame >= 0 and self.currentFrame < self.maxFrames :
#                 painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(31, 31, 0, 255), 3,
#                                           QtCore.Qt.SolidLine, QtCore.Qt.FlatCap, QtCore.Qt.MiterJoin))
#                 painter.drawLine(xLoc+self.currentFrame*self.scaleFactor, 0, xLoc+self.currentFrame*self.scaleFactor, self.GRAPH_EXTRA-2)
                
#                 painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(255, 255, 0, 255), 1,
#                                           QtCore.Qt.SolidLine, QtCore.Qt.FlatCap, QtCore.Qt.MiterJoin))
#                 painter.drawLine(xLoc+self.currentFrame*self.scaleFactor, 1, xLoc+self.currentFrame*self.scaleFactor, self.GRAPH_EXTRA-2)
                
                
                ## draw triangle at the end
                trianglePoly = QtGui.QPolygonF()
                trianglePoly.append(QtCore.QPointF(xLoc+self.currentFrame*self.scaleFactor-1.5, 0))
                trianglePoly.append(QtCore.QPointF(xLoc+self.currentFrame*self.scaleFactor+1.5, 0))
                trianglePoly.append(QtCore.QPointF(xLoc+self.currentFrame*self.scaleFactor+1.5, self.GRAPH_EXTRA-2)) ##
                trianglePoly.append(QtCore.QPointF(xLoc+self.currentFrame*self.scaleFactor, self.GRAPH_EXTRA)) ##
                trianglePoly.append(QtCore.QPointF(xLoc+self.currentFrame*self.scaleFactor-1.5, self.GRAPH_EXTRA-2)) ##
                
                
                painter.setRenderHints(QtGui.QPainter.Antialiasing)
                painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(31, 31, 0, 255), 1,
                                          QtCore.Qt.SolidLine, QtCore.Qt.FlatCap, QtCore.Qt.MiterJoin))
                painter.setBrush(QtGui.QBrush(QtGui.QColor.fromRgb(31, 31, 0, 255)))
                painter.drawPolygon(trianglePoly)
                painter.setBrush(QtCore.Qt.NoBrush)
                
                
                painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(255, 255, 0, 255), 1,
                                          QtCore.Qt.SolidLine, QtCore.Qt.FlatCap, QtCore.Qt.MiterJoin))
                painter.drawLine(xLoc+self.currentFrame*self.scaleFactor, 1, xLoc+self.currentFrame*self.scaleFactor, self.GRAPH_EXTRA-2)
                
            
            ## draw labelled frames
            painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(0, 0, 0, 128), 2,
                                      QtCore.Qt.SolidLine, QtCore.Qt.FlatCap, QtCore.Qt.MiterJoin))
            for c in self.labelledFrames :
                for f in c :
                    painter.drawLine(xLoc+f*self.scaleFactor, 3, xLoc+f*self.scaleFactor, self.PLOT_HEIGHT+self.GRAPH_EXTRA)
                    
            painter.setRenderHints(QtGui.QPainter.Antialiasing, on=False)
            
            ## draw a line at the bottom
            painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(0, 255, 0, 255), 2,
                                      QtCore.Qt.SolidLine, QtCore.Qt.FlatCap, QtCore.Qt.MiterJoin))
            painter.drawLine(xLoc, self.PLOT_HEIGHT+self.GRAPH_EXTRA, xLoc+self.image.width(), self.PLOT_HEIGHT+self.GRAPH_EXTRA)
        
        painter.end()


# In[ ]:

class AddFramesToSequenceDialog(QtGui.QDialog):
    def __init__(self, parent=None, frameLocs=[], bbox=[], title=""):
        super(AddFramesToSequenceDialog, self).__init__(parent)
        
        self.MAX_FRAME_WIDTH = 250.0
        
        self.frameLocs = frameLocs
        self.numOfFrames = len(self.frameLocs)
        self.scale = 1.0
        self.bboxPoly = QtGui.QPolygonF()
        self.frameIdx = 0
        self.startFrame = self.frameIdx
        self.endFrame = self.numOfFrames-1
        
        if self.numOfFrames > 0 :
            self.scale = np.min([1.0, self.MAX_FRAME_WIDTH/np.array(Image.open(self.frameLocs[0])).shape[1]])
            
            for p in bbox :
                self.bboxPoly.append(p*self.scale)
        
        self.createGUI()
        self.showFrame(self.frameIdx)
        self.updateInfoLabel()
        
        self.setWindowTitle(title)
        
    def accept(self):        
        self.done(1)
    
    def reject(self):        
        self.done(0)
        
    def eventFilter(self, obj, event) :
        if obj == self.frameLabel and event.type() == QtCore.QEvent.Type.Paint :
            self.paintFrameLabel(obj, event)
            return True
        return QtGui.QWidget.eventFilter(self, obj, event)
    
    def keyPressEvent(self, e) :
        if e.key() == QtCore.Qt.Key_S :
            if self.frameIdx <= self.endFrame :
                self.startFrame = self.frameIdx
                self.updateInfoLabel()
        elif e.key() == QtCore.Qt.Key_E :
            if self.frameIdx >= self.startFrame :
                self.endFrame = self.frameIdx
                self.updateInfoLabel()
            
    def updateInfoLabel(self) :
        self.infoLabel.setText("<p align='center'>Selected interval <b>[{0}, {1}]</b><br>Use <b>S</b> to set start frame<br>and <b>E</b> for end frame</p>".format(int(self.startFrame), int(self.endFrame)))
    
    def paintFrameLabel(self, frameLabel, event) :
        super(QtGui.QLabel, frameLabel).paintEvent(event)
        frameLabel.paintEvent(event)
        
        if self.numOfFrames > 0 :
            painter = QtGui.QPainter(frameLabel)
            painter.setRenderHints(QtGui.QPainter.Antialiasing)
            
            if not self.bboxPoly.isEmpty() :
                painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(0, 0, 0), 3, 
                                          QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
                painter.drawPolygon(self.bboxPoly)

                painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(255, 255, 255), 1, 
                                          QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
                painter.drawPolygon(self.bboxPoly)
            
            painter.end()
            
    def showFrame(self, idx) :
        if idx >= 0 and idx < self.numOfFrames :
            self.frameIdx = idx
            self.image = np.ascontiguousarray(np.array(Image.open(self.frameLocs[self.frameIdx]))[:, :, :3])
            img = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], self.image.strides[0], QtGui.QImage.Format_RGB888)
            
            self.frameLabel.setPixmap(QtGui.QPixmap.fromImage(img.scaledToWidth(self.MAX_FRAME_WIDTH)))
            self.update()
            
    def createGUI(self):
        
        self.frameLabel = QtGui.QLabel("Frame")
        self.frameLabel.setFixedWidth(self.MAX_FRAME_WIDTH)
        self.frameLabel.installEventFilter(self)
        
        self.infoLabel = QtGui.QLabel("Info")
        
        self.setBBoxBox = QtGui.QCheckBox("Set Visualized BBox for Frames")
        self.setBBoxBox.setChecked(not self.bboxPoly.isEmpty())
        self.setBBoxBox.setEnabled(not self.bboxPoly.isEmpty())
        
        self.frameIdxSlider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.frameIdxSlider.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Minimum)
        self.frameIdxSlider.setMinimum(0)
        self.frameIdxSlider.setMaximum(np.max([0, self.numOfFrames-1]))
        
        self.frameIdxSpinBox = QtGui.QSpinBox()
        self.frameIdxSpinBox.setRange(0, np.max([0, self.numOfFrames-1]))
        self.frameIdxSpinBox.setSingleStep(1)
        
        self.buttonBox = QtGui.QDialogButtonBox(QtGui.QDialogButtonBox.Ok | QtGui.QDialogButtonBox.Cancel);
         
        ## SIGNALS ##
        
        self.frameIdxSlider.valueChanged[int].connect(self.frameIdxSpinBox.setValue)
        self.frameIdxSpinBox.valueChanged[int].connect(self.frameIdxSlider.setValue)
        self.frameIdxSpinBox.valueChanged[int].connect(self.showFrame)
        
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        
        ## LAYOUTS ##
        
        sliderLayout = QtGui.QHBoxLayout()
        sliderLayout.addWidget(self.frameIdxSlider)
        sliderLayout.addWidget(self.frameIdxSpinBox)
        
        boxLayout = QtGui.QHBoxLayout()
        boxLayout.addStretch()
        boxLayout.addWidget(self.buttonBox)
        boxLayout.addStretch()
        
        mainLayout = QtGui.QVBoxLayout()
        
        mainLayout.addWidget(self.frameLabel, QtCore.Qt.AlignCenter)
        mainLayout.addWidget(self.infoLabel, QtCore.Qt.AlignCenter)
        mainLayout.addWidget(self.setBBoxBox, QtCore.Qt.AlignCenter)
        mainLayout.addLayout(sliderLayout, QtCore.Qt.AlignCenter)
        mainLayout.addLayout(boxLayout, QtCore.Qt.AlignCenter)
        
        self.setLayout(mainLayout)


# In[ ]:

class LoadingDialog(QtGui.QDialog):
    doCancelSignal = QtCore.Signal()
    
    def __init__(self, parent=None, title=""):
        super(LoadingDialog, self).__init__(parent, QtCore.Qt.CustomizeWindowHint|QtCore.Qt.WindowTitleHint)
        
        self.createGUI()
#         self.doCancel = False
        
        self.setFixedWidth(300)
        
        self.setWindowTitle(title)
        
    def setOperationText(self, operationText) :
        self.operationTextLabel.setText(operationText)
    
    def setOperationProgress(self, operationProgress) :
        self.progressBar.setValue(operationProgress)
        
    def doneLoading(self) :
        self.done(0)
        
    def cancel(self) :
#         self.doCancel = True
        self.doCancelSignal.emit()
        self.done(1)
    
    def createGUI(self):
        
        self.operationTextLabel = QtGui.QLabel("Operation Text")
        self.operationTextLabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignHCenter)
        self.operationTextLabel.setWordWrap(True)
        self.operationTextLabel.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.MinimumExpanding)
        
        self.progressBar = QtGui.QProgressBar()
        self.progressBar.setRange(0, 100)
        self.progressBar.setValue(0)
        
        self.cancelButton = QtGui.QPushButton("Cancel")
        self.cancelButton.setSizePolicy(QtGui.QSizePolicy.Maximum, QtGui.QSizePolicy.Maximum)
         
        ## SIGNALS ##
        
        self.cancelButton.clicked.connect(self.cancel)
        
        ## LAYOUTS ##
        
        buttonLayout = QtGui.QHBoxLayout()
        buttonLayout.addStretch()
        buttonLayout.addWidget(self.cancelButton, QtCore.Qt.AlignCenter)
        buttonLayout.addStretch()
        
        mainLayout = QtGui.QVBoxLayout()
        
        mainLayout.addWidget(self.operationTextLabel, QtCore.Qt.AlignCenter)
        mainLayout.addWidget(self.progressBar, QtCore.Qt.AlignCenter)
        mainLayout.addLayout(buttonLayout, QtCore.Qt.AlignCenter)
        
        self.setLayout(mainLayout)

        
class LongOperationThread(QtCore.QThread):
    updateOperationProgressSignal = QtCore.Signal(int)
    updateOperationTextSignal = QtCore.Signal(str)
    abortOperationSignal = QtCore.Signal()
    doneOperationSignal = QtCore.Signal()
    
    def __init__(self, parent = None):
        super(LongOperationThread, self).__init__(parent)
        
        self.longOperation = None
        self.args = None
        
    def doQuit(self) :
        self.abortOperationSignal.emit()
        
        
    def doRun(self, longOperation, args) :
        self.longOperation = longOperation
        self.args = args
        
        if not self.isRunning() and np.all(self.longOperation != None) and np.all(self.args != None) :
            self.abortOperationSignal.connect(self.longOperation.doAbort)
            self.longOperation.updateOperationProgressSignal.connect(self.updateOperationProgressSignal)
            self.longOperation.updateOperationTextSignal.connect(self.updateOperationTextSignal)
            self.start()
            
    def run(self):
        if np.all(self.longOperation != None) and np.all(self.longOperation != None) :
            print "starting the operation"; sys.stdout.flush()
            self.longOperation.run(*self.args)
            print "ending the operation"; sys.stdout.flush()
            self.doneOperationSignal.emit()
        return


# In[ ]:

class LongOperationClass(QtCore.QObject) :
    updateOperationProgressSignal = QtCore.Signal(int)
    updateOperationTextSignal = QtCore.Signal(str)
    
    def __init__(self, parent = None):
        super(LongOperationClass, self).__init__(parent)
        
        self.abortRequested = False
    
    def doAbort(self) :
        self.abortRequested = True
        
class ComputeMedian(LongOperationClass) :
    
    def __init__(self, parent = None):
        super(ComputeMedian, self).__init__(parent)
    
    def run(self, framesLocation) :
        frameLocs = np.sort(glob.glob(framesLocation + "/frame-*.png"))
        numOfFrames = len(frameLocs)
        if numOfFrames > 0 :
#             print numOfFrames
            frameSize = np.array(Image.open(frameLocs[0])).shape[0:2]
            medianImage = np.zeros((frameSize[0], frameSize[1], 3), dtype=np.uint8)
            allFrames = np.zeros((frameSize[0], frameSize[1], numOfFrames), dtype=np.uint8)
            for c in xrange(3) :
                for i in xrange(numOfFrames) :
#                     print i
                    self.updateOperationTextSignal.emit("Loading channel {0} of all images".format(c+1))
                    allFrames[:, :, i] = np.array(Image.open(frameLocs[i]))[:, :, c]
                    if np.mod(i, 10) == 0 :
                        self.updateOperationProgressSignal.emit(float(i+numOfFrames*c)/float(numOfFrames*3)*100)
                    if self.abortRequested :
                        return
                self.updateOperationTextSignal.emit("Computing channel {0}'s median".format(c+1))
                medianImage[:, :, c] = np.median(allFrames, axis=-1)
                if self.abortRequested :
                    return
            self.updateOperationTextSignal.emit("Saving computed median")
            Image.fromarray(medianImage).save(framesLocation + "/median.png")
            
            
class ComputeDistanceMatrix(LongOperationClass) :
    
    def __init__(self, parent = None):
        super(ComputeDistanceMatrix, self).__init__(parent)
    
    def run(self, semanticSequence, distSaveLocation, costSaveLocation, isMovingSprite, sigmaMultiplier) :
        numFrames = len(semanticSequence[DICT_FRAMES_LOCATIONS].keys())
        downsampleRate = 1
        if numFrames > 0 :
            progress = 0.0
            sequenceLocation = "/".join(semanticSequence[DICT_SEQUENCE_LOCATION].split(os.sep)[:-1]) + "/"
            ## get keys of tracked frames and size of frame
            sortedKeys = np.sort(semanticSequence[DICT_FRAMES_LOCATIONS].keys())
            frameSize = np.array(Image.open(semanticSequence[DICT_FRAMES_LOCATIONS][sortedKeys[0]])).shape[:2]
            if downsampleRate != 1 :
                frameSize = np.array(frameSize)/downsampleRate
            frameLocs = np.array([semanticSequence[DICT_FRAMES_LOCATIONS][key] for key in sortedKeys])
            
            ##
            budget = 0.25
            self.updateOperationTextSignal.emit("Computing patch of interest")
            
            ## find sub-patch if frames have been segmented
            if DICT_MASK_LOCATION in semanticSequence.keys() :
                topLeft = np.array([frameSize[0], frameSize[1]])
                bottomRight = np.array([0, 0])
#                 frameLocs = np.sort(glob.glob(semanticSequence[DICT_MASK_LOCATION]+"frame-0*.png"))
                for i, frameLoc in enumerate(frameLocs) :
                    frameName = frameLoc.split(os.sep)[-1]
                    if os.path.isfile(semanticSequence[DICT_MASK_LOCATION]+frameName) :
                        if downsampleRate == 1 :
                            alpha = np.array(Image.open(semanticSequence[DICT_MASK_LOCATION]+frameName))[:, :, -1]
                        else :
                            tmp = Image.open(semanticSequence[DICT_MASK_LOCATION]+frameName)
                            tmp.thumbnail((frameSize[1], frameSize[0]), Image.ANTIALIAS)
                            alpha = np.array(tmp)[:, :, -1]
                        vis = np.argwhere(alpha != 0)
                        tl = np.min(vis, axis=0)
                        topLeft[0] = np.min([topLeft[0], tl[0]])
                        topLeft[1] = np.min([topLeft[1], tl[1]])

                        br = np.max(vis, axis=0)
                        bottomRight[0] = np.max([bottomRight[0], br[0]])
                        bottomRight[1] = np.max([bottomRight[1], br[1]])

                    ##
                    progress += 1.0/len(frameLocs)*budget
                    if np.mod(i, 10) == 0 :
                        self.updateOperationProgressSignal.emit(progress*100)
                    if self.abortRequested :
                        return
                
                if downsampleRate == 1 :
                    bgPatch = np.array(Image.open(sequenceLocation+"median.png"))[topLeft[0]:bottomRight[0], topLeft[1]:bottomRight[1], 0:3]/255.0
                else :
                    tmp = Image.open(sequenceLocation+"median.png")
                    tmp.thumbnail((frameSize[1], frameSize[0]), Image.ANTIALIAS)
                    bgPatch = np.array(tmp)[topLeft[0]:bottomRight[0], topLeft[1]:bottomRight[1], 0:3]/255.0
            else :
                topLeft = np.array([0, 0])
                bottomRight = np.array([frameSize[0], frameSize[1]])
#                 frameLocs = np.sort([semanticSequence[DICT_FRAMES_LOCATIONS][key] for key in sortedKeys])
                bgPatch = np.zeros([frameSize[0], frameSize[1], 3])

                ##
                progress += budget
                self.updateOperationProgressSignal.emit(progress*100)

            if self.abortRequested :
                return
            budget = 0.1
            self.updateOperationTextSignal.emit("Rendering bounding boxes")

            ## render bboxes
            if DICT_BBOXES in semanticSequence.keys() :
                numVisibile = np.zeros(numFrames, int)
                renderedBBoxes = np.zeros((np.prod(frameSize), numFrames), np.uint8)
                for i, key in enumerate(sortedKeys) :
                    img = np.zeros((frameSize[0], frameSize[1]), np.uint8)
                    if key in semanticSequence[DICT_BBOXES] :
                        cv2.fillConvexPoly(img, semanticSequence[DICT_BBOXES][key].astype(int)[[0, 1, 2, 3, 0], :], 1)
                    renderedBBoxes[:, i] = img.flatten()
                    numVisibile[i] = len(np.argwhere(img.flatten() == 1))
                    
                    ##
                    progress += 1.0/len(sortedKeys)*budget
                    if np.mod(i, 10) == 0 :
                        self.updateOperationProgressSignal.emit(progress*100)
                    if self.abortRequested :
                        return
                        
            else :
                numVisibile = np.ones(numFrames, int)*np.prod(frameSize)
                renderedBBoxes = np.ones((1, numFrames), np.uint8)*np.sqrt(np.prod(frameSize))
                
                ##
                progress += budget
                self.updateOperationProgressSignal.emit(progress*100)

            if self.abortRequested :
                return
            ## figure out how to split the data to fit into memory
            memNeededPerFrame = np.prod(bgPatch.shape)*8/1000000.0#*len(semanticSequence[DICT_BBOXES])
            maxMemToUse = psutil.virtual_memory()[1]/1000000*0.65/2 ## use 0.5 of the available memory
            numFramesThatFit = np.round(maxMemToUse/memNeededPerFrame)
            numBlocks = int(np.ceil(numFrames/numFramesThatFit))
            blockSize = int(np.ceil(numFrames/float(numBlocks)))
            print "need", memNeededPerFrame*numFrames, "MBs: splitting into", numBlocks, "blocks of", blockSize, "frames (", blockSize*memNeededPerFrame, "MBs)"; sys.stdout.flush()

            frameIdxs = np.arange(numFrames)
            distMat = np.zeros([numFrames, numFrames])

            ##
            budget = 0.6
            totalBlocks = np.sum(np.arange(1, numBlocks+1))
            self.updateOperationTextSignal.emit("Computing distance matrix (split into {0} blocks)\nMoving sprite: {1}, Jump quality: {2}".format(totalBlocks, isMovingSprite, sigmaMultiplier))
            for i in xrange(numBlocks) :
                idxsToUse1 = frameIdxs[i*blockSize:(i+1)*blockSize]

                if self.abortRequested :
                    return

                f1s = np.zeros(np.hstack([bgPatch.shape[0], bgPatch.shape[1], 3, len(idxsToUse1)]), dtype=float)
                for idx, frame in enumerate(frameLocs[idxsToUse1]) :
                    if DICT_MASK_LOCATION in semanticSequence.keys() :
                        frameName = frame.split(os.sep)[-1]
                        if os.path.isfile(semanticSequence[DICT_MASK_LOCATION]+frameName) :
                            if downsampleRate == 1 :
                                img = np.array(Image.open(semanticSequence[DICT_MASK_LOCATION]+frameName), dtype=float)[topLeft[0]:bottomRight[0], topLeft[1]:bottomRight[1], :]
                            else :
                                tmp = Image.open(semanticSequence[DICT_MASK_LOCATION]+frameName)
                                tmp.thumbnail((frameSize[1], frameSize[0]), Image.ANTIALIAS)
                                img = np.array(tmp, dtype=float)[topLeft[0]:bottomRight[0], topLeft[1]:bottomRight[1], :]
                            alpha = img[:, :, -1]/255.0
                            f1s[:, :, :, idx] = ((img[:, :, :-1]/255.0)*np.reshape(alpha, (alpha.shape[0], alpha.shape[1], 1)) + 
                                                 bgPatch*(1.0-np.reshape(alpha, (alpha.shape[0], alpha.shape[1], 1))))
                    else :
                        if downsampleRate == 1 :
                            f1s[:, :, :, idx] = np.array(Image.open(frame), dtype=float)[topLeft[0]:bottomRight[0], topLeft[1]:bottomRight[1], :3]/255.0
                        else :
                            tmp = Image.open(frame)
                            tmp.thumbnail((frameSize[1], frameSize[0]),Image.ANTIALIAS)
                            f1s[:, :, :, idx] = np.array(tmp, dtype=float)[topLeft[0]:bottomRight[0], topLeft[1]:bottomRight[1], :3]/255.0
                    if self.abortRequested :
                        del f1s
                        return
                    ##
                    progress += 1.0/len(idxsToUse1)*1.0/totalBlocks*budget*0.85
                    if np.mod(idx, 10) == 0 :
                        self.updateOperationProgressSignal.emit(progress*100)
                
                if self.abortRequested :
                    del f1s
                    return
                data1 = np.reshape(f1s, [np.prod(f1s.shape[0:-1]), f1s.shape[-1]]).T
                distMat[i*blockSize:i*blockSize+len(idxsToUse1), i*blockSize:i*blockSize+len(idxsToUse1)] = ssd(data1)

                if self.abortRequested :
                    del f1s, data1
                    return
                ##
                progress += 1.0/totalBlocks*budget*0.15
                self.updateOperationProgressSignal.emit(progress*100)

                for j in xrange(i+1, numBlocks) :
                    idxsToUse2 = frameIdxs[j*blockSize:(j+1)*blockSize]

                    f2s = np.zeros(np.hstack([bgPatch.shape[0], bgPatch.shape[1], 3, len(idxsToUse2)]), dtype=float)
                    for idx, frame in enumerate(frameLocs[idxsToUse2]) :
                        if DICT_MASK_LOCATION in semanticSequence.keys() :
                            frameName = frame.split(os.sep)[-1]
                            if os.path.isfile(semanticSequence[DICT_MASK_LOCATION]+frameName) :
                                if downsampleRate == 1 :
                                    img = np.array(Image.open(semanticSequence[DICT_MASK_LOCATION]+frameName), dtype=float)[topLeft[0]:bottomRight[0], topLeft[1]:bottomRight[1], :]
                                else :
                                    tmp = Image.open(semanticSequence[DICT_MASK_LOCATION]+frameName)
                                    tmp.thumbnail((frameSize[1], frameSize[0]), Image.ANTIALIAS)
                                    img = np.array(tmp, dtype=float)[topLeft[0]:bottomRight[0], topLeft[1]:bottomRight[1], :]
                                alpha = img[:, :, -1]/255.0
                                f2s[:, :, :, idx] = ((img[:, :, :-1]/255.0)*np.reshape(alpha, (alpha.shape[0], alpha.shape[1], 1)) + 
                                                     bgPatch*(1.0-np.reshape(alpha, (alpha.shape[0], alpha.shape[1], 1))))
                        else :
                            if downsampleRate == 1 :
                                f2s[:, :, :, idx] = np.array(Image.open(frame), dtype=float)[topLeft[0]:bottomRight[0], topLeft[1]:bottomRight[1], :3]/255.0
                            else :
                                tmp = Image.open(frame)
                                tmp.thumbnail((frameSize[1], frameSize[0]), Image.ANTIALIAS)
                                f2s[:, :, :, idx] = np.array(tmp, dtype=float)[topLeft[0]:bottomRight[0], topLeft[1]:bottomRight[1], :3]/255.0
                                
                        if self.abortRequested :
                            del f1s, f2s, data1
                            return
                        ##
                        progress += 1.0/len(idxsToUse2)*1.0/totalBlocks*budget*0.85
                        if np.mod(idx, 10) == 0 :
                            self.updateOperationProgressSignal.emit(progress*100)
                        
                    if self.abortRequested :
                        del f1s, f2s, data1
                        return
                    data2 = np.reshape(f2s, [np.prod(f2s.shape[0:-1]), f2s.shape[-1]]).T
                    distMat[i*blockSize:i*blockSize+len(idxsToUse1), j*blockSize:j*blockSize+len(idxsToUse2)] = ssd2(data1, data2)
                    distMat[j*blockSize:j*blockSize+len(idxsToUse2), i*blockSize:i*blockSize+len(idxsToUse1)] = distMat[i*blockSize:i*blockSize+len(idxsToUse1), j*blockSize:j*blockSize+len(idxsToUse2)].T

                    del f2s, data2
                        
                    if self.abortRequested :
                        del f1s, data1
                        return

                    ##
                    progress += 1.0/totalBlocks*budget*0.15
                    self.updateOperationProgressSignal.emit(progress*100)

                del f1s, data1
                if self.abortRequested :
                    return


            ## due to imprecision I need to make this check
            distMat[distMat > 0.0] = np.sqrt(distMat[distMat > 0.0])
            distMat[distMat <= 0.0] = 0.0

            #     np.save(dataPath+dataSet+semanticSequence[DICT_SEQUENCE_NAME]+"-vanilla_distMat.npy", distMat/(numVisibile1.reshape((blockSize, 1)) + numVisibile1.reshape((1, blockSize))))
            
            if self.abortRequested :
                return
            self.updateOperationTextSignal.emit("Normalizing distance matrix")
            tmp = np.copy(renderedBBoxes.T).astype(float)
            numOverlapPixels = np.dot(tmp, tmp.T)
            del tmp
            distMat /= ((2.0*numOverlapPixels)/(numVisibile.reshape((numFrames, 1)) + numVisibile.reshape((1, numFrames)) + 0.01)+0.01)
            
            if self.abortRequested :
                return
            ## car sprites
#             transitionCostMat = computeTransitionMatrix(distMat, 4, 0.1, 20, True, True, 0.002)
            ## for flowers
#             transitionCostMat = computeTransitionMatrix(distMat, 4, 0.1, 20, True, False, 0.3)
            
            if isMovingSprite :
                ## allow speed up and loop at last frame
                transitionCostMat = computeTransitionMatrix(distMat, 4, 0.1, 20, True, True, sigmaMultiplier)
            else :
                ## filter all short jumps and do not loop at the last frame
                transitionCostMat = computeTransitionMatrix(distMat, 4, 0.1, 20, False, False, sigmaMultiplier)
            
            np.save(distSaveLocation, distMat)
            np.save(costSaveLocation, transitionCostMat)
            
            self.updateOperationProgressSignal.emit(100.0)


# In[43]:

# filterSize = 4
# threshPercentile = 0.1 ## percentile of transitions to base threshold on
# minJumpLength = 20
# onlyBackwards = True ## indicates if only backward jumps need filtering out (i.e. the syntehsised sequence can be sped up but not slowed down)
# loopOnLast = True ## indicates if an empty frame has been added at the end of the sequence that the synthesis can keep showing without concenquences
# sigmaMultiplier = 0.002
# distMat = np.load(testSequence['sequence_precomputed_distance_matrix_location'])

def computeTransitionMatrix(distMat, filterSize, threshPercentile, minJumpLength, onlyBackwards, loopOnLast, sigmaMultiplier) :

# for seqLoc in np.sort(glob.glob("/home/ilisescu/PhD/data/street/semantic_sequence*.npy"))[1:2] :
#     testSequence = np.load(seqLoc).item()
#     print testSequence[DICT_SEQUENCE_NAME]
#     gwv.showCustomGraph(distMat)
    
    ## filter to preserve dynamics
    kernel = np.eye(filterSize*2+1)
    
    optimizedDistMat = cv2.filter2D(distMat, -1, kernel)
    correction = 1
    
#     gwv.showCustomGraph(optimizedDistMat)
    
    ## init costs
#     testCosts = np.zeros_like(optimizedDistMat)
#     testCosts[0:-1, 0:-1] = np.copy(optimizedDistMat[1:, 0:-1])
#     testCosts[-1, 1:] = optimizedDistMat
    testCosts = np.copy(np.roll(optimizedDistMat, 1, axis=1))    
    
    # find threshold to use based on percentile
    thresh = np.sort(testCosts.flatten())[int(len(testCosts.flatten())*threshPercentile)]
    print "THRESH", thresh
    
    sigma = np.average(testCosts)*sigmaMultiplier
    
    ## don't want to jump too close so increase costs in a window
    if onlyBackwards :
        tmp = (np.triu(np.ones(optimizedDistMat.shape), k=2) +
               np.tril(np.ones(optimizedDistMat.shape), k=-minJumpLength) +
               np.eye(optimizedDistMat.shape[0], k=1))
    else :
        tmp = (np.triu(np.ones(optimizedDistMat.shape), k=minJumpLength) +
               np.tril(np.ones(optimizedDistMat.shape), k=-minJumpLength) +
               np.eye(optimizedDistMat.shape[0], k=1))
    tmp[tmp == 0] = 10.0
    testCosts *= tmp
    
    
    ## actual filtering
    invalidJumps = testCosts > thresh
    testCosts[invalidJumps] = GRAPH_MAX_COST
    testCosts[np.negative(invalidJumps)] = np.exp(testCosts[np.negative(invalidJumps)]/sigma)
    
    
#     ## adding extra rows and columns to compensate for the index shift indicated by correction
#     testCosts = np.concatenate((testCosts,
#                                 np.ones((testCosts.shape[0], correction))*(np.max(testCosts)-np.min(testCosts)) + np.min(testCosts)), axis=1)
#     testCosts = np.concatenate((testCosts,
#                                 np.ones((correction, testCosts.shape[1]))*(np.max(testCosts)-np.min(testCosts)) + np.min(testCosts)), axis=0)
    
#     ## setting transition from N-1 to N to minCost
#     testCosts[-2, -1] = np.min(testCosts)
    
    if loopOnLast :
        ## setting the looping from the last frame
        testCosts[-2, 0] = 0.0#np.min(testCosts)
        ## setting the looping from the empty frame and in place looping
        testCosts[-1, 0] = testCosts[-1, -1] = 0.0#np.min(testCosts)
    else :
        testCosts[-1, 0] = np.max(testCosts)
    
#     gwv.showCustomGraph(testCosts)
    
    return testCosts
    
#     testSequence[DICT_TRANSITION_COSTS_LOCATION] = "/".join(seqLoc.split("/")[:-1])+"/"+"transition_costs_no_normalization-"+testSequence[DICT_SEQUENCE_NAME]+".npy"
#     print 
#     print testSequence[DICT_TRANSITION_COSTS_LOCATION], testCosts.shape
#     print "------------------"
#     np.save(testSequence[DICT_TRANSITION_COSTS_LOCATION], testCosts)
#     np.save(testSequence[DICT_SEQUENCE_LOCATION], testSequence)


# In[ ]:

class TrackerError(Exception) :
    pass

class SemanticsDefinitionTab(QtGui.QWidget) :
    
    def __init__(self, seqDir, parent=None):
        super(SemanticsDefinitionTab, self).__init__(parent)
        
        self.numOfFrames = 0
        self.loadPath = os.path.expanduser("~")
        
        self.createGUI()
        
        self.settingBBox = False
        self.movingBBox = False
        self.rotatingBBox = False
        self.movingSegment = False
        self.bboxIsSet = False ## prevents stuff to happen if the bounding box is being set (because being translated, rotated...)
        self.scribbling = False
        self.isModeBBox = True
        self.bboxHasBeenDrawn = False ## if a bbox has been drawn, then it makes sense I keep drawing it to check if handles are available
        self.bboxChangedAndSaved = True ## used to figure out if I should render the a green bbox (only when self.bbox == bbox of current frame)
        
        self.brushSize = 20
        
        self.bbox = np.array([QtCore.QPointF(), QtCore.QPointF(), QtCore.QPointF(), 
                              QtCore.QPointF()])
        self.centerPoint = QtCore.QPointF()
        
        self.prevPoint = None
        self.copiedBBox = np.array([QtCore.QPointF(), QtCore.QPointF(), QtCore.QPointF(),
                              QtCore.QPointF()])
        self.copiedCenter = QtCore.QPointF()
        
        self.tracker = None
        self.tracking = False
        self.segmenting = False
        
        self.semanticSequences = []
        self.selectedSemSequenceIdx = -1
        
        self.frameIdx = 0
        self.frameImg = None
        self.overlayImg = QtGui.QImage(QtCore.QSize(1, 1), QtGui.QImage.Format_ARGB32)
        self.scribble = QtGui.QImage(QtCore.QSize(1, 1), QtGui.QImage.Format_RGB888)
        self.seqDir = ""
        self.frameLocs = []
        self.numOfFrames = 0
        self.bgImage = np.zeros([720, 1280, 3], np.uint8)
        
        self.loadFrameSequence(seqDir)
        
        self.operationThread = LongOperationThread()
        
#         self.semanticsToDraw = []

        self.showFrame(self.frameIdx)
        self.setFocus()
        
        
    def loadFrameSequencePressed(self) :
        seqDir = QtGui.QFileDialog.getExistingDirectory(self, "Load Frame Sequence", self.loadPath)
        self.loadFrameSequence(seqDir)
        
        return seqDir
        
    def loadFrameSequence(self, seqDir) :
        if seqDir != "" :
            self.tracker = None
            self.tracking = False

            self.semanticSequences = []
            self.selectedSemSequenceIdx = -1

            self.frameIdx = 0
            self.loadPath = os.sep.join(seqDir.split(os.sep)[:-1])
            
            maxChars = 30
            desiredText = "Loaded Frame Sequence: " + seqDir
            desiredText = "\n ".join([desiredText[i:i+maxChars] for i in np.arange(0, len(desiredText), maxChars)])
            self.loadedFrameSequenceLabel.setText(desiredText)
            
            self.seqDir = seqDir
            self.frameLocs = np.sort(glob.glob(self.seqDir+"/frame-0*.png"))
            self.numOfFrames = len(self.frameLocs)
            
#             self.allFrames = np.zeros((self.bgImage.shape[0], self.bgImage.shape[1], self.bgImage.shape[2], self.numOfFrames), dtype=np.uint8)
#             for idx, loc in enumerate(self.frameLocs) :
#                 self.allFrames[:, :, :, idx] = np.array(Image.open(loc)).astype(np.uint8)
            
            if self.numOfFrames > 0 :
                if os.path.isfile(self.seqDir+"/median.png") :
                    self.bgImage = np.array(Image.open(self.seqDir+"/median.png"))[:, :, 0:3]
                else :
                    frameSize = np.array(Image.open(self.frameLocs[0])).shape
                    self.bgImage = np.zeros([frameSize[0], frameSize[1], 3], np.uint8)
            else :
                self.bgImage = np.zeros([720, 1280, 3], np.uint8)
    
            self.frameIdxSlider.setMaximum(np.max([0, self.numOfFrames-1]))
            self.frameIdxSpinBox.setRange(0, np.max([0, self.numOfFrames-1]))
        
            self.allXs = np.arange(self.bgImage.shape[1], dtype=np.float32).reshape((1, self.bgImage.shape[1])).repeat(self.bgImage.shape[0], axis=0)
            self.allYs = np.arange(self.bgImage.shape[0], dtype=np.float32).reshape((self.bgImage.shape[0], 1)).repeat(self.bgImage.shape[1], axis=1)
            
            self.loadSemanticSequences()
            
        self.showFrame(self.frameIdx)
        
    def addFramesToSequencePressed(self) :
        if self.selectedSemSequenceIdx < len(self.semanticSequences) and self.selectedSemSequenceIdx >= 0 :
            bbox = []
            if self.bboxIsSet :
                bbox = self.bbox

            addFramesToSequenceDialog = AddFramesToSequenceDialog(self, self.frameLocs, bbox, "Add Frames To Actor Sequence")
            accept = bool(addFramesToSequenceDialog.exec_())

            if accept :
                print "Add Frames [{0}, {1}]".format(addFramesToSequenceDialog.startFrame, addFramesToSequenceDialog.endFrame)
                ## update sequence dict
                if self.selectedSemSequenceIdx < len(self.semanticSequences) and self.selectedSemSequenceIdx >= 0 :

                    rot = np.mod(np.arctan2(-(self.bbox[TL_IDX]-self.bbox[TR_IDX]).y(), 
                                             (self.bbox[TL_IDX]-self.bbox[TR_IDX]).x()),2*np.pi)

                    if addFramesToSequenceDialog.setBBoxBox.isChecked() :
                        if DICT_BBOXES not in self.semanticSequences[self.selectedSemSequenceIdx].keys() :
                            self.semanticSequences[self.selectedSemSequenceIdx][DICT_BBOXES] = {}
                            self.semanticSequences[self.selectedSemSequenceIdx][DICT_BBOX_CENTERS] = {}
                            self.semanticSequences[self.selectedSemSequenceIdx][DICT_BBOX_ROTATIONS] = {}

                    for i in xrange(addFramesToSequenceDialog.startFrame, addFramesToSequenceDialog.endFrame+1) :
                        if addFramesToSequenceDialog.setBBoxBox.isChecked() :
                            self.semanticSequences[self.selectedSemSequenceIdx][DICT_BBOXES][i] = np.array([[self.bbox[TL_IDX].x(), self.bbox[TL_IDX].y()], 
                                                                                                                [self.bbox[TR_IDX].x(), self.bbox[TR_IDX].y()], 
                                                                                                                [self.bbox[BR_IDX].x(), self.bbox[BR_IDX].y()], 
                                                                                                                [self.bbox[BL_IDX].x(), self.bbox[BL_IDX].y()]])
                            self.semanticSequences[self.selectedSemSequenceIdx][DICT_BBOX_CENTERS][i] = np.array([self.centerPoint.x(), self.centerPoint.y()])
                            self.semanticSequences[self.selectedSemSequenceIdx][DICT_BBOX_ROTATIONS][i] = rot
                            self.bboxChangedAndSaved = True

                        self.semanticSequences[self.selectedSemSequenceIdx][DICT_FRAMES_LOCATIONS][i] = self.frameLocs[i]

                    self.setSemanticsToDraw()
                    
        self.setFocus()
                
    def addEndFrameToSequencePressed(self) :
        if self.selectedSemSequenceIdx < len(self.semanticSequences) and self.selectedSemSequenceIdx >= 0 :
            endFrame = np.max(self.semanticSequences[self.selectedSemSequenceIdx][DICT_FRAMES_LOCATIONS].keys())
            proceed = True
            if DICT_MASK_LOCATION in self.semanticSequences[self.selectedSemSequenceIdx].keys() :
                if DICT_BBOXES in self.semanticSequences[self.selectedSemSequenceIdx].keys() and endFrame not in self.semanticSequences[self.selectedSemSequenceIdx][DICT_BBOXES].keys() :
                    proceed = QtGui.QMessageBox.question(self, 'Already added empty frame', "It looks like you already added an empty frame for "+
                                                         "{0}.\nDo you want to add another?".format(self.semanticSequences[self.selectedSemSequenceIdx][DICT_SEQUENCE_NAME]), 
                                                         QtGui.QMessageBox.Yes | QtGui.QMessageBox.No, QtGui.QMessageBox.No) == QtGui.QMessageBox.Yes
            
            if proceed :
                print endFrame, self.numOfFrames
                if self.frameIdx < self.numOfFrames-1 :
                    self.semanticSequences[self.selectedSemSequenceIdx][DICT_FRAMES_LOCATIONS][endFrame+1] = self.frameLocs[endFrame+1]
                elif endFrame == self.numOfFrames-1 :
                    self.semanticSequences[self.selectedSemSequenceIdx][DICT_FRAMES_LOCATIONS][endFrame+1] = ""

                    self.frameIdxSlider.setMaximum(np.max([0, self.numOfFrames]))
                    self.frameIdxSpinBox.setRange(0, np.max([0, self.numOfFrames]))
                self.setSemanticsToDraw()

    def computeMedianPressed(self) :
        proceed = True
        if self.seqDir != "" and os.path.isfile(self.seqDir + "/median.png") :
            proceed = QtGui.QMessageBox.question(self, 'Compute Median Image',
                                "The median image for this sequence already exists\nDo you want to override?", 
                                QtGui.QMessageBox.Yes | QtGui.QMessageBox.No, QtGui.QMessageBox.No) == QtGui.QMessageBox.Yes
            
        if self.seqDir != "" and proceed :
            loadingDialog = LoadingDialog(self, "Compute Median Image")
            loadingDialog.doCancelSignal.connect(self.operationThread.doQuit)
            self.operationThread.updateOperationProgressSignal.connect(loadingDialog.setOperationProgress)
            self.operationThread.updateOperationTextSignal.connect(loadingDialog.setOperationText)
            self.operationThread.doneOperationSignal.connect(loadingDialog.doneLoading)
            self.operationThread.doRun(ComputeMedian(), [self.seqDir])
            returnValue = loadingDialog.exec_()
            print "MEDIAN EXIT:", returnValue
            if returnValue == 0 :
                self.bgImage = np.array(Image.open(self.seqDir+"/median.png"))[:, :, 0:3]
                
    def computeDistanceMatrixPressed(self) :
        if self.selectedSemSequenceIdx >= 0 and self.selectedSemSequenceIdx < len(self.semanticSequences) :
            loadingDialog = LoadingDialog(self, "Compute Distance Matrix")
            loadingDialog.doCancelSignal.connect(self.operationThread.doQuit)
            self.operationThread.updateOperationProgressSignal.connect(loadingDialog.setOperationProgress)
            self.operationThread.updateOperationTextSignal.connect(loadingDialog.setOperationText)
            self.operationThread.doneOperationSignal.connect(loadingDialog.doneLoading)

            distanceLocation = self.seqDir + "/overlap_normalization_distMat-" + self.semanticSequences[self.selectedSemSequenceIdx][DICT_SEQUENCE_NAME] + ".npy"
            costLocation = self.seqDir + "/transition_costs_no_normalization-" + self.semanticSequences[self.selectedSemSequenceIdx][DICT_SEQUENCE_NAME] + ".npy"
            self.operationThread.doRun(ComputeDistanceMatrix(), [self.semanticSequences[self.selectedSemSequenceIdx], distanceLocation, costLocation,
                                                                 self.isMovingSpriteBox.isChecked(), self.sigmaMultiplierSlider.value()/1000.0])
            returnValue = loadingDialog.exec_()
            print "DISTANCE EXIT:", returnValue
            if returnValue == 0 :
                self.semanticSequences[self.selectedSemSequenceIdx][DICT_DISTANCE_MATRIX_LOCATION] = distanceLocation
                self.semanticSequences[self.selectedSemSequenceIdx][DICT_TRANSITION_COSTS_LOCATION] = costLocation
        self.setFocus()
                

    def cleanup(self) :
        if self.operationThread.isRunning() :
            print "CLOSING THREAD"; sys.stdout.flush()
            self.operationThread.doQuit()
            
        for index, seq in enumerate(self.semanticSequences) :
            if index in self.preloadedPatches and self.preloadedPatches[index]['needs_saving'] :
                np.save(seq[DICT_PATCHES_LOCATION], self.preloadedPatches[index])
                print "saved patches for", seq[DICT_SEQUENCE_NAME]
                
        try :
#             del self.allFrames
            del self.preloadedPatches
        except :
            print
        
    def initTracker(self) :
        im0 = cv2.imread(self.frameLocs[self.frameIdx])
        im_gray0 = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
        im_draw = np.copy(im0)
        
        self.tracker = CMT.CMT()
        self.tracker.estimate_scale = True
        self.tracker.estimate_rotation = True
        pause_time = 10
        
        self.tracker.initialise(im_gray0, (self.bbox[TL_IDX].x(), self.bbox[TL_IDX].y()), 
                                          (self.bbox[BR_IDX].x(), self.bbox[BR_IDX].y()), 
                                          (self.bbox[TR_IDX].x(), self.bbox[TR_IDX].y()), 
                                          (self.bbox[BL_IDX].x(), self.bbox[BL_IDX].y()))
        
        rot = np.mod(np.arctan2(-(self.bbox[TL_IDX]-self.bbox[TR_IDX]).y(), 
                                 (self.bbox[TL_IDX]-self.bbox[TR_IDX]).x()),2*np.pi)
        
            
        ## update sequence dict
        if self.selectedSemSequenceIdx < len(self.semanticSequences) and self.selectedSemSequenceIdx >= 0 :
            if DICT_BBOXES not in self.semanticSequences[self.selectedSemSequenceIdx].keys() :
                self.semanticSequences[self.selectedSemSequenceIdx][DICT_BBOXES] = {}
                self.semanticSequences[self.selectedSemSequenceIdx][DICT_BBOX_CENTERS] = {}
                self.semanticSequences[self.selectedSemSequenceIdx][DICT_BBOX_ROTATIONS] = {}
                
            self.semanticSequences[self.selectedSemSequenceIdx][DICT_BBOXES][self.frameIdx] = np.array([[self.bbox[TL_IDX].x(), self.bbox[TL_IDX].y()], 
                                                                                                [self.bbox[TR_IDX].x(), self.bbox[TR_IDX].y()], 
                                                                                                [self.bbox[BR_IDX].x(), self.bbox[BR_IDX].y()], 
                                                                                                [self.bbox[BL_IDX].x(), self.bbox[BL_IDX].y()]])
            self.semanticSequences[self.selectedSemSequenceIdx][DICT_BBOX_CENTERS][self.frameIdx] = np.array([self.centerPoint.x(), self.centerPoint.y()])
            self.semanticSequences[self.selectedSemSequenceIdx][DICT_BBOX_ROTATIONS][self.frameIdx] = np.copy(rot)
            self.semanticSequences[self.selectedSemSequenceIdx][DICT_FRAMES_LOCATIONS][self.frameIdx] = self.frameLocs[self.frameIdx]
            self.bboxChangedAndSaved = True
            
        ## draw
        if self.drawOverlay(False, drawingSavedBBox=self.bboxChangedAndSaved) :
            self.frameLabel.setOverlay(self.overlayImg)
        
    def trackInFrame(self) :
        im = cv2.imread(self.frameLocs[self.frameIdx])
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im_draw = np.copy(im)
    
        self.tracker.process_frame(im_gray)
    
        # Display results
    
        # Draw updated estimate
        if self.tracker.has_result:
            ## update bbox
            self.bbox[TL_IDX].setX(self.tracker.tl[0])
            self.bbox[TL_IDX].setY(self.tracker.tl[1])
            self.bbox[TR_IDX].setX(self.tracker.tr[0])
            self.bbox[TR_IDX].setY(self.tracker.tr[1])
            self.bbox[BR_IDX].setX(self.tracker.br[0])
            self.bbox[BR_IDX].setY(self.tracker.br[1])
            self.bbox[BL_IDX].setX(self.tracker.bl[0])
            self.bbox[BL_IDX].setY(self.tracker.bl[1])
            
            ## update center point. NOTE: bbox center point is != self.tracker.center(= center of feature points)
            minX = np.min((self.bbox[TL_IDX].x(), self.bbox[TR_IDX].x(), self.bbox[BR_IDX].x(), self.bbox[BL_IDX].x()))
            maxX = np.max((self.bbox[TL_IDX].x(), self.bbox[TR_IDX].x(), self.bbox[BR_IDX].x(), self.bbox[BL_IDX].x()))
            minY = np.min((self.bbox[TL_IDX].y(), self.bbox[TR_IDX].y(), self.bbox[BR_IDX].y(), self.bbox[BL_IDX].y()))
            maxY = np.max((self.bbox[TL_IDX].y(), self.bbox[TR_IDX].y(), self.bbox[BR_IDX].y(), self.bbox[BL_IDX].y()))
            self.centerPoint.setX(minX + (maxX - minX)/2.0)
            self.centerPoint.setY(minY + (maxY - minY)/2.0)
            
            ## compute rotation in radians
            rot = np.mod(np.arctan2(-(self.bbox[TL_IDX]-self.bbox[TR_IDX]).y(), 
                                     (self.bbox[TL_IDX]-self.bbox[TR_IDX]).x()),2*np.pi)
#             print "rotation", rot*180.0/np.pi
            
            ## update sequence dict
            if self.selectedSemSequenceIdx < len(self.semanticSequences) and self.selectedSemSequenceIdx >= 0 :
                if DICT_BBOXES not in self.semanticSequences[self.selectedSemSequenceIdx].keys() :
                    self.semanticSequences[self.selectedSemSequenceIdx][DICT_BBOXES] = {}
                    self.semanticSequences[self.selectedSemSequenceIdx][DICT_BBOX_CENTERS] = {}
                    self.semanticSequences[self.selectedSemSequenceIdx][DICT_BBOX_ROTATIONS] = {}
                    
                self.semanticSequences[self.selectedSemSequenceIdx][DICT_BBOXES][self.frameIdx] = np.array([[self.bbox[TL_IDX].x(), self.bbox[TL_IDX].y()], 
                                                                                                    [self.bbox[TR_IDX].x(), self.bbox[TR_IDX].y()], 
                                                                                                    [self.bbox[BR_IDX].x(), self.bbox[BR_IDX].y()], 
                                                                                                    [self.bbox[BL_IDX].x(), self.bbox[BL_IDX].y()]])
                self.semanticSequences[self.selectedSemSequenceIdx][DICT_BBOX_CENTERS][self.frameIdx] = np.array([self.centerPoint.x(), self.centerPoint.y()])
                self.semanticSequences[self.selectedSemSequenceIdx][DICT_BBOX_ROTATIONS][self.frameIdx] = rot
                self.semanticSequences[self.selectedSemSequenceIdx][DICT_FRAMES_LOCATIONS][self.frameIdx] = self.frameLocs[self.frameIdx]
                self.bboxChangedAndSaved = True
            
            ## draw
            if self.drawOverlay(drawingSavedBBox=self.bboxChangedAndSaved) :
                self.frameLabel.setOverlay(self.overlayImg)
                
        else :
            raise TrackerError("Tracker Failed")
        
        
    def trackInVideo(self, goForward) :
        if not self.isModeBBox :
            self.toggleDefineMode()
        if goForward :
            self.frameIdx += 1
        else :
            self.frameIdx -= 1
            
        while self.frameIdx >= 0 and self.frameIdx < self.numOfFrames and self.tracking :
            try :
                self.trackInFrame()
            except TrackerError as e :
                QtGui.QMessageBox.critical(self, "Tracker Failed", ("The tracker has failed to produce a result"))
                break
            except Exception as e :
                QtGui.QMessageBox.critical(self, "Tracker Failed", (str(e)))
                
            self.frameIdxSpinBox.setValue(self.frameIdx)
            QtCore.QCoreApplication.processEvents()
        
            # Advance frame number
            if goForward :
                self.frameIdx += 1
            else :
                self.frameIdx -= 1
        
        self.tracking = False
        self.frameIdxSpinBox.setValue(self.frameIdx)
    
    def showFrame(self, idx, doRefreshSegmentation=False) :
        self.frameIdx = idx
        if idx >= 0 and idx < len(self.frameLocs) :
#             print "showing", self.frameIdx
            areFramesSegmented = False
            ## HACK ##
            im = np.ascontiguousarray(np.array(Image.open(self.frameLocs[self.frameIdx]))[:, :, :3])
#             im = np.ascontiguousarray(self.allFrames[:, :, :, self.frameIdx])
            if im.shape[-1] == 3 :
                self.frameImg = QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_RGB888);
            else :
                ## this is a hack to deal with candle_wind
                areFramesSegmented = True
                bg = np.zeros([im.shape[0], im.shape[1], 3], np.uint8)
                self.frameImg = QtGui.QImage(bg.data, bg.shape[1], bg.shape[0], bg.strides[0], QtGui.QImage.Format_RGB888);
                
                qim = QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_ARGB32);
                self.frameLabel.setSegmentedImage(qim)

            self.frameLabel.setFixedSize(self.frameImg.width(), self.frameImg.height())
            self.frameLabel.setImage(self.frameImg)

            self.frameInfo.setText(self.frameLocs[self.frameIdx])

            if (self.selectedSemSequenceIdx < len(self.semanticSequences) and self.selectedSemSequenceIdx >= 0 and
                self.frameIdx in self.semanticSequences[self.selectedSemSequenceIdx][DICT_FRAMES_LOCATIONS].keys()) :
                ## set self.bbox to bbox computed for current frame if it exists
                if not self.tracking :
                    if (DICT_BBOXES in self.semanticSequences[self.selectedSemSequenceIdx].keys() and
                        self.frameIdx in self.semanticSequences[self.selectedSemSequenceIdx][DICT_BBOXES].keys()) :
                        self.bbox[TL_IDX].setX(self.semanticSequences[self.selectedSemSequenceIdx][DICT_BBOXES][self.frameIdx][TL_IDX, 0])
                        self.bbox[TL_IDX].setY(self.semanticSequences[self.selectedSemSequenceIdx][DICT_BBOXES][self.frameIdx][TL_IDX, 1])
                        self.bbox[TR_IDX].setX(self.semanticSequences[self.selectedSemSequenceIdx][DICT_BBOXES][self.frameIdx][TR_IDX, 0])
                        self.bbox[TR_IDX].setY(self.semanticSequences[self.selectedSemSequenceIdx][DICT_BBOXES][self.frameIdx][TR_IDX, 1])
                        self.bbox[BR_IDX].setX(self.semanticSequences[self.selectedSemSequenceIdx][DICT_BBOXES][self.frameIdx][BR_IDX, 0])
                        self.bbox[BR_IDX].setY(self.semanticSequences[self.selectedSemSequenceIdx][DICT_BBOXES][self.frameIdx][BR_IDX, 1])
                        self.bbox[BL_IDX].setX(self.semanticSequences[self.selectedSemSequenceIdx][DICT_BBOXES][self.frameIdx][BL_IDX, 0])
                        self.bbox[BL_IDX].setY(self.semanticSequences[self.selectedSemSequenceIdx][DICT_BBOXES][self.frameIdx][BL_IDX, 1])

                        self.centerPoint.setX(self.semanticSequences[self.selectedSemSequenceIdx][DICT_BBOX_CENTERS][self.frameIdx][0])
                        self.centerPoint.setY(self.semanticSequences[self.selectedSemSequenceIdx][DICT_BBOX_CENTERS][self.frameIdx][1])

                        self.bboxIsSet = True
                        self.bboxChangedAndSaved = True
                        if self.drawOverlay(False, drawingSavedBBox=self.bboxChangedAndSaved) :
                            self.frameLabel.setOverlay(self.overlayImg)
                    else :
                        self.bboxIsSet = False
                        self.bboxChangedAndSaved = False
                        if self.drawOverlay(False, False, False) :
                            self.frameLabel.setOverlay(self.overlayImg)
                            
                    ## deal with segmented image
                    if DICT_MASK_LOCATION in self.semanticSequences[self.selectedSemSequenceIdx].keys() :                    
                        ## deal with scribble
                        frameName = self.frameLocs[self.frameIdx].split(os.sep)[-1]
                        if self.scribble.size() != self.frameImg.size() :
                            self.scribble = self.scribble.scaled(self.frameImg.size())
                        self.scribble.fill(QtGui.QColor.fromRgb(255, 255, 255))

                        if os.path.isfile(self.semanticSequences[self.selectedSemSequenceIdx][DICT_MASK_LOCATION]+"scribble-"+frameName) :
                            im = np.ascontiguousarray(Image.open(self.semanticSequences[self.selectedSemSequenceIdx][DICT_MASK_LOCATION]+"scribble-"+frameName))
                            qim = QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_RGB888)
                            painter = QtGui.QPainter(self.scribble)
                            painter.drawImage(QtCore.QPoint(0, 0), qim)
                            self.frameLabel.setScribbleImage(self.scribble)
                        else :
                            self.frameLabel.setScribbleImage(None)
                            
                        im = self.getSegmentedImage(doRefreshSegmentation)
                        if np.all(im != None) :
                            im = np.ascontiguousarray(im)
                            qim = QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_ARGB32);
                            self.frameLabel.setSegmentedImage(qim)
                        else :
                            self.frameLabel.setSegmentedImage(None)
                    else :
                        if not areFramesSegmented :
                            self.frameLabel.setSegmentedImage(None)
                        self.frameLabel.setScribbleImage(None)
                        
                    if DICT_FRAME_SEMANTICS in self.semanticSequences[self.selectedSemSequenceIdx].keys() :
                        frame = self.frameIdx-np.min(self.semanticSequences[self.selectedSemSequenceIdx][DICT_FRAMES_LOCATIONS].keys())
                        if frame >= 0 and frame < len(self.semanticSequences[self.selectedSemSequenceIdx][DICT_FRAME_SEMANTICS]) :
                            sems = self.semanticSequences[self.selectedSemSequenceIdx][DICT_FRAME_SEMANTICS][frame, :]
                            self.frameInfo.setText(self.frameInfo.text() + " <b>showing \"{0}\" [{1}]</b>".format(self.semanticSequences[self.selectedSemSequenceIdx][DICT_SEMANTICS_NAMES][int(np.argmax(sems))], np.max(sems)))
                        
            else :
                self.bboxIsSet = False
                if self.drawOverlay(False, False, False) :
                    self.frameLabel.setOverlay(self.overlayImg)
                if not areFramesSegmented :
                    self.frameLabel.setSegmentedImage(None)
                self.frameLabel.setScribbleImage(None)
                
        else :
            self.frameLabel.setImage(None)
            self.bboxIsSet = False
            if self.drawOverlay(False, False, False) :
                self.frameLabel.setOverlay(self.overlayImg)
            self.frameLabel.setSegmentedImage(None)
            self.frameLabel.setScribbleImage(None)
            
            if (self.selectedSemSequenceIdx < len(self.semanticSequences) and self.selectedSemSequenceIdx >= 0 and
                self.frameIdx in self.semanticSequences[self.selectedSemSequenceIdx][DICT_FRAMES_LOCATIONS].keys()) :
                frameLocInfo = self.semanticSequences[self.selectedSemSequenceIdx][DICT_FRAMES_LOCATIONS][self.frameIdx]
                if frameLocInfo == "" :
                    frameLocInfo = "Empty frame"
                self.frameInfo.setText(frameLocInfo)
                
                if DICT_FRAME_SEMANTICS in self.semanticSequences[self.selectedSemSequenceIdx].keys() :
                    frame = self.frameIdx-np.min(self.semanticSequences[self.selectedSemSequenceIdx][DICT_FRAMES_LOCATIONS].keys())
                    if frame >= 0 and frame < len(self.semanticSequences[self.selectedSemSequenceIdx][DICT_FRAME_SEMANTICS]) :
                        sems = self.semanticSequences[self.selectedSemSequenceIdx][DICT_FRAME_SEMANTICS][frame, :]
                        self.frameInfo.setText(self.frameInfo.text() + " <b>showing \"{0}\" [{1}]</b>".format(self.semanticSequences[self.selectedSemSequenceIdx][DICT_SEMANTICS_NAMES][int(np.argmax(sems))], np.max(sems)))
            
            
    def getSegmentedImage(self, refresh=False) :
        ## returns bgra
        if DICT_BBOXES in self.semanticSequences[self.selectedSemSequenceIdx].keys() and self.frameIdx in self.semanticSequences[self.selectedSemSequenceIdx][DICT_BBOXES].keys() :
            startTime = time.time()
            frameName =  self.frameLocs[self.frameIdx].split(os.sep)[-1]
            
            if not refresh and self.selectedSemSequenceIdx in self.preloadedPatches and self.frameIdx in self.preloadedPatches[self.selectedSemSequenceIdx].keys() :
#                 print "showing preloaded"
                spritePatch = self.preloadedPatches[self.selectedSemSequenceIdx][self.frameIdx]
                currentFrame = np.zeros((self.bgImage.shape[0], self.bgImage.shape[1], 4), dtype=np.uint8)
                currentFrame[spritePatch['visible_indices'][:, 0]+spritePatch['top_left_pos'][0],
                             spritePatch['visible_indices'][:, 1]+spritePatch['top_left_pos'][1], :] = spritePatch['sprite_colors']
                ## spritePatch colors are bgra
                return currentFrame            
            elif not refresh and os.path.isfile(self.semanticSequences[self.selectedSemSequenceIdx][DICT_MASK_LOCATION]+frameName) :
                print "showing from disk"
                currentFrame = np.array(Image.open(self.semanticSequences[self.selectedSemSequenceIdx][DICT_MASK_LOCATION]+frameName))
            else :
                print "computed", 
                t = time.time()
                fgPatch, offset, patchSize, touchedBorders = getForegroundPatch(self.semanticSequences[self.selectedSemSequenceIdx], self.frameIdx, 
                                                                                self.frameImg.width(), self.frameImg.height())
                bgPatch = np.copy(self.bgImage[offset[1]:offset[1]+patchSize[0], offset[0]:offset[0]+patchSize[1], :])
    #             print "patches", time.time() - t
                t = time.time()

                if self.frameIdx-1 in self.semanticSequences[self.selectedSemSequenceIdx][DICT_BBOXES].keys() :
                    bgPrior, fgPrior = getPatchPriors(bgPatch, fgPatch, offset, patchSize, self.semanticSequences[self.selectedSemSequenceIdx],
                                                          self.frameIdx, self.seqDir+"/", self.allXs, self.allYs, prevFrameKey=self.frameIdx-1,
                                                          prevFrameAlphaLoc=self.seqDir+"/"+self.semanticSequences[self.selectedSemSequenceIdx][DICT_SEQUENCE_NAME] + "-maskedFlow/",
                                                          useOpticalFlow=self.doUseOpticalFlowPriorBox.isChecked(),
                                                          useDiffPatch=self.doUsePatchDiffPriorBox.isChecked(),
                                                          prevMaskImportance=self.prevMaskImportanceSpinBox.value(),
                                                          prevMaskDilate=self.prevMaskDilateSpinBox.value(),
                                                          prevMaskBlurSize=self.prevMaskBlurSizeSpinBox.value(),
                                                          prevMaskBlurSigma=self.prevMaskBlurSigmaSpinBox.value(),
                                                          diffPatchImportance=self.diffPatchImportanceSpinBox.value(),
                                                          diffPatchMultiplier=self.diffPatchMultiplierSpinBox.value())
    #                 print "priors with flow", time.time() - t
    #                 gwv.showCustomGraph(fgPrior)
                    t = time.time()
                else :
                    bgPrior, fgPrior = getPatchPriors(bgPatch, fgPatch, offset, patchSize, self.semanticSequences[self.selectedSemSequenceIdx],
                                                          self.frameIdx, self.seqDir+"/", self.allXs, self.allYs,
                                                          useOpticalFlow=self.doUseOpticalFlowPriorBox.isChecked(),
                                                          useDiffPatch=self.doUsePatchDiffPriorBox.isChecked(),
                                                          prevMaskImportance=self.prevMaskImportanceSpinBox.value(),
                                                          prevMaskDilate=self.prevMaskDilateSpinBox.value(),
                                                          prevMaskBlurSize=self.prevMaskBlurSizeSpinBox.value(),
                                                          prevMaskBlurSigma=self.prevMaskBlurSigmaSpinBox.value(),
                                                          diffPatchImportance=self.diffPatchImportanceSpinBox.value(),
                                                          diffPatchMultiplier=self.diffPatchMultiplierSpinBox.value())
    #                 print "priors without flow", time.time() - t
                    t = time.time()

                labels = mergePatches(bgPatch, fgPatch, bgPrior, fgPrior, offset, patchSize, touchedBorders,
                                      scribble=np.frombuffer(self.scribble.constBits(), dtype=np.uint8).reshape((self.scribble.height(), self.scribble.width(), 3)),
                                      useCenterSquare=self.doUseCenterSquareBox.isChecked(),
                                      useGradients=self.doUseGradientsCostBox.isChecked())
    #             print "merging", time.time() - t
                t = time.time()

                outputPatch = np.zeros((bgPatch.shape[0], bgPatch.shape[1], bgPatch.shape[2]+1), dtype=np.uint8)
                for i in xrange(labels.shape[0]) :
                    for j in xrange(labels.shape[1]) :
                        if labels[i, j] == 0 :
                            ## patA stands for the bgPatch but I want to set the pixels here to 0 to save space
                            outputPatch[i, j, 0:-1] = 0#bgPatch[i, j, :]
                        else :
                            outputPatch[i, j, 0:-1] = fgPatch[i, j, :]
                            outputPatch[i, j, -1] = 255
                            

                currentFrame = np.zeros((self.frameImg.height(), self.frameImg.width(), 4), dtype=np.uint8)
                currentFrame[offset[1]:offset[1]+patchSize[0], offset[0]:offset[0]+patchSize[1], :] = np.copy(outputPatch)
    #             print "putting together the frame", time.time() - t
                t = time.time()

                Image.fromarray((currentFrame).astype(np.uint8)).save(self.semanticSequences[self.selectedSemSequenceIdx][DICT_MASK_LOCATION] + frameName)
    #             print "saving", time.time() - t
                t = time.time()

                print "in", time.time() - startTime
            
            ## add currentFrame to preloadedPatches if this place is reached
            if self.selectedSemSequenceIdx not in self.preloadedPatches.keys() :
                self.preloadedPatches[self.selectedSemSequenceIdx] = {}
                self.preloadedPatches[self.selectedSemSequenceIdx]['needs_saving'] = True
            if DICT_PATCHES_LOCATION not in self.semanticSequences[self.selectedSemSequenceIdx] :
                self.semanticSequences[self.selectedSemSequenceIdx][DICT_PATCHES_LOCATION] = (self.seqDir+"/preloaded_patches-"+
                                                                                              self.semanticSequences[self.selectedSemSequenceIdx][DICT_SEQUENCE_NAME]+".npy")
            visiblePixels = np.argwhere(currentFrame[:, :, -1] != 0).astype(np.uint16)
            topLeft = np.min(visiblePixels, axis=0)
            patchSize = np.max(visiblePixels, axis=0) - topLeft + 1
            
            currentFrame = currentFrame[:, :, [2, 1, 0, 3]]
            self.preloadedPatches[self.selectedSemSequenceIdx][self.frameIdx] = {'top_left_pos':topLeft,
                                                                                 'sprite_colors':currentFrame[visiblePixels[:, 0], visiblePixels[:, 1], :], 
                                                                                 'visible_indices': visiblePixels-topLeft, 'patch_size': patchSize}
            
            self.preloadedPatches[self.selectedSemSequenceIdx]['needs_saving'] = True
                
            
            return currentFrame
        else :
            return None
            
    def updateBBox(self) :
        if self.settingBBox :
#             print "settingBBox"
            self.bbox[TR_IDX] = QtCore.QPointF(self.bbox[BR_IDX].x(), self.bbox[TL_IDX].y())
            self.bbox[BL_IDX] = QtCore.QPointF(self.bbox[TL_IDX].x(), self.bbox[BR_IDX].y())
            
            tl = self.bbox[TL_IDX]
            br = self.bbox[BR_IDX]
            self.centerPoint = QtCore.QPointF(min((tl.x(), br.x())) + (max((tl.x(), br.x())) - min((tl.x(), br.x())))/2.0, 
                                              min((tl.y(), br.y())) + (max((tl.y(), br.y())) - min((tl.y(), br.y())))/2.0)
            
            if self.drawOverlay(False) :
                self.frameLabel.setOverlay(self.overlayImg)
            
    def drawOverlay(self, doDrawFeats = True, doDrawBBox = True, doDrawCenter = True, drawingSavedBBox=False, handleToDraw = (-1, -1)) :
        if np.any(self.frameImg != None) :
            if self.overlayImg.size() != self.frameImg.size() :
                self.overlayImg = self.overlayImg.scaled(self.frameImg.size())
            
            ## empty image
            self.overlayImg.fill(QtGui.QColor.fromRgb(255, 255, 255, 0))
            
            painter = QtGui.QPainter(self.overlayImg)
            painter.setRenderHints(QtGui.QPainter.Antialiasing)
            
            ## draw handle
            if handleToDraw[1] >= 0 and handleToDraw[1] < 4 :
                painter.setBrush(QtGui.QBrush(QtGui.QColor.fromRgb(225, 225, 225, 128)))
                painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(128, 128, 128, 128), 1, 
                                                  QtCore.Qt.SolidLine, QtCore.Qt.SquareCap, QtCore.Qt.MiterJoin))
                if handleToDraw[0] == HANDLE_MOVE_SEGMENT :
                    handlePoly = self.getSegmentHandle(np.array([self.bbox[handleToDraw[1]].x(),
                                                                 self.bbox[handleToDraw[1]].y()]),
                                                       np.array([self.bbox[np.mod(handleToDraw[1]+1, 4)].x(),
                                                                 self.bbox[np.mod(handleToDraw[1]+1, 4)].y()]))

                    painter.drawPolygon(handlePoly)
                elif handleToDraw[0] == HANDLE_ROTATE_BBOX :
                    painter.drawEllipse(self.bbox[handleToDraw[1]], ROTATE_BBOX_HANDLE_SIZE, ROTATE_BBOX_HANDLE_SIZE)
                painter.setBrush(QtGui.QBrush(QtGui.QColor.fromRgb(0, 0, 0, 0)))
            
            ## draw bbox
            if doDrawBBox :
                bboxPoly = QtGui.QPolygonF()
                for p in self.bbox :
                    bboxPoly.append(p)
#                 for p1, p2 in zip(self.bbox[0:-1], self.bbox[1:]) :
                painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(0, 0, 0), 4, 
                                          QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
                painter.drawPolygon(bboxPoly)
                if drawingSavedBBox :
                    painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(0, 255, 0), 2, 
                                              QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
                else :
                    painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(255, 255, 255), 2, 
                                              QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
                painter.drawPolygon(bboxPoly)
                self.bboxHasBeenDrawn = True
            else :    
                self.bboxHasBeenDrawn = False
            
            ## draw bbox center
            if doDrawCenter :
                painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(0, 0, 0), 4, 
                                          QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
                painter.drawPoint(self.centerPoint)
                if drawingSavedBBox :
                    painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(0, 255, 0), 2, 
                                              QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
                else :
                    painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(255, 255, 255), 2, 
                                              QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
                painter.drawPoint(self.centerPoint)
                
            
            ## draw tracked features
            if doDrawFeats :
                if np.any(self.tracker != None) and self.tracker.has_result :
                    
                    painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(255, 255, 255, 255), 1, 
                                          QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
                    for point in self.tracker.tracked_keypoints[:, 0:-1] :
                        painter.drawEllipse(QtCore.QPointF(point[0], point[1]), 3, 3)
                        
                    painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(0, 0, 255, 255), 1, 
                                          QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
                    for point in self.tracker.votes[:, :2] :
                        painter.drawEllipse(QtCore.QPointF(point[0], point[1]), 3, 3)
                    
                    painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(255, 0, 0, 255), 1, 
                                          QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
                    for point in self.tracker.outliers[:, :2] :
                        painter.drawEllipse(QtCore.QPointF(point[0], point[1]), 3, 3)
            
            painter.end()
                    
            return True
        else :
            return False
    
    def createNewSemanticSequence(self) :
#         la = createNewSemantics(self)
        if len(self.frameLocs) > 0 :
            im = np.ascontiguousarray(np.array(Image.open(self.frameLocs[self.frameIdx]))[:, :, :3])
            frameImg = QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_RGB888);
            seqName, topLeft, size, color, status = createNewSemantics(self, "New Actor Sequence", frameImg)

            if status :
                print seqName, topLeft, size, color, self.frameIdx
                proceed = True
                for i in xrange(len(self.semanticSequences)) :
                    if seqName == self.semanticSequences[i][DICT_SEQUENCE_NAME] :
                        proceed = QtGui.QMessageBox.question(self, 'Override Actor Sequence',
                                        "An actor sequence named \"" + seqName + "\" already exists.\nDo you want to override?", 
                                        QtGui.QMessageBox.Yes | QtGui.QMessageBox.No, QtGui.QMessageBox.No) == QtGui.QMessageBox.Yes
                        if proceed :
                            del self.semanticSequences[i]
                        break
                if proceed :
                    print "adding semantic sequence:", seqName

                    self.semanticSequences.append({
                                                   DICT_SEQUENCE_NAME:seqName,
                                                   DICT_SEQUENCE_LOCATION:self.seqDir+"/semantic_sequence-"+seqName+".npy",
                                                   DICT_ICON_TOP_LEFT:topLeft,
                                                   DICT_ICON_SIZE:size,
                                                   DICT_ICON_FRAME_KEY:self.frameIdx,
                                                   DICT_REPRESENTATIVE_COLOR:color,
                                                   DICT_FRAMES_LOCATIONS:{}
                                                 })

    #                 self.selectedSemSequenceIdx = 
                    currentFrame = self.frameIdx
                    self.setSemanticsToDraw()
                    self.setListOfLoadedSemSequences()
                    self.loadedSequencesListTable.selectRow(len(self.semanticSequences)-1)
                    self.changeSelectedSemSequence(self.loadedSequencesListModel.item(len(self.semanticSequences)-1).index())
                    self.frameIdxSpinBox.setValue(currentFrame)
    #                 self.showFrame(self.frameIdx)
                    sys.stdout.flush()

#                     self.toggleDefineMode()

        self.setFocus()
        
    def setSemanticsToDraw(self) :
        self.semanticsToDraw = []
        if len(self.semanticSequences) > 0  :
            for i in xrange(0, len(self.semanticSequences)):
                if DICT_REPRESENTATIVE_COLOR in self.semanticSequences[i].keys() :
                    col = self.semanticSequences[i][DICT_REPRESENTATIVE_COLOR]
                else :
                    col = np.array([0, 0, 0])
                
                if len(self.semanticSequences[i][DICT_FRAMES_LOCATIONS].keys()) > 0 :
                    self.semanticsToDraw.append({
                                                    DRAW_COLOR:col,
                                                    DRAW_FIRST_FRAME:np.min(self.semanticSequences[i][DICT_FRAMES_LOCATIONS].keys()),
                                                    DRAW_LAST_FRAME:np.max(self.semanticSequences[i][DICT_FRAMES_LOCATIONS].keys())
                                                })
                
        self.frameIdxSlider.setSemanticsToDraw(self.semanticsToDraw, self.numOfFrames)
            
            
    def changeSelectedSemSequence(self, row) :
        print "selecting sequence", row.row()
        if len(self.semanticSequences) > row.row() :
            self.selectedSemSequenceIdx = row.row()
            print "semantic sequence: ", self.semanticSequences[self.selectedSemSequenceIdx][DICT_SEQUENCE_NAME]
            self.semanticsLabel.setSemantics(self.semanticSequences[self.selectedSemSequenceIdx])
            sys.stdout.flush()
            ## go to the first frame used by this semantic sequence
            if len(self.semanticSequences[self.selectedSemSequenceIdx][DICT_FRAMES_LOCATIONS].keys()) > 0 :
                ## go to last if we already at first
                if self.frameIdx == np.min(self.semanticSequences[self.selectedSemSequenceIdx][DICT_FRAMES_LOCATIONS].keys()) :
                    self.frameIdxSpinBox.setValue(np.max(self.semanticSequences[self.selectedSemSequenceIdx][DICT_FRAMES_LOCATIONS].keys()))
                ## go to first
                else :
                    self.frameIdxSpinBox.setValue(np.min(self.semanticSequences[self.selectedSemSequenceIdx][DICT_FRAMES_LOCATIONS].keys()))
            ## go to the first frame in video
            else :
                self.frameIdxSpinBox.setValue(0)
            self.showFrame(self.frameIdx)
            
            self.frameIdxSlider.setSelectedSemantics(self.selectedSemSequenceIdx)
            
            
            ##
            if DO_SAVE_LOGS :
                with open(np.sort(glob.glob("logFiles/log-*"))[-1], "a+") as f :
                    f.write("LOG:DEFINITION:Selecting {0}-&-".format(self.semanticSequences[self.selectedSemSequenceIdx][DICT_SEQUENCE_NAME]) + str(datetime.datetime.now()) +"\n")
        else :
            self.semanticsLabel.setSemantics(None)
            
        self.setFocus()
            
    def loadSemanticSequences(self) :
        try :
            del self.preloadedPatches
            self.preloadedPatches = {}
        except :
            self.preloadedPatches = {}
        
        ## going to first frame of first sequence if there were none before loading
        goToNewSequence = len(self.semanticSequences) == 0
        for index, seq in enumerate(np.sort(glob.glob(self.seqDir+"/semantic_sequence-*.npy"))) :
            with open(seq) as f :
                self.semanticSequences.append(np.load(f).item())
                if (DICT_FRAMES_LOCATIONS in self.semanticSequences[-1].keys() and len(self.semanticSequences[-1][DICT_FRAMES_LOCATIONS].keys()) > 0 and
                    np.max(self.semanticSequences[-1][DICT_FRAMES_LOCATIONS].keys()) == self.numOfFrames) :
                    self.frameIdxSlider.setMaximum(np.max([0, self.numOfFrames-1]))
                    self.frameIdxSpinBox.setRange(0, np.max([0, self.numOfFrames-1]))
                    
            if DICT_PATCHES_LOCATION in self.semanticSequences[-1].keys() :
                with open(self.semanticSequences[-1][DICT_PATCHES_LOCATION]) as f :
                    self.preloadedPatches[index] = np.load(f).item()
                if 'needs_saving' not in self.preloadedPatches[index] :
                    self.preloadedPatches[index]['needs_saving'] = False
            if DICT_FRAME_SEMANTICS in self.semanticSequences[-1].keys() and DICT_NUM_SEMANTICS not in self.semanticSequences[-1].keys() :
                print "DICT_NUM_SEMANTICS not present even though there are", self.semanticSequences[-1][DICT_FRAME_SEMANTICS].shape[1], "semantics in DICT_FRAME_SEMANTICS.",
#                 print "Goto this piece of code and uncomment the saving code"
                self.semanticSequences[-1][DICT_NUM_SEMANTICS] = self.semanticSequences[-1][DICT_FRAME_SEMANTICS].shape[1]
                np.save(self.semanticSequences[-1][DICT_SEQUENCE_LOCATION], self.semanticSequences[-1])
                    
            ## check that they have defined actions and if they do that they have assigned names
            if DICT_NUM_SEMANTICS in self.semanticSequences[-1].keys() :
                tmpDoSave = False
                if DICT_SEMANTICS_NAMES not in self.semanticSequences[-1].keys() :
                    self.semanticSequences[-1][DICT_SEMANTICS_NAMES] = {}
                    tmpDoSave = True
                    
                for semIdx in xrange(self.semanticSequences[-1][DICT_NUM_SEMANTICS]) :
                    if semIdx not in self.semanticSequences[-1][DICT_SEMANTICS_NAMES].keys() :
                        self.semanticSequences[-1][DICT_SEMANTICS_NAMES][semIdx] = "action{0:d}".format(semIdx)
                        print "ACTION NAME", self.semanticSequences[-1][DICT_SEMANTICS_NAMES][semIdx], "added for", self.semanticSequences[-1][DICT_SEQUENCE_NAME]
                        tmpDoSave = True
                    
                numFrames = len(self.semanticSequences[-1][DICT_FRAMES_LOCATIONS].keys())
            #     print numFrames, seqLoc
                if DICT_LABELLED_FRAMES in self.semanticSequences[-1].keys() and DICT_NUM_EXTRA_FRAMES in self.semanticSequences[-1].keys() :
                    for i in xrange(len(self.semanticSequences[-1][DICT_LABELLED_FRAMES])) :
                        listToKeep = []
                        for j in xrange(len(self.semanticSequences[-1][DICT_LABELLED_FRAMES][i])) :
                            if self.semanticSequences[-1][DICT_LABELLED_FRAMES][i][j] >= 0 and self.semanticSequences[-1][DICT_LABELLED_FRAMES][i][j] < numFrames :
                                listToKeep.append(j)
                                
                        if len(listToKeep) != len(self.semanticSequences[-1][DICT_LABELLED_FRAMES][i]) :
                            tmpDoSave = True
            #             print self.semanticSequences[-1][DICT_LABELLED_FRAMES][i], listToKeep,
                        self.semanticSequences[-1][DICT_LABELLED_FRAMES][i] = [self.semanticSequences[-1][DICT_LABELLED_FRAMES][i][j] for j in listToKeep]
                        self.semanticSequences[-1][DICT_NUM_EXTRA_FRAMES][i] = [self.semanticSequences[-1][DICT_NUM_EXTRA_FRAMES][i][j] for j in listToKeep]
            #             print self.semanticSequences[-1][DICT_LABELLED_FRAMES][i], self.semanticSequences[-1][DICT_NUM_EXTRA_FRAMES][i]
                        
                if tmpDoSave :
                    np.save(self.semanticSequences[-1][DICT_SEQUENCE_LOCATION], self.semanticSequences[-1])
                
                
        
        self.setListOfLoadedSemSequences()
        
        if len(self.semanticSequences) > 0 and goToNewSequence :
            self.loadedSequencesListTable.selectRow(0)
            self.changeSelectedSemSequence(self.loadedSequencesListModel.item(0).index())
        else :
            self.semanticsLabel.setSemantics(None)
            
        self.setSemanticsToDraw()

    def setListOfLoadedSemSequences(self) :
        if len(self.semanticSequences) > 0 :
            self.loadedSequencesListModel.setRowCount(len(self.semanticSequences))
            self.delegateList = []
                
            for i in xrange(0, len(self.semanticSequences)):
                self.delegateList.append(ListDelegate())
                self.loadedSequencesListTable.setItemDelegateForRow(i, self.delegateList[-1])
                
                ## set sequence name
                self.loadedSequencesListModel.setItem(i, 0, QtGui.QStandardItem(self.semanticSequences[i][DICT_SEQUENCE_NAME]))
    
                ## set sequence icon
                if (DICT_ICON_TOP_LEFT in self.semanticSequences[i].keys() and
                    DICT_ICON_SIZE in self.semanticSequences[i].keys() and
                    DICT_ICON_FRAME_KEY in self.semanticSequences[i].keys()) :
                    
                    maskDir = self.seqDir+"/"+self.semanticSequences[i][DICT_SEQUENCE_NAME] + "-maskedFlow-blended"
                    ## means I've the icon frame and it's been masked otherwise just load the original frame and use for the icon
                    if (os.path.isdir(maskDir) and
                        self.semanticSequences[i][DICT_ICON_FRAME_KEY] in self.semanticSequences[i][DICT_FRAMES_LOCATIONS].keys()) :

                        frameName = self.semanticSequences[i][DICT_FRAMES_LOCATIONS][self.semanticSequences[i][DICT_ICON_FRAME_KEY]].split(os.sep)[-1]

                        framePatch = np.array(Image.open(maskDir+"/"+frameName))
                        framePatch = framePatch[self.semanticSequences[i][DICT_ICON_TOP_LEFT][0]:self.semanticSequences[i][DICT_ICON_TOP_LEFT][0]+self.semanticSequences[i][DICT_ICON_SIZE],
                                                self.semanticSequences[i][DICT_ICON_TOP_LEFT][1]:self.semanticSequences[i][DICT_ICON_TOP_LEFT][1]+self.semanticSequences[i][DICT_ICON_SIZE], :]

                        bgPatch = self.bgImage[self.semanticSequences[i][DICT_ICON_TOP_LEFT][0]:self.semanticSequences[i][DICT_ICON_TOP_LEFT][0]+self.semanticSequences[i][DICT_ICON_SIZE],
                                               self.semanticSequences[i][DICT_ICON_TOP_LEFT][1]:self.semanticSequences[i][DICT_ICON_TOP_LEFT][1]+self.semanticSequences[i][DICT_ICON_SIZE], :]

                        iconPatch = (framePatch[:, :, :3]*(framePatch[:, :, -1].reshape((framePatch.shape[0], framePatch.shape[1], 1))/255.0) + 
                                     bgPatch[:, :, :3]*(1.0-(framePatch[:, :, -1].reshape((framePatch.shape[0], framePatch.shape[1], 1)))/255.0)).astype(np.uint8)
                        self.framePatch = framePatch

                        self.iconImage = np.ascontiguousarray(cv2.resize(iconPatch, (LIST_SECTION_SIZE, LIST_SECTION_SIZE), interpolation=cv2.INTER_AREA))
                    else :
                        
                        framePatch = np.array(Image.open(self.seqDir+"/frame-{0:05d}.png".format(self.semanticSequences[i][DICT_ICON_FRAME_KEY]+1)))
                        framePatch = framePatch[self.semanticSequences[i][DICT_ICON_TOP_LEFT][0]:self.semanticSequences[i][DICT_ICON_TOP_LEFT][0]+self.semanticSequences[i][DICT_ICON_SIZE],
                                                self.semanticSequences[i][DICT_ICON_TOP_LEFT][1]:self.semanticSequences[i][DICT_ICON_TOP_LEFT][1]+self.semanticSequences[i][DICT_ICON_SIZE], :]
                        
                        self.iconImage = np.ascontiguousarray(cv2.resize(framePatch[:, :, :3], (LIST_SECTION_SIZE, LIST_SECTION_SIZE), interpolation=cv2.INTER_AREA))
                        
                else :
                    self.iconImage = np.ascontiguousarray(self.bgImage[:LIST_SECTION_SIZE, :LIST_SECTION_SIZE, :3])
                
                self.loadedSequencesListTable.itemDelegateForRow(i).setIconImage(self.iconImage)
                
                ## set sequence color
                if DICT_REPRESENTATIVE_COLOR in self.semanticSequences[i].keys() :
                    col = self.semanticSequences[i][DICT_REPRESENTATIVE_COLOR]
                    self.loadedSequencesListTable.itemDelegateForRow(i).setBackgroundColor(QtGui.QColor.fromRgb(col[0], col[1], col[2], 255))
                    
            self.selectedSemSequenceIdx = 0
            
            self.loadedSequencesListTable.setEnabled(True)
        else :
            self.loadedSequencesListModel.setRowCount(1)
#             self.loadedSequencesListModel.setColumnCount(1)
            
            self.delegateList = [ListDelegate()]
            self.loadedSequencesListTable.setItemDelegateForRow(0, self.delegateList[-1])
            self.loadedSequencesListModel.setItem(0, 0, QtGui.QStandardItem("None"))
            self.loadedSequencesListTable.setEnabled(False)
        
    def deleteCurrentSemSequenceFrameBBox(self) :
        if self.selectedSemSequenceIdx < len(self.semanticSequences) and self.selectedSemSequenceIdx >= 0 :
            if DICT_FRAMES_LOCATIONS in self.semanticSequences[self.selectedSemSequenceIdx].keys() and self.frameIdx in self.semanticSequences[self.selectedSemSequenceIdx][DICT_FRAMES_LOCATIONS].keys() :
                del self.semanticSequences[self.selectedSemSequenceIdx][DICT_FRAMES_LOCATIONS][self.frameIdx]
                
            if DICT_BBOXES in self.semanticSequences[self.selectedSemSequenceIdx].keys() and self.frameIdx in self.semanticSequences[self.selectedSemSequenceIdx][DICT_BBOXES].keys() :
                del self.semanticSequences[self.selectedSemSequenceIdx][DICT_BBOXES][self.frameIdx]
                del self.semanticSequences[self.selectedSemSequenceIdx][DICT_BBOX_CENTERS][self.frameIdx]
                del self.semanticSequences[self.selectedSemSequenceIdx][DICT_BBOX_ROTATIONS][self.frameIdx]
            ## refresh current frame so that new bbox gets drawn
            self.showFrame(self.frameIdx)
            
    def setCurrentSemSequenceFrameBBox(self) :
        if self.selectedSemSequenceIdx < len(self.semanticSequences) and self.selectedSemSequenceIdx >= 0:
            ## compute rotation in radians
            rot = np.mod(np.arctan2(-(self.bbox[TL_IDX]-self.bbox[TR_IDX]).y(), 
                                     (self.bbox[TL_IDX]-self.bbox[TR_IDX]).x()),2*np.pi)
            
            if DICT_BBOXES not in self.semanticSequences[self.selectedSemSequenceIdx].keys() :
                self.semanticSequences[self.selectedSemSequenceIdx][DICT_BBOXES] = {}
                self.semanticSequences[self.selectedSemSequenceIdx][DICT_BBOX_CENTERS] = {}
                self.semanticSequences[self.selectedSemSequenceIdx][DICT_BBOX_ROTATIONS] = {}
            
            self.semanticSequences[self.selectedSemSequenceIdx][DICT_BBOXES][self.frameIdx] = np.array([[self.bbox[TL_IDX].x(), self.bbox[TL_IDX].y()], 
                                                                                                [self.bbox[TR_IDX].x(), self.bbox[TR_IDX].y()], 
                                                                                                [self.bbox[BR_IDX].x(), self.bbox[BR_IDX].y()], 
                                                                                                [self.bbox[BL_IDX].x(), self.bbox[BL_IDX].y()]])
            self.semanticSequences[self.selectedSemSequenceIdx][DICT_BBOX_CENTERS][self.frameIdx] = np.array([self.centerPoint.x(), self.centerPoint.y()])
            self.semanticSequences[self.selectedSemSequenceIdx][DICT_BBOX_ROTATIONS][self.frameIdx] = rot
            self.semanticSequences[self.selectedSemSequenceIdx][DICT_FRAMES_LOCATIONS][self.frameIdx] = self.frameLocs[self.frameIdx]
            ## refresh current frame so that new bbox gets drawn
            self.showFrame(self.frameIdx)
        
    def saveSemanticSequences(self) :
        for i, seq in enumerate(self.semanticSequences) :
            np.save(seq[DICT_SEQUENCE_LOCATION], seq)
            if DICT_PATCHES_LOCATION in seq.keys() and i in self.preloadedPatches.keys() :
                if self.preloadedPatches[i]['needs_saving'] :
                    self.preloadedPatches[i]['needs_saving'] = False
                    np.save(seq[DICT_PATCHES_LOCATION], self.preloadedPatches[i])
            print seq[DICT_SEQUENCE_NAME], "saved"
            sys.stdout.flush()
        
    def isInsideBBox(self, point) :
        bboxPoly = QtGui.QPolygonF()
        for p in self.bbox :
            bboxPoly.append(p)
            
        return bboxPoly.containsPoint(point, QtCore.Qt.WindingFill)
            
    def closeEvent(self, event) :
        print "closing"
        sys.stdout.flush()
        self.saveSemanticSequences()
    
    def mousePressed(self, event):
#         print event.pos()
#         sys.stdout.flush()
        if not self.tracking and not self.segmenting :
            if self.semanticsColorButton.isChecked() :
                if event.button() == QtCore.Qt.LeftButton and np.any(self.frameLabel.image != None) :
                    self.setSemanticsColor(QtGui.QColor(self.frameLabel.image.pixel(event.pos())))
            else :
                if self.isModeBBox :
                    ## CHANGING BBOX
                    if event.button() == QtCore.Qt.LeftButton :
                        handleUnderPointer = self.getHandleUnderPointer(event.posF())
                        if not self.settingBBox and handleUnderPointer[0] == HANDLE_MOVE_SEGMENT :
                            self.frameLabel.setCursor(QtCore.Qt.ClosedHandCursor)
                            ## only moving one segment
                            self.movingSegment = True
                            self.segmentToMove = handleUnderPointer[1]
                            self.bboxIsSet = False
                            self.prevPoint = event.posF()
                        elif not self.settingBBox and handleUnderPointer[0] == HANDLE_ROTATE_BBOX :
                            self.frameLabel.setCursor(QtCore.Qt.ClosedHandCursor)
                            if self.bboxIsSet :
                #                 print "rotatingBBox"
                                self.rotatingBBox = True
                                self.cornerToRotate = handleUnderPointer[1]
                                self.bboxIsSet = False
                                self.prevPoint = event.posF()
                        else :
                            if not self.settingBBox and self.isInsideBBox(event.posF()) :
                                self.frameLabel.setCursor(QtCore.Qt.ClosedHandCursor)
                #                 print "movingBBox"
                                self.movingBBox = True
                                self.bboxIsSet = False
                                self.prevPoint = event.posF()
                            else :
                                if not self.settingBBox :
                                    self.bbox[:] = event.posF()
                                    self.settingBBox = True
                                    self.bboxIsSet = False
                                    self.updateBBox()
                                else :
                                    self.bbox[BR_IDX] = event.posF()
                                    self.updateBBox()
                                    self.settingBBox = False
                                    self.bboxIsSet = True
                else :
                    ## SCRIBBLING
                    if event.button() == QtCore.Qt.LeftButton or event.button() == QtCore.Qt.RightButton :
                        self.prevPoint = event.posF()
                        self.scribbling = True          
                
#         sys.stdout.flush()
        
    def getSegmentHandle(self, a, b, handleSize=MOVE_SEGMENT_HANDLE_SIZE) :
        pointDiff = b-a
        normalDir = np.array([-pointDiff[1], pointDiff[0]])
        if np.linalg.norm(normalDir) == 0 or np.linalg.norm(pointDiff) == 0 :
            return None
        normalDir /= np.linalg.norm(normalDir)
        segmentDir = pointDiff/np.linalg.norm(pointDiff)
        
        handleBorders = np.array([a+segmentDir*handleSize*0.1, b-segmentDir*handleSize*0.1])
        segmentHandle = np.array([handleBorders[0]+normalDir*handleSize/2.0, handleBorders[1]+normalDir*handleSize/2.0,
                                  handleBorders[1]-normalDir*handleSize/2.0, handleBorders[0]-normalDir*handleSize/2.0])
        
        handlePoly = QtGui.QPolygonF()
        for p in segmentHandle :
            handlePoly.append(QtCore.QPointF(p[0], p[1]))
        
        return handlePoly
    
    def getHandleUnderPointer(self, pointerPos) :
        for i, j in zip(np.arange(4), np.mod(np.arange(1, 5), 4)) :
            
            handlePoly = self.getSegmentHandle(np.array([self.bbox[j].x(), self.bbox[j].y()]),
                                               np.array([self.bbox[i].x(), self.bbox[i].y()]))
            
            if handlePoly!= None and handlePoly.boundingRect().width() > 10 and handlePoly.boundingRect().height() > 10 :
                if handlePoly.containsPoint(pointerPos, QtCore.Qt.WindingFill) :
                    return (HANDLE_MOVE_SEGMENT, i)
            
            if np.linalg.norm(np.array([pointerPos.x(), pointerPos.y()])-np.array([self.bbox[i].x(), self.bbox[i].y()])) <= ROTATE_BBOX_HANDLE_SIZE :
                return (HANDLE_ROTATE_BBOX, i)
            
        return (-1, -1)
    
    def moveBBoxSegment(self, segmentIdx, eventPoint) :
        nextSegmentLength = np.linalg.norm(np.array([self.bbox[np.mod(segmentIdx+1, 4)].x(), self.bbox[np.mod(segmentIdx+1, 4)].y()])-
                                           np.array([self.bbox[np.mod(segmentIdx+2, 4)].x(), self.bbox[np.mod(segmentIdx+2, 4)].y()]))
        
        prevPos = np.array([self.prevPoint.x(), self.prevPoint.y(), 1])
        eventPos = np.array([eventPoint.x(), eventPoint.y(), 1])
        
        if self.bbox[np.mod(segmentIdx+1, 4)].x() - self.bbox[segmentIdx].x() == 0 :
            projPoints = np.vstack([[prevPos[:-1]], [eventPos[:-1]]])
            projPoints[:, 0] = self.bbox[segmentIdx].x()
        elif self.bbox[np.mod(segmentIdx+1, 4)].y() - self.bbox[segmentIdx].y() == 0 :
            projPoints = np.vstack([[prevPos[:-1]], [eventPos[:-1]]])
            projPoints[:, 1] = self.bbox[segmentIdx].y()
        else :
            m = (self.bbox[np.mod(segmentIdx+1, 4)].y() - self.bbox[segmentIdx].y()) / (self.bbox[np.mod(segmentIdx+1, 4)].x() - self.bbox[segmentIdx].x())
            b = self.bbox[segmentIdx].y() - (m * self.bbox[segmentIdx].x())
            ## project above points onto segment
            projPoints = np.dot(np.vstack([[prevPos], [eventPos]]), np.array([[1, m, -m*b], [m, m**2, b]]).T)/(m**2+1)
        
        deltaPos =  prevPos[:-1] + projPoints[1, :] - projPoints[0, :] - eventPos[:-1]
        
        ## if the signs of normalDir and deltaPos are opposite it means I'm moving the segment towards the center
        a = np.array([self.bbox[segmentIdx].x(), self.bbox[segmentIdx].y()])
        b = np.array([self.bbox[np.mod(segmentIdx+1, 4)].x(), self.bbox[np.mod(segmentIdx+1, 4)].y()])
        pointDiff = b-a
        normalDir = np.array([-pointDiff[1], pointDiff[0]])
        goingTowardsCenter = np.logical_xor(normalDir[0] < 0, deltaPos[0] < 0) or np.logical_xor(normalDir[1] < 0, deltaPos[1] < 0)
        
        if goingTowardsCenter and np.linalg.norm(deltaPos) != 0.0 :
            deltaPos = deltaPos/np.linalg.norm(deltaPos)*np.min([nextSegmentLength-10, np.linalg.norm(deltaPos)])
        
        deltaPos = QtCore.QPointF(deltaPos[0], deltaPos[1])
        ## move bbox
        self.bbox[segmentIdx] -= deltaPos
        self.bbox[np.mod(segmentIdx+1, 4)] -= deltaPos
        ## move center
        self.centerPoint -= deltaPos/2.0
        
    def rotateBBox(self, eventPoint) :
        a = np.array([self.prevPoint.x(), self.prevPoint.y()])
        b = np.array([eventPoint.x(), eventPoint.y()])
        c = np.array([self.centerPoint.x(), self.centerPoint.y()])
        
        deltaR = np.arccos(np.dot(b-c, a-c)/(np.linalg.norm(b-c)*np.linalg.norm(a-c)))
        if np.cross(b-c, a-c) > 0 :
            deltaR = -deltaR
            
        if not np.isnan(deltaR) :
            t = QtGui.QTransform()            
            t.rotateRadians(deltaR, QtCore.Qt.Axis.ZAxis)
            self.bbox = np.array(t.map(self.bbox-self.centerPoint))+self.centerPoint
                
    def mouseMoved(self, event):
        if not self.tracking and not self.segmenting :
            if self.isModeBBox :
                ## CHANGING BBOX
                if self.bboxIsSet and self.bboxHasBeenDrawn :
                    handleUnderPointer = self.getHandleUnderPointer(event.posF())
                    if self.drawOverlay(False, drawingSavedBBox=self.bboxChangedAndSaved, handleToDraw = handleUnderPointer) :
                        self.frameLabel.setOverlay(self.overlayImg)
                    if handleUnderPointer[0] != -1 or self.isInsideBBox(event.posF()) :
                        self.frameLabel.setCursor(QtCore.Qt.OpenHandCursor)
                    else :
                        self.frameLabel.setCursor(QtCore.Qt.ArrowCursor)

                if self.movingSegment :
                    self.frameLabel.setCursor(QtCore.Qt.ClosedHandCursor)
                    moveDir = np.array([event.posF().x(), event.posF().y()]) - np.array([self.prevPoint.x(), self.prevPoint.y()])
                    self.moveBBoxSegment(self.segmentToMove, event.posF())
                    if self.drawOverlay(False, handleToDraw = (HANDLE_MOVE_SEGMENT, self.segmentToMove)) :
                        self.frameLabel.setOverlay(self.overlayImg)
                    self.prevPoint = event.posF()
                    self.bboxChangedAndSaved = False
                elif self.settingBBox :
                    self.frameLabel.setCursor(QtCore.Qt.ArrowCursor)
                    self.bbox[BR_IDX] = event.posF()
                    self.updateBBox()
                    self.bboxChangedAndSaved = False
                elif self.movingBBox and np.any(self.prevPoint != None) :
                    self.frameLabel.setCursor(QtCore.Qt.ClosedHandCursor)
                    self.bbox = self.bbox - self.prevPoint + event.posF()
                    self.centerPoint = self.centerPoint - self.prevPoint + event.posF()
                    self.prevPoint = event.posF()
                    if self.drawOverlay(False) :
                        self.frameLabel.setOverlay(self.overlayImg)
                    self.bboxChangedAndSaved = False
                elif self.rotatingBBox and np.any(self.prevPoint != None) :
                    self.frameLabel.setCursor(QtCore.Qt.ClosedHandCursor)
                    self.rotateBBox(event.posF())
                    self.prevPoint = event.posF()
                    if self.drawOverlay(False, handleToDraw = (HANDLE_ROTATE_BBOX, self.cornerToRotate)) :
                        self.frameLabel.setOverlay(self.overlayImg)
                    self.bboxChangedAndSaved = False
            else :
                ## SCRIBBLING
                if ((event.buttons() & QtCore.Qt.LeftButton) or (event.buttons() & QtCore.Qt.RightButton)) and self.scribbling:

                    if event.buttons() & QtCore.Qt.LeftButton :
                        ## foreground
                        penColor = QtGui.QColor.fromRgb(0, 255, 0)
                    else :
                        ## background
                        penColor = QtGui.QColor.fromRgb(0, 0, 255)

                    if QtGui.QApplication.keyboardModifiers() & QtCore.Qt.ShiftModifier :
                        ## delete
                        penColor = QtGui.QColor.fromRgb(255, 255, 255)

                    self.drawLineTo(event.posF(), penColor)

            
    def mouseReleased(self, event):
        if not self.tracking and not self.segmenting :
            if self.isModeBBox :
                ## CHANGING BBOX
                handleUnderPointer = self.getHandleUnderPointer(event.posF())
                if handleUnderPointer[0] != -1 or self.isInsideBBox(event.posF()) :
                    self.frameLabel.setCursor(QtCore.Qt.OpenHandCursor)
                else :
                    self.frameLabel.setCursor(QtCore.Qt.ArrowCursor)
                    
                if self.movingSegment and np.any(self.prevPoint != None) :
                    moveDir = np.array([event.posF().x(), event.posF().y()]) - np.array([self.prevPoint.x(), self.prevPoint.y()])
                    self.moveBBoxSegment(self.segmentToMove, event.posF())
                    self.prevPoint = event.posF()
                    if self.drawOverlay(False, handleToDraw = (HANDLE_MOVE_SEGMENT, self.segmentToMove)) :
                        self.frameLabel.setOverlay(self.overlayImg)
                    self.movingSegment = False
                    self.bboxIsSet = True
                elif self.movingBBox and np.any(self.prevPoint != None) :
                    self.bbox = self.bbox - self.prevPoint + event.posF()
                    self.prevPoint = event.posF()
                    if self.drawOverlay(False) :
                        self.frameLabel.setOverlay(self.overlayImg)
                    self.movingBBox = False
                    self.bboxIsSet = True
                elif self.rotatingBBox and np.any(self.prevPoint != None) :
                    self.rotateBBox(event.posF())
                    self.prevPoint = event.posF()
                    if self.drawOverlay(False, handleToDraw = (HANDLE_ROTATE_BBOX, self.cornerToRotate)) :
                        self.frameLabel.setOverlay(self.overlayImg)
                    self.rotatingBBox = False
                    self.bboxIsSet = True
            else :
                ## SCRIBBLING
                if DICT_MASK_LOCATION in self.semanticSequences[self.selectedSemSequenceIdx].keys() :
                    if ((event.buttons() & QtCore.Qt.LeftButton) or (event.buttons() & QtCore.Qt.RightButton)) and self.scribbling:

                        if event.buttons() & QtCore.Qt.LeftButton :
                            ## foreground
                            penColor = QtGui.QColor.fromRgb(0, 255, 0)
                        else :
                            ## background
                            penColor = QtGui.QColor.fromRgb(0, 0, 255)

                        if QtGui.QApplication.keyboardModifiers() & QtCore.Qt.ShiftModifier :
                            ## delete
                            penColor = QtGui.QColor.fromRgb(255, 255, 255)

                        self.drawLineTo(event.posF(), penColor)

                        self.scribbling = False

                    if self.frameIdx >= 0 and self.frameIdx < self.numOfFrames :
                        frameName = self.frameLocs[self.frameIdx].split(os.sep)[-1]
                        self.scribble.save(self.semanticSequences[self.selectedSemSequenceIdx][DICT_MASK_LOCATION]+"scribble-" + frameName)

            
    def drawLineTo(self, endPoint, penColor):
        painter = QtGui.QPainter(self.scribble)
            
        painter.setPen(QtGui.QPen(penColor, self.brushSize,
                QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
        painter.drawLine(self.prevPoint, endPoint)
 
        self.prevPoint = QtCore.QPointF(endPoint)
        
        self.frameLabel.setScribbleImage(self.scribble)
    
    def setOriginalImageOpacity(self, opacity) :
        self.frameLabel.setImageOpacity(opacity/100.0)
        
    def setScribbleOpacity(self, opacity) :
        self.frameLabel.setScribbleOpacity(opacity/100.0)
        
    def setBrushSizeAndCursor(self, brushSize) :
        self.brushSize = brushSize
        if not self.isModeBBox :
            tmp = QtGui.QImage(QtCore.QSize(self.brushSize+1, self.brushSize+1), QtGui.QImage.Format_ARGB32)
            tmp.fill(QtGui.QColor.fromRgb(0, 0, 0, 0))
            painter = QtGui.QPainter(tmp)
            painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(255, 255, 255, 255), 1, 
                                  QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
            painter.setBrush(QtGui.QBrush(QtGui.QColor.fromRgb(0, 0, 0, 0)))
            painter.drawEllipse(QtCore.QRect(0, 0, self.brushSize, self.brushSize))
            painter.end()

            self.frameLabel.setCursor(QtGui.QCursor(QtGui.QPixmap.fromImage(tmp)))
            
    def toggleDefineMode(self) :
        
        self.isModeBBox = not self.isModeBBox
        self.frameLabel.setModeBBox(self.isModeBBox)
        if self.isModeBBox :                    
            handleUnderPointer = self.getHandleUnderPointer(self.mapFromGlobal(QtGui.QCursor.pos())-self.frameLabel.pos())
            if self.drawOverlay(False, handleToDraw = handleUnderPointer) :
                self.frameLabel.setOverlay(self.overlayImg)
            if handleUnderPointer[0] != -1 :
                self.frameLabel.setCursor(QtCore.Qt.PointingHandCursor)
            else :
                self.frameLabel.setCursor(QtCore.Qt.ArrowCursor)

            self.frameLabelWidget.setStyleSheet("QGroupBox { background-color:rgba(255, 0, 0, 28); }")

            ##
            if DO_SAVE_LOGS :
                with open(np.sort(glob.glob("logFiles/log-*"))[-1], "a+") as f :
                    f.write("LOG:DEFINITION:Start Tracking-&-" + str(datetime.datetime.now()) +"\n")
        else :                    
            self.setBrushSizeAndCursor(self.brushSize)
            if self.drawOverlay(False) :
                self.frameLabel.setOverlay(self.overlayImg)

            self.frameLabelWidget.setStyleSheet("QGroupBox { background-color:rgba(0, 0, 255, 28); }")

            ##
            if DO_SAVE_LOGS :
                with open(np.sort(glob.glob("logFiles/log-*"))[-1], "a+") as f :
                    f.write("LOG:DEFINITION:Start Segmenting-&-" + str(datetime.datetime.now()) +"\n")
            
    def computeLabelPropagation(self) :
        self.setFocus()
        
        if self.selectedSemSequenceIdx >= len(self.semanticSequences) or self.selectedSemSequenceIdx < 0 :
            return
        
        if DICT_DISTANCE_MATRIX_LOCATION not in self.semanticSequences[self.selectedSemSequenceIdx] :
            QtGui.QMessageBox.warning(self, "Distance Matrix Not Found", ("Cannot refresh semantic labels as the distance matrix has not been "+
                                                                          "computed for this semantic sequence. However, the labelled frame has been "+
                                                                          "saved and will be used as soon as a distance matrix is available."))
        else :
            if DICT_LABELLED_FRAMES in self.semanticSequences[self.selectedSemSequenceIdx].keys() :
            
                if len(self.semanticSequences[self.selectedSemSequenceIdx][DICT_LABELLED_FRAMES]) > 1 :
                    try :
#                         delta = 0
#                         if DICT_FRAME_SEMANTICS in self.semanticSequences[self.selectedSemSequenceIdx].keys() and DICT_FRAME_COMPATIBILITY_LABELS in self.semanticSequences[self.selectedSemSequenceIdx].keys() :
#                             if self.semanticSequences[self.selectedSemSequenceIdx][DICT_FRAME_SEMANTICS].shape[1] > self.semanticSequences[self.selectedSemSequenceIdx][DICT_FRAME_COMPATIBILITY_LABELS].shape[1] :
#                                 raise Exception("Must have fewer (or equal number of) semantic classes than compatibility classes!")
#                             else :
#                                 delta = self.semanticSequences[self.selectedSemSequenceIdx][DICT_FRAME_SEMANTICS].shape[1] - self.semanticSequences[self.selectedSemSequenceIdx][DICT_FRAME_COMPATIBILITY_LABELS].shape[1]

#                         if delta < 0 :
#                             self.semanticSequences[self.selectedSemSequenceIdx][DICT_FRAME_SEMANTICS] = propagateLabels(np.load(self.semanticSequences[self.selectedSemSequenceIdx][DICT_DISTANCE_MATRIX_LOCATION]),
#                                                                                                                         self.semanticSequences[self.selectedSemSequenceIdx][DICT_LABELLED_FRAMES][:delta],
#                                                                                                                         self.semanticSequences[self.selectedSemSequenceIdx][DICT_NUM_EXTRA_FRAMES][:delta], True,
#                                                                                                                         self.semanticsSigmaSpinBox.value()/100.0)
#                         else :
#                             self.semanticSequences[self.selectedSemSequenceIdx][DICT_FRAME_SEMANTICS] = propagateLabels(np.load(self.semanticSequences[self.selectedSemSequenceIdx][DICT_DISTANCE_MATRIX_LOCATION]),
#                                                                                                                         self.semanticSequences[self.selectedSemSequenceIdx][DICT_LABELLED_FRAMES],
#                                                                                                                         self.semanticSequences[self.selectedSemSequenceIdx][DICT_NUM_EXTRA_FRAMES], True,
#                                                                                                                         self.semanticsSigmaSpinBox.value()/100.0)

                        numSemanticClasses = 0
#                         if DICT_FRAME_SEMANTICS in self.semanticSequences[self.selectedSemSequenceIdx].keys() :
#                             numSemanticClasses = self.semanticSequences[self.selectedSemSequenceIdx][DICT_FRAME_SEMANTICS].shape[1]
                        if DICT_NUM_SEMANTICS in self.semanticSequences[self.selectedSemSequenceIdx].keys() :
                            numSemanticClasses = self.semanticSequences[self.selectedSemSequenceIdx][DICT_NUM_SEMANTICS]
                
                        self.semanticSequences[self.selectedSemSequenceIdx][DICT_FRAME_SEMANTICS] = propagateLabels(np.load(self.semanticSequences[self.selectedSemSequenceIdx][DICT_DISTANCE_MATRIX_LOCATION]),
                                                                                                                    self.semanticSequences[self.selectedSemSequenceIdx][DICT_LABELLED_FRAMES][:numSemanticClasses],
                                                                                                                    self.semanticSequences[self.selectedSemSequenceIdx][DICT_NUM_EXTRA_FRAMES][:numSemanticClasses], True,
                                                                                                                    self.semanticsSigmaSpinBox.value()/100.0)
                
                
                        self.semanticsLabel.setSemantics(self.semanticSequences[self.selectedSemSequenceIdx])
                    except np.linalg.LinAlgError as e :
                        QtGui.QMessageBox.critical(self, "Linear Algebra Error", ("A Linear Algebra error has been caught: \""+np.string_(e)+
                                                                                  "\". This is often due to low sigma value for label propagation."))
                    except Exception as e :
                        QtGui.QMessageBox.critical(self, "Exception", ("An exception has been caught: \""+np.string_(e)+"\""))
                elif len(self.semanticSequences[self.selectedSemSequenceIdx][DICT_LABELLED_FRAMES]) == 1 :
                    self.semanticSequences[self.selectedSemSequenceIdx][DICT_FRAME_SEMANTICS] = np.ones([len(self.semanticSequences[self.selectedSemSequenceIdx][DICT_FRAMES_LOCATIONS]), 1])
                    self.semanticsLabel.setSemantics(self.semanticSequences[self.selectedSemSequenceIdx])
                    
                self.showFrame(self.frameIdx)
            
    def keyPressEvent(self, e) :
        if e.key() == e.key() >= QtCore.Qt.Key_0 and e.key() <= QtCore.Qt.Key_9 :
            pressedNum = np.mod(e.key()-int(QtCore.Qt.Key_0), int(QtCore.Qt.Key_9))
            if not self.tracking and not self.segmenting :
                startFrame = np.min(self.semanticSequences[self.selectedSemSequenceIdx][DICT_FRAMES_LOCATIONS].keys())
                labelledFrame = self.frameIdx-startFrame
                doRecomputeSemantics = False
                if labelledFrame >= 0 and labelledFrame < len(self.semanticSequences[self.selectedSemSequenceIdx][DICT_FRAMES_LOCATIONS]) :
                    proceed = True
                    ## check if I already labelled the current frame and if the user says so delete it from the other classes
                    if DICT_LABELLED_FRAMES in self.semanticSequences[self.selectedSemSequenceIdx].keys() :
                        for classIdx in xrange(len(self.semanticSequences[self.selectedSemSequenceIdx][DICT_LABELLED_FRAMES])) :
                            classLabelledFrames = np.array(self.semanticSequences[self.selectedSemSequenceIdx][DICT_LABELLED_FRAMES][classIdx])
                            classNumExtraFrames = np.array(self.semanticSequences[self.selectedSemSequenceIdx][DICT_NUM_EXTRA_FRAMES][classIdx])
                            targetFrames = np.arange(labelledFrame-self.numExtraFramesSpinBox.value()/2, labelledFrame+self.numExtraFramesSpinBox.value()/2+1).reshape((1, self.numExtraFramesSpinBox.value()+1))
                            found = np.any(np.abs(targetFrames - classLabelledFrames.reshape((len(classLabelledFrames), 1))) <= classNumExtraFrames.reshape((len(classNumExtraFrames), 1))/2, axis=1)
#                             found = np.abs(labelledFrame-classLabelledFrames) <= classNumExtraFrames/2
                            if np.any(found) :
                                classTypeName = "<u>compatibility</u>"
                                if classIdx < self.semanticSequences[self.selectedSemSequenceIdx][DICT_NUM_SEMANTICS] :
                                    classTypeName = "<u>semantic</u>"
                                    
                                if np.all(found) :
                                    text = "<p align='center'>The current frame has previously been labelled as "+classTypeName+" class {0}. "
                                    text += "If it is overwritten, class {0} will have no remaining examples.<br>Do you want to proceed?</p>"
                                    text = text.format(classIdx)
                                else :
                                    text = "<p align='center'>The current frame has previously been labelled as "+classTypeName+" class {0}.<br>Do you want to override?</p>".format(classIdx)
                                proceed = QtGui.QMessageBox.question(self, 'Frame already labelled', text, 
                                                                     QtGui.QMessageBox.Yes | QtGui.QMessageBox.No, QtGui.QMessageBox.No) == QtGui.QMessageBox.Yes
                                if proceed :
                                    self.semanticSequences[self.selectedSemSequenceIdx][DICT_LABELLED_FRAMES][classIdx] = [x for i, x in enumerate(classLabelledFrames) if not found[i]]
                                    self.semanticSequences[self.selectedSemSequenceIdx][DICT_NUM_EXTRA_FRAMES][classIdx] = [x for i, x in enumerate(classNumExtraFrames) if not found[i]]
            
                    if proceed :
                        numSemanticClasses = 0
#                         if DICT_FRAME_SEMANTICS in self.semanticSequences[self.selectedSemSequenceIdx].keys() :
#                             numSemanticClasses = self.semanticSequences[self.selectedSemSequenceIdx][DICT_FRAME_SEMANTICS].shape[1]
                        if DICT_NUM_SEMANTICS in self.semanticSequences[self.selectedSemSequenceIdx].keys() :
                            numSemanticClasses = self.semanticSequences[self.selectedSemSequenceIdx][DICT_NUM_SEMANTICS]
                        
                        ## add new example to existing class
                        if pressedNum >= 0 and pressedNum < numSemanticClasses : #len(self.semanticSequences[self.selectedSemSequenceIdx][DICT_LABELLED_FRAMES]) :
                            if DICT_LABELLED_FRAMES not in self.semanticSequences[self.selectedSemSequenceIdx].keys() :
                                self.semanticSequences[self.selectedSemSequenceIdx][DICT_LABELLED_FRAMES] = []
                                self.semanticSequences[self.selectedSemSequenceIdx][DICT_NUM_EXTRA_FRAMES] = []
                                
                            self.semanticSequences[self.selectedSemSequenceIdx][DICT_LABELLED_FRAMES][pressedNum].append(labelledFrame)
                            self.semanticSequences[self.selectedSemSequenceIdx][DICT_NUM_EXTRA_FRAMES][pressedNum].append(self.numExtraFramesSpinBox.value())
                            doRecomputeSemantics = True
                        ## add new class
                        elif pressedNum >= 0 and pressedNum == numSemanticClasses : #len(self.semanticSequences[self.selectedSemSequenceIdx][DICT_LABELLED_FRAMES]) :
                            if DICT_LABELLED_FRAMES not in self.semanticSequences[self.selectedSemSequenceIdx].keys() :
                                self.semanticSequences[self.selectedSemSequenceIdx][DICT_LABELLED_FRAMES] = []
                                self.semanticSequences[self.selectedSemSequenceIdx][DICT_NUM_EXTRA_FRAMES] = []
#                             insertLoc = 0
#                             if DICT_FRAME_SEMANTICS in self.semanticSequences[self.selectedSemSequenceIdx].keys() :
#                                 insertLoc = self.semanticSequences[self.selectedSemSequenceIdx][DICT_FRAME_SEMANTICS].shape[1]
                            proceed = True
                            if numSemanticClasses != len(self.semanticSequences[self.selectedSemSequenceIdx][DICT_LABELLED_FRAMES]) :
                                proceed = QtGui.QMessageBox.question(self, 'Conflicts previously marked',
                                                    "Some conflicts have been marked for this sequence.\nBy adding a new semantic class, these conflicts will be lost!", 
                                                    QtGui.QMessageBox.Apply | QtGui.QMessageBox.Cancel, QtGui.QMessageBox.Cancel) == QtGui.QMessageBox.Apply
                            if proceed :
                                if numSemanticClasses != len(self.semanticSequences[self.selectedSemSequenceIdx][DICT_LABELLED_FRAMES]) :
                                    self.semanticSequences[self.selectedSemSequenceIdx][DICT_LABELLED_FRAMES] = self.semanticSequences[self.selectedSemSequenceIdx][DICT_LABELLED_FRAMES][:numSemanticClasses]
                                    self.semanticSequences[self.selectedSemSequenceIdx][DICT_NUM_EXTRA_FRAMES] = self.semanticSequences[self.selectedSemSequenceIdx][DICT_NUM_EXTRA_FRAMES][:numSemanticClasses]
                                    
                                    ## delete conflicts to current sequence
                                    for key in self.semanticSequences[self.selectedSemSequenceIdx][DICT_CONFLICTING_SEQUENCES] :
                                        seq = np.load(key).item()
                                        del seq[DICT_CONFLICTING_SEQUENCES][self.semanticSequences[self.selectedSemSequenceIdx][DICT_SEQUENCE_LOCATION]]
                                        np.save(seq[DICT_SEQUENCE_LOCATION], seq)
                                        for idx, loadedSeq in enumerate(self.semanticSequences) :
                                            if loadedSeq[DICT_SEQUENCE_LOCATION] == seq[DICT_SEQUENCE_LOCATION] :
                                                self.semanticSequences[idx] = np.load(key).item()
                                                
                                    ## delete conflicts from current sequence
                                    del self.semanticSequences[self.selectedSemSequenceIdx][DICT_CONFLICTING_SEQUENCES]
                                    
                                self.semanticSequences[self.selectedSemSequenceIdx][DICT_LABELLED_FRAMES].append([labelledFrame])
                                self.semanticSequences[self.selectedSemSequenceIdx][DICT_NUM_EXTRA_FRAMES].append([self.numExtraFramesSpinBox.value()])
                                
                                if DICT_NUM_SEMANTICS not in self.semanticSequences[self.selectedSemSequenceIdx].keys() :
                                    self.semanticSequences[self.selectedSemSequenceIdx][DICT_NUM_SEMANTICS] = 0
                                self.semanticSequences[self.selectedSemSequenceIdx][DICT_NUM_SEMANTICS] += 1
                                
                                doRecomputeSemantics = True
                                
                                semActionIdx = self.semanticSequences[self.selectedSemSequenceIdx][DICT_NUM_SEMANTICS]-1
                                if DICT_SEMANTICS_NAMES not in self.semanticSequences[self.selectedSemSequenceIdx].keys() :
                                    self.semanticSequences[self.selectedSemSequenceIdx][DICT_SEMANTICS_NAMES] = {}
                                self.semanticSequences[self.selectedSemSequenceIdx][DICT_SEMANTICS_NAMES][semActionIdx] = "action{0:d}".format(semActionIdx)
                                self.nameSemanticAction(self.selectedSemSequenceIdx, semActionIdx)
                                    
                        ## don't do anything as a number too large was pressed
                        else :
                            print "PRESSED INVALID NUMBER", pressedNum
                else :
                    print "LABELLED FRAME NOT PART OF SEQUENCE", labelledFrame
                    
                if DICT_LABELLED_FRAMES in self.semanticSequences[self.selectedSemSequenceIdx].keys() :
                    print "LABELLED FRAMES:", self.semanticSequences[self.selectedSemSequenceIdx][DICT_LABELLED_FRAMES]
                
                if doRecomputeSemantics :
                    self.computeLabelPropagation()
                            
        elif e.key() == QtCore.Qt.Key_Minus :
            ## COPIED FROM ABOVE
            startFrame = np.min(self.semanticSequences[self.selectedSemSequenceIdx][DICT_FRAMES_LOCATIONS].keys())
            labelledFrame = self.frameIdx-startFrame
            if labelledFrame >= 0 and labelledFrame < len(self.semanticSequences[self.selectedSemSequenceIdx][DICT_FRAMES_LOCATIONS]) :
                proceed = True
                ## check if I already labelled the current frame and if the user says so delete it from the other classes
                if DICT_LABELLED_FRAMES in self.semanticSequences[self.selectedSemSequenceIdx].keys() :
                    for classIdx in xrange(len(self.semanticSequences[self.selectedSemSequenceIdx][DICT_LABELLED_FRAMES])) :
                        classLabelledFrames = np.array(self.semanticSequences[self.selectedSemSequenceIdx][DICT_LABELLED_FRAMES][classIdx])
                        classNumExtraFrames = np.array(self.semanticSequences[self.selectedSemSequenceIdx][DICT_NUM_EXTRA_FRAMES][classIdx])
                        targetFrames = np.arange(labelledFrame-self.numExtraFramesSpinBox.value()/2, labelledFrame+self.numExtraFramesSpinBox.value()/2+1).reshape((1, self.numExtraFramesSpinBox.value()+1))
                        found = np.any(np.abs(targetFrames - classLabelledFrames.reshape((len(classLabelledFrames), 1))) <= classNumExtraFrames.reshape((len(classNumExtraFrames), 1))/2, axis=1)
        #                             found = np.abs(labelledFrame-classLabelledFrames) <= classNumExtraFrames/2
                        if np.any(found) :
                            classTypeName = "<u>compatibility</u>"
                            if classIdx < self.semanticSequences[self.selectedSemSequenceIdx][DICT_NUM_SEMANTICS] :
                                classTypeName = "<u>semantic</u>"

                            if np.all(found) :
                                text = "<p align='center'>The current frame has previously been labelled as "+classTypeName+" class {0}. "
                                text += "If it is deleted, class {0} will have no remaining examples.<br>Do you want to proceed?</p>"
                                text = text.format(classIdx)
                            else :
                                text = "<p align='center'>The current frame has previously been labelled as "+classTypeName+" class {0}.<br>Do you want to delete?</p>".format(classIdx)
                            proceed = QtGui.QMessageBox.question(self, 'Frame already labelled', text, 
                                                                 QtGui.QMessageBox.Yes | QtGui.QMessageBox.No, QtGui.QMessageBox.No) == QtGui.QMessageBox.Yes
                            if proceed :
                                self.semanticSequences[self.selectedSemSequenceIdx][DICT_LABELLED_FRAMES][classIdx] = [x for i, x in enumerate(classLabelledFrames) if not found[i]]
                                self.semanticSequences[self.selectedSemSequenceIdx][DICT_NUM_EXTRA_FRAMES][classIdx] = [x for i, x in enumerate(classNumExtraFrames) if not found[i]]
                                self.computeLabelPropagation()
                                
        elif e.key() == QtCore.Qt.Key_M :
            self.toggleDefineMode()
                
        elif self.bboxIsSet and e.key() == QtCore.Qt.Key_Return and e.modifiers() & QtCore.Qt.Modifier.CTRL : ## Track forward
            self.tracking = True
            try :
                self.initTracker()
                self.trackInVideo(True)
            except TrackerError as e :
                QtGui.QMessageBox.critical(self, "Tracker Failed", ("The tracker has failed to produce a result"))
            except Exception as e :
                QtGui.QMessageBox.critical(self, "Tracker Failed", (str(e)))
#             print "tracking forward", self.bbox, self.rotation
            self.setSemanticsToDraw()
            self.tracking = False
        elif self.bboxIsSet and e.key() == QtCore.Qt.Key_Backspace and e.modifiers() & QtCore.Qt.Modifier.CTRL : ## Track backward
            self.tracking = True
            try :
                self.initTracker()
                self.trackInVideo(False)
            except TrackerError as e :
                QtGui.QMessageBox.critical(self, "Tracker Failed", ("The tracker has failed to produce a result"))
            except Exception as e :
                QtGui.QMessageBox.critical(self, "Tracker Failed", (str(e)))
#             print "tracking backward", self.bbox, self.rotation
            self.setSemanticsToDraw()
            self.tracking = False
#         elif e.key() == QtCore.Qt.Key_Space : ## stop tracking or segmenting
#             if self.tracking == True :
#                 self.setSemanticsToDraw()
#                 self.tracking = False
#             if self.segmenting == True :
#                 self.segmenting = False
        elif e.key() == QtCore.Qt.Key_Right :
            self.frameIdxSpinBox.setValue(self.frameIdx+1)
            if e.modifiers() & QtCore.Qt.Modifier.CTRL and self.bboxIsSet and np.any(self.tracker != None) :
                self.tracking = True
                try :
                    self.trackInFrame()
                except TrackerError as e :
                    QtGui.QMessageBox.critical(self, "Tracker Failed", ("The tracker has failed to produce a result"))
                except Exception as e :
                    QtGui.QMessageBox.critical(self, "Tracker Failed", (str(e)))

                self.setSemanticsToDraw()
                self.tracking = False
        elif e.key() == QtCore.Qt.Key_Left :
            self.frameIdxSpinBox.setValue(self.frameIdx-1)
            if e.modifiers() & QtCore.Qt.Modifier.CTRL and self.bboxIsSet and np.any(self.tracker != None) :
                self.tracking = True
                try :
                    self.trackInFrame()
                except TrackerError as e :
                    QtGui.QMessageBox.critical(self, "Tracker Failed", ("The tracker has failed to produce a result"))
                except Exception as e :
                    QtGui.QMessageBox.critical(self, "Tracker Failed", (e))

                self.setSemanticsToDraw()
                self.tracking = False
        elif e.key() == QtCore.Qt.Key_Delete :
            self.deleteCurrentSemSequenceFrameBBox()
            self.setSemanticsToDraw()
        elif e.key() == QtCore.Qt.Key_Enter :
            self.setCurrentSemSequenceFrameBBox()
            self.setSemanticsToDraw()
        elif e.key() == QtCore.Qt.Key_C and e.modifiers() & QtCore.Qt.Modifier.CTRL :
#             print "copying bbox"
            self.copiedBBox[TL_IDX].setX(self.bbox[TL_IDX].x())
            self.copiedBBox[TL_IDX].setY(self.bbox[TL_IDX].y())
            self.copiedBBox[TR_IDX].setX(self.bbox[TR_IDX].x())
            self.copiedBBox[TR_IDX].setY(self.bbox[TR_IDX].y())
            self.copiedBBox[BR_IDX].setX(self.bbox[BR_IDX].x())
            self.copiedBBox[BR_IDX].setY(self.bbox[BR_IDX].y())
            self.copiedBBox[BL_IDX].setX(self.bbox[BL_IDX].x())
            self.copiedBBox[BL_IDX].setY(self.bbox[BL_IDX].y())
            
            self.copiedCenter.setX(self.centerPoint.x())
            self.copiedCenter.setY(self.centerPoint.y())
        elif e.key() == QtCore.Qt.Key_V and e.modifiers() & QtCore.Qt.Modifier.CTRL :
            if np.any(self.copiedBBox != None) and np.any(self.bbox != None) :
#                 print "pasting bbox"
                self.bbox[TL_IDX].setX(self.copiedBBox[TL_IDX].x())
                self.bbox[TL_IDX].setY(self.copiedBBox[TL_IDX].y())
                self.bbox[TR_IDX].setX(self.copiedBBox[TR_IDX].x())
                self.bbox[TR_IDX].setY(self.copiedBBox[TR_IDX].y())
                self.bbox[BR_IDX].setX(self.copiedBBox[BR_IDX].x())
                self.bbox[BR_IDX].setY(self.copiedBBox[BR_IDX].y())
                self.bbox[BL_IDX].setX(self.copiedBBox[BL_IDX].x())
                self.bbox[BL_IDX].setY(self.copiedBBox[BL_IDX].y())
                
                self.centerPoint.setX(self.copiedCenter.x())
                self.centerPoint.setY(self.copiedCenter.y())
                
                self.bboxIsSet = True
                self.bboxChangedAndSaved = False
                if self.drawOverlay(False) :
                    self.frameLabel.setOverlay(self.overlayImg)
                self.doShowBBox = True
        elif e.key() == QtCore.Qt.Key_S and e.modifiers() & QtCore.Qt.Modifier.CTRL :
            self.saveSemanticSequences()
        elif e.key() == QtCore.Qt.Key_Escape :
            if self.tracking == True :
                self.setSemanticsToDraw()
                self.tracking = False
            if self.segmenting == True :
                self.segmenting = False
            if self.settingBBox :
                if (self.selectedSemSequenceIdx >= 0 and self.selectedSemSequenceIdx < len(self.semanticSequences) and
                    DICT_BBOXES in self.semanticSequences[self.selectedSemSequenceIdx].keys() and
                    self.frameIdx in self.semanticSequences[self.selectedSemSequenceIdx][DICT_BBOXES].keys()) :
                    
                    self.bbox[TL_IDX].setX(self.semanticSequences[self.selectedSemSequenceIdx][DICT_BBOXES][self.frameIdx][TL_IDX][0])
                    self.bbox[TL_IDX].setY(self.semanticSequences[self.selectedSemSequenceIdx][DICT_BBOXES][self.frameIdx][TL_IDX][1])
                    self.bbox[TR_IDX].setX(self.semanticSequences[self.selectedSemSequenceIdx][DICT_BBOXES][self.frameIdx][TR_IDX][0])
                    self.bbox[TR_IDX].setY(self.semanticSequences[self.selectedSemSequenceIdx][DICT_BBOXES][self.frameIdx][TR_IDX][1])
                    self.bbox[BR_IDX].setX(self.semanticSequences[self.selectedSemSequenceIdx][DICT_BBOXES][self.frameIdx][BR_IDX][0])
                    self.bbox[BR_IDX].setY(self.semanticSequences[self.selectedSemSequenceIdx][DICT_BBOXES][self.frameIdx][BR_IDX][1])
                    self.bbox[BL_IDX].setX(self.semanticSequences[self.selectedSemSequenceIdx][DICT_BBOXES][self.frameIdx][BL_IDX][0])
                    self.bbox[BL_IDX].setY(self.semanticSequences[self.selectedSemSequenceIdx][DICT_BBOXES][self.frameIdx][BL_IDX][1])

                    self.centerPoint.setX(self.semanticSequences[self.selectedSemSequenceIdx][DICT_BBOX_CENTERS][self.frameIdx][0])
                    self.centerPoint.setY(self.semanticSequences[self.selectedSemSequenceIdx][DICT_BBOX_CENTERS][self.frameIdx][1])
                    self.bboxIsSet = True
                    self.bboxChangedAndSaved = True
                    if self.drawOverlay(False, drawingSavedBBox=self.bboxChangedAndSaved) :
                        self.frameLabel.setOverlay(self.overlayImg)
                else :
                    self.bbox[TL_IDX].setX(0)
                    self.bbox[TL_IDX].setY(0)
                    self.bbox[TR_IDX].setX(0)
                    self.bbox[TR_IDX].setY(0)
                    self.bbox[BR_IDX].setX(0)
                    self.bbox[BR_IDX].setY(0)
                    self.bbox[BL_IDX].setX(0)
                    self.bbox[BL_IDX].setY(0)

                    self.centerPoint.setX(0)
                    self.centerPoint.setY(0)
                    self.bboxIsSet = False
                    if self.drawOverlay(False, False, False) :
                        self.frameLabel.setOverlay(self.overlayImg)
                    
                self.settingBBox = False
        elif e.key() == QtCore.Qt.Key_R :
            if self.selectedSemSequenceIdx >= 0 and self.selectedSemSequenceIdx < len(self.semanticSequences) :
                
                frame = self.frameIdx-np.min(self.semanticSequences[self.selectedSemSequenceIdx][DICT_FRAMES_LOCATIONS].keys())
                if frame >= 0 and frame < len(self.semanticSequences[self.selectedSemSequenceIdx][DICT_FRAME_SEMANTICS]) :
                    sems = self.semanticSequences[self.selectedSemSequenceIdx][DICT_FRAME_SEMANTICS][frame, :]                        
                    self.nameSemanticAction(self.selectedSemSequenceIdx, int(np.argmax(sems)), True)
            
            
        sys.stdout.flush()
        
    def nameSemanticAction(self, sequenceIdx, actionIdx, doSave = False) :
        newName, ok = QtGui.QInputDialog.getText(self, "Name Action", "Name of {0}'s Action {1}".format(self.semanticSequences[sequenceIdx][DICT_SEQUENCE_NAME], actionIdx),
                                                 QtGui.QLineEdit.Normal, self.semanticSequences[sequenceIdx][DICT_SEMANTICS_NAMES][actionIdx])
        if ok and newName :
            print "NAMING", self.semanticSequences[sequenceIdx][DICT_SEMANTICS_NAMES][actionIdx], "AS", newName
            self.semanticSequences[sequenceIdx][DICT_SEMANTICS_NAMES][actionIdx] = newName
            self.showFrame(self.frameIdx)
            if doSave :
                np.save(self.semanticSequences[sequenceIdx][DICT_SEQUENCE_LOCATION], self.semanticSequences[sequenceIdx])
        
    def wheelEvent(self, e) :
        if not self.isModeBBox and e.modifiers() & QtCore.Qt.Modifier.CTRL :
            if e.delta() < 0 :
                self.scribbleBrushSizeSlider.setValue(np.max([2, self.brushSize-1]))
            else :
                self.scribbleBrushSizeSlider.setValue(np.min([250, self.brushSize+1]))
        else :
            if e.delta() < 0 :
                self.frameIdxSpinBox.setValue(self.frameIdx-1)
            else :
                self.frameIdxSpinBox.setValue(self.frameIdx+1)
            time.sleep(0.01)
        
    def eventFilter(self, obj, event) :
        if obj == self.semanticsLabel and event.type() == QtCore.QEvent.Type.MouseMove :
            if self.semanticsLabel.image != None and self.selectedSemSequenceIdx >= 0 and self.selectedSemSequenceIdx < len(self.semanticSequences) :
                xLoc = (self.semanticsLabel.width()-self.semanticsLabel.image.width())/2.0
                frame = (event.pos().x()-xLoc-1)/self.semanticsLabel.scaleFactor
#                 print event.pos().x(), xLoc, self.semanticsLabel.scaleFactor, self.semanticsLabel.image.width(), frame, int(np.round(frame))
                if frame >= 0 and frame < len(self.semanticSequences[self.selectedSemSequenceIdx][DICT_FRAME_SEMANTICS]) :
                    frame = int(np.round(frame))
                    sems = self.semanticSequences[self.selectedSemSequenceIdx][DICT_FRAME_SEMANTICS][frame, :]
                    QtGui.QToolTip.showText(QtGui.QCursor.pos(), "<b>{0}</b> showing <b>{1}</b> at frame <b>{2}</b>".format(self.semanticSequences[self.selectedSemSequenceIdx][DICT_SEQUENCE_NAME],
                                                                                                                            self.semanticSequences[self.selectedSemSequenceIdx][DICT_SEMANTICS_NAMES][int(np.argmax(sems))],
                                                                                                                            np.string0(frame+
                                                                                                                                       np.min(self.semanticSequences[self.selectedSemSequenceIdx][DICT_FRAMES_LOCATIONS].keys()))
                                                                                                                            ))
                    return True
        
        if obj == self.frameLabel and event.type() == QtCore.QEvent.Type.MouseMove :
            self.mouseMoved(event)
            return True
        elif obj == self.frameLabel and event.type() == QtCore.QEvent.Type.MouseButtonPress :
            self.mousePressed(event)
            return True
        elif obj == self.frameLabel and event.type() == QtCore.QEvent.Type.MouseButtonRelease :
            self.mouseReleased(event)
            return True
        elif (obj == self.frameIdxSpinBox or obj == self.frameIdxSlider) and event.type() == QtCore.QEvent.Type.KeyPress :
            self.keyPressEvent(event)
            return True
        elif obj == self.opticalFlowPriorControls and event.type() == QtCore.QEvent.Type.MouseButtonPress :
            self.showOpticalFlowPriorControls()
            return True
        elif obj == self.patchDifferencePriorControls and event.type() == QtCore.QEvent.Type.MouseButtonPress :
            self.showPatchDifferencePriorControls()
            return True
        elif obj == self.extraSegmentationControls and event.type() == QtCore.QEvent.Type.MouseButtonPress :
            self.showExtraSegmentationControls()
            return True
        
        return QtGui.QWidget.eventFilter(self, obj, event)
    
    def setSemanticsColor(self, startColor) :
        ## sets the color associated to the currently selected semantics
        if self.selectedSemSequenceIdx < len(self.semanticSequences) and self.selectedSemSequenceIdx >= 0:
            newSemanticsColor = QtGui.QColorDialog.getColor(startColor, self, "Choose Sequence Color")
            if newSemanticsColor.isValid() :
                self.semanticSequences[self.selectedSemSequenceIdx][DICT_REPRESENTATIVE_COLOR] = np.array([newSemanticsColor.red(),
                                                                                          newSemanticsColor.green(),
                                                                                          newSemanticsColor.blue()])
                np.save(self.semanticSequences[self.selectedSemSequenceIdx][DICT_SEQUENCE_LOCATION], self.semanticSequences[self.selectedSemSequenceIdx])
                
                self.semanticSequences = []
                self.loadSemanticSequences()
                
    def showOpticalFlowPriorControls(self, doShow=None) :
        
        if doShow != None :
            showVisible = doShow
        else :
            showVisible = not self.prevMaskImportanceSpinBox.isVisible()
        
        self.opticalFlowPriorControls.layout().itemAtPosition(1, 0).widget().setVisible(showVisible)
        self.opticalFlowPriorControls.layout().itemAtPosition(2, 0).widget().setVisible(showVisible)
        self.opticalFlowPriorControls.layout().itemAtPosition(3, 0).widget().setVisible(showVisible)
        self.opticalFlowPriorControls.layout().itemAtPosition(4, 0).widget().setVisible(showVisible)
        
        self.prevMaskImportanceSpinBox.setVisible(showVisible)
        self.prevMaskDilateSpinBox.setVisible(showVisible)
        self.prevMaskBlurSizeSpinBox.setVisible(showVisible)
        self.prevMaskBlurSigmaSpinBox.setVisible(showVisible)
        
    def showPatchDifferencePriorControls(self, doShow=None) :
        
        if doShow != None :
            showVisible = doShow
        else :
            showVisible = not self.diffPatchImportanceSpinBox.isVisible()
        
        self.patchDifferencePriorControls.layout().itemAtPosition(1, 0).widget().setVisible(showVisible)
        self.patchDifferencePriorControls.layout().itemAtPosition(2, 0).widget().setVisible(showVisible)
        
        self.diffPatchImportanceSpinBox.setVisible(showVisible)
        self.diffPatchMultiplierSpinBox.setVisible(showVisible)
        
    def showExtraSegmentationControls(self, doShow=None) :
        
        if doShow != None :
            showVisible = doShow
        else :
            showVisible = not self.doUseCenterSquareBox.isVisible()
        
        self.extraSegmentationControls.layout().itemAtPosition(1, 0).widget().setVisible(showVisible)
        self.extraSegmentationControls.layout().itemAtPosition(2, 0).widget().setVisible(showVisible)
        self.extraSegmentationControls.layout().itemAtPosition(3, 0).widget().setVisible(showVisible)
        self.extraSegmentationControls.layout().itemAtPosition(4, 0).widget().setVisible(showVisible)
        
        self.doUseCenterSquareBox.setVisible(showVisible)
        self.originalImageOpacitySlider.setVisible(showVisible)
        self.scribbleOpacitySlider.setVisible(showVisible)
        self.scribbleBrushSizeSlider.setVisible(showVisible)
        
    def refreshSegmentation(self) :
        self.setFocus()
        if self.selectedSemSequenceIdx >= 0 and self.selectedSemSequenceIdx < len(self.semanticSequences) and DICT_BBOXES in self.semanticSequences[self.selectedSemSequenceIdx].keys() :
            
            if DICT_MASK_LOCATION not in self.semanticSequences[self.selectedSemSequenceIdx].keys() :
                self.semanticSequences[self.selectedSemSequenceIdx][DICT_MASK_LOCATION] = self.seqDir + "/" + self.semanticSequences[self.selectedSemSequenceIdx][DICT_SEQUENCE_NAME] + "-maskedFlow/"
                if not os.path.isdir(self.semanticSequences[self.selectedSemSequenceIdx][DICT_MASK_LOCATION]) :
                    os.mkdir(self.semanticSequences[self.selectedSemSequenceIdx][DICT_MASK_LOCATION])
                np.save(self.semanticSequences[self.selectedSemSequenceIdx][DICT_SEQUENCE_LOCATION], self.semanticSequences[self.selectedSemSequenceIdx])
                
            if self.seqDir != "" and not os.path.isfile(self.seqDir+"/median.png") :
                QtGui.QMessageBox.warning(self, "Median Image Not Computed", ("The median image for this sequence has not been computed. The segmentation will "+
                                                                              "not produce the expected result. Please <i>Compute Median</i> first. "))
                return
            self.showFrame(self.frameIdx, True)
        
    def segmentSequence(self) :
        if self.seqDir != "" and not os.path.isfile(self.seqDir+"/median.png") :
            QtGui.QMessageBox.warning(self, "Median Image Not Computed", ("The median image for this sequence has not been computed. The segmentation will "+
                                                                          "not produce the expected result. Please <i>Compute Median</i> first. "))
            return
        if self.selectedSemSequenceIdx >= 0 and self.selectedSemSequenceIdx < len(self.semanticSequences) :
            if self.isModeBBox :
                self.toggleDefineMode()

            self.segmenting = True
            self.setFocus()
            if DICT_BBOXES in self.semanticSequences[self.selectedSemSequenceIdx].keys() and self.frameIdx in self.semanticSequences[self.selectedSemSequenceIdx][DICT_BBOXES].keys() :
                for frameIdx in np.arange(self.frameIdx, np.max(self.semanticSequences[self.selectedSemSequenceIdx][DICT_BBOXES].keys())+1) :
                    self.frameIdx = frameIdx
                    self.refreshSegmentation()
                    QtGui.QApplication.processEvents()
                    if not self.segmenting :
                        break;
            elif DICT_BBOXES in self.semanticSequences[self.selectedSemSequenceIdx].keys() :
                for frameIdx in np.sort(self.semanticSequences[self.selectedSemSequenceIdx][DICT_BBOXES].keys()) :
                    self.frameIdx = frameIdx
                    self.refreshSegmentation()
                    QtGui.QApplication.processEvents()
                    if not self.segmenting :
                        break;
            self.segmenting = False
        
    def createGUI(self) :
        
        ## WIDGETS ##
        self.loadedFrameSequenceLabel = QtGui.QLabel("No Raw Frame Sequence Loaded [Alt+F + Alt+R]")
        self.loadedFrameSequenceLabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignHCenter)
        
        self.frameLabel = ImageLabel("Frame")
        self.frameLabel.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.MinimumExpanding)
        self.frameLabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignHCenter)
        self.frameLabel.setStyleSheet("QLabel { margin: 0px; border: 1px solid gray; border-radius: 0px; }")
        self.frameLabel.installEventFilter(self)
        
        self.frameInfo = QtGui.QLabel("Info text")
        self.frameInfo.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignHCenter)
        
        self.frameIdxSlider = SemanticsSlider(QtCore.Qt.Horizontal)
        self.frameIdxSlider.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Minimum)
        self.frameIdxSlider.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.frameIdxSlider.setTickPosition(QtGui.QSlider.TicksBothSides)
        self.frameIdxSlider.setMinimum(0)
        self.frameIdxSlider.setMaximum(0)
        self.frameIdxSlider.setTickInterval(100)
        self.frameIdxSlider.setSingleStep(1)
        self.frameIdxSlider.setPageStep(100)
        self.frameIdxSlider.installEventFilter(self)
    
        self.frameIdxSpinBox = QtGui.QSpinBox()
        self.frameIdxSpinBox.setRange(0, 0)
        self.frameIdxSpinBox.setSingleStep(1)
        self.frameIdxSpinBox.installEventFilter(self)
        
        self.semanticsLabel = SemanticsLabel("Actions")
        self.semanticsLabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignHCenter)
        self.semanticsLabel.setMouseTracking(True)
        self.semanticsLabel.installEventFilter(self)
        
#         self.loadedSequencesListTable = QtGui.QTableWidget(1, 1)
#         self.loadedSequencesListTable.horizontalHeader().setStretchLastSection(True)
#         self.loadedSequencesListTable.setHorizontalHeaderItem(0, QtGui.QTableWidgetItem("Loaded Semantic Sequences"))
#         self.loadedSequencesListTable.horizontalHeader().setResizeMode(QtGui.QHeaderView.Fixed)
#         self.loadedSequencesListTable.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
#         self.loadedSequencesListTable.setEditTriggers(QtGui.QAbstractItemView.NoEditTriggers)
#         self.loadedSequencesListTable.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
#         self.loadedSequencesListTable.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
#         self.loadedSequencesListTable.setItem(0, 0, QtGui.QTableWidgetItem("None"))

        self.loadedSequencesListModel = QtGui.QStandardItemModel(1, 1)
        self.loadedSequencesListModel.setHorizontalHeaderLabels(["Loaded Actor Sequences"])
        self.loadedSequencesListModel.setItem(0, 0, QtGui.QStandardItem("None"))
        
        self.loadedSequencesListTable = QtGui.QTableView()
        self.loadedSequencesListTable.setEditTriggers(QtGui.QAbstractItemView.NoEditTriggers)
        self.loadedSequencesListTable.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
        self.loadedSequencesListTable.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
        self.loadedSequencesListTable.horizontalHeader().setStretchLastSection(True)
        self.loadedSequencesListTable.horizontalHeader().setResizeMode(QtGui.QHeaderView.Fixed)
        self.loadedSequencesListTable.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.MinimumExpanding)
        self.loadedSequencesListTable.verticalHeader().setVisible(False)
        self.loadedSequencesListTable.verticalHeader().setDefaultSectionSize(LIST_SECTION_SIZE)
        self.loadedSequencesListTable.setMinimumHeight(10)
        self.loadedSequencesListTable.setEnabled(False)

        self.delegateList = [ListDelegate()]
        self.loadedSequencesListTable.setItemDelegateForRow(0, self.delegateList[-1])
        self.loadedSequencesListTable.setModel(self.loadedSequencesListModel)
        
        self.computeMedianImageButton = QtGui.QPushButton("Compute Median")
        self.computeDistanceMatrixButton = QtGui.QPushButton("Compute Distance Matrix")
        
        self.newSemSequenceButton = QtGui.QPushButton("&New Actor Sequence")
        self.semanticsColorButton = QtGui.QPushButton("Set &Color")
        self.semanticsColorButton.setCheckable(True)
        
        self.addFramesToSequenceButton = QtGui.QPushButton("Add Frames")
        self.addFramesToSequenceButton.setToolTip("Add frames within interval to the sequence and set their bbox if desired")
        
        self.addEndFrameToSequenceButton = QtGui.QPushButton("Add Empty Frame")
        self.addEndFrameToSequenceButton.setToolTip("Adds an empty frame at the end of the sequence (useful when the sprite needs an <i>invisible</i> semantic state)")
        
        
        ## TRACKING ##
        
        
        self.deleteCurrentSemSequenceBBoxButton = QtGui.QPushButton("Delete BBox")
        self.deleteCurrentSemSequenceBBoxButton.setToolTip("Deletes the bounding box defined for currently selected semantic sequence at the current frame")
        self.setCurrentSemSequenceBBoxButton = QtGui.QPushButton("Set BBox")
        self.setCurrentSemSequenceBBoxButton.setToolTip("Sets the currently visualized bounding box for the currently selected semantic sequence at the current frame")
        
        
        ## SEGMENTATION ##
        
        
        self.doUseOpticalFlowPriorBox = QtGui.QCheckBox()
        self.doUseOpticalFlowPriorBox.setChecked(True)
        self.doUseOpticalFlowPriorBox.setToolTip("Use displaced mask (dilated and blurred) of previous frame as a prior for segmentation")
        
        self.prevMaskImportanceSpinBox = QtGui.QDoubleSpinBox()
        self.prevMaskImportanceSpinBox.setRange(0.0, 1.0)
        self.prevMaskImportanceSpinBox.setSingleStep(0.01)
        self.prevMaskImportanceSpinBox.setValue(0.35)
        self.prevMaskImportanceSpinBox.setToolTip("The higher, the more important will the displaced mask be")
        
        self.prevMaskDilateSpinBox = QtGui.QSpinBox()
        self.prevMaskDilateSpinBox.setRange(1, 33)
        self.prevMaskDilateSpinBox.setSingleStep(2)
        self.prevMaskDilateSpinBox.setValue(13)
        self.prevMaskDilateSpinBox.setToolTip("Number of pixels to dilate the displaced mask by")
        
        self.prevMaskBlurSizeSpinBox = QtGui.QSpinBox()
        self.prevMaskBlurSizeSpinBox.setRange(1, 65)
        self.prevMaskBlurSizeSpinBox.setSingleStep(2)
        self.prevMaskBlurSizeSpinBox.setValue(31)
        self.prevMaskBlurSizeSpinBox.setToolTip("Size of blur to apply to dilated and dispalced mask")
        
        self.prevMaskBlurSigmaSpinBox = QtGui.QDoubleSpinBox()
        self.prevMaskBlurSigmaSpinBox.setRange(0.5, 5.0)
        self.prevMaskBlurSigmaSpinBox.setSingleStep(0.1)
        self.prevMaskBlurSigmaSpinBox.setValue(2.5)
        
        
        
        self.doUsePatchDiffPriorBox = QtGui.QCheckBox()
        self.doUsePatchDiffPriorBox.setToolTip("Use difference of patch to the static background as a priro for segmentation")
        
        self.diffPatchImportanceSpinBox = QtGui.QDoubleSpinBox()
        self.diffPatchImportanceSpinBox.setRange(0.0, 1.0)
        self.diffPatchImportanceSpinBox.setSingleStep(0.001)
        self.diffPatchImportanceSpinBox.setValue(0.015)
        self.diffPatchImportanceSpinBox.setToolTip("The higher, the more important will the patch difference be")
        
        self.diffPatchMultiplierSpinBox = QtGui.QDoubleSpinBox()
        self.diffPatchMultiplierSpinBox.setRange(1.0, 10000.0)
        self.diffPatchMultiplierSpinBox.setSingleStep(10.0)
        self.diffPatchMultiplierSpinBox.setValue(1000.0)
        self.diffPatchMultiplierSpinBox.setToolTip("The higher, the more likely a pixel will be considered foreground if they look different from the background")
        
        

        self.doUseGradientsCostBox = QtGui.QCheckBox()
        self.doUseGradientsCostBox.setToolTip("Use the gradient normalization when defining the pairwise cost (Eq. 5 in Graphcut Textures)")
        
        self.doUseCenterSquareBox = QtGui.QCheckBox()
        self.doUseCenterSquareBox.setChecked(True)
        self.doUseCenterSquareBox.setToolTip("Force 6x6 patch in the middle of the bounding box to be assigned the foreground label")
        
        self.originalImageOpacitySlider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.originalImageOpacitySlider.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Minimum)
        self.originalImageOpacitySlider.setMinimum(0)
        self.originalImageOpacitySlider.setMaximum(100)
        self.originalImageOpacitySlider.setValue(50)
        
        self.scribbleOpacitySlider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.scribbleOpacitySlider.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Minimum)
        self.scribbleOpacitySlider.setMinimum(0)
        self.scribbleOpacitySlider.setMaximum(100)
        self.scribbleOpacitySlider.setValue(50)
        
        self.scribbleBrushSizeSlider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.scribbleBrushSizeSlider.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Minimum)
        self.scribbleBrushSizeSlider.setMinimum(2)
        self.scribbleBrushSizeSlider.setMaximum(250)
        self.scribbleBrushSizeSlider.setValue(20)
        self.scribbleBrushSizeSlider.setToolTip("Use CTRL+mouseWheel to change in scribble mode")
        
        
        
        self.refreshSegmentationButton = QtGui.QPushButton("(&Re-) Segment")
        self.segmentSequenceButton = QtGui.QPushButton("&Segment Sequence")
        
        
        
        self.semanticsSigmaSpinBox = QtGui.QDoubleSpinBox()
        self.semanticsSigmaSpinBox.setRange(0.0, 100.0)
        self.semanticsSigmaSpinBox.setSingleStep(0.01)
        self.semanticsSigmaSpinBox.setValue(6)
        self.semanticsSigmaSpinBox.setToolTip("The higher, the wider the probability distribution computed from the distance matrix")
        
        self.numExtraFramesSpinBox = QtGui.QSpinBox()
        self.numExtraFramesSpinBox.setRange(0, 16)
        self.numExtraFramesSpinBox.setSingleStep(2)
        self.numExtraFramesSpinBox.setValue(0)
        self.numExtraFramesSpinBox.setToolTip("Number of extra frames to add as extra training around the manually tagged one")
        
        
        self.isMovingSpriteBox = QtGui.QCheckBox("Is Moving")
        self.isMovingSpriteBox.setToolTip("Indicates if the current input sequence is to be treated as a moving sprite")
        
        self.sigmaMultiplierSlider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.sigmaMultiplierSlider.setToolTip("The higher the value (right), the less important is the jump quality and more jumps can be used during synthesis")
        self.sigmaMultiplierSlider.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Minimum)
        self.sigmaMultiplierSlider.setMinimum(1)
        self.sigmaMultiplierSlider.setMaximum(1000)
        self.sigmaMultiplierSlider.setValue(2)
        
        
        ## SIGNALS ##
        
        self.frameIdxSlider.valueChanged[int].connect(self.frameIdxSpinBox.setValue)
        self.frameIdxSpinBox.valueChanged[int].connect(self.frameIdxSlider.setValue)
        self.frameIdxSpinBox.valueChanged[int].connect(self.showFrame)
        self.frameIdxSpinBox.valueChanged[int].connect(self.semanticsLabel.setCurrentFrame)
        
        self.computeMedianImageButton.clicked.connect(self.computeMedianPressed)
        self.computeDistanceMatrixButton.clicked.connect(self.computeDistanceMatrixPressed)
        self.addFramesToSequenceButton.clicked.connect(self.addFramesToSequencePressed)
        self.addEndFrameToSequenceButton.clicked.connect(self.addEndFrameToSequencePressed)
    
        self.loadedSequencesListTable.clicked.connect(self.changeSelectedSemSequence)
        
        self.newSemSequenceButton.clicked.connect(self.createNewSemanticSequence)
        
        self.deleteCurrentSemSequenceBBoxButton.clicked.connect(self.deleteCurrentSemSequenceFrameBBox)
        self.setCurrentSemSequenceBBoxButton.clicked.connect(self.setCurrentSemSequenceFrameBBox)
        
        self.doUseOpticalFlowPriorBox.stateChanged.connect(self.refreshSegmentation)
        self.prevMaskImportanceSpinBox.editingFinished.connect(self.refreshSegmentation)
        self.prevMaskDilateSpinBox.editingFinished.connect(self.refreshSegmentation)
        self.prevMaskBlurSizeSpinBox.editingFinished.connect(self.refreshSegmentation)
        self.prevMaskBlurSigmaSpinBox.editingFinished.connect(self.refreshSegmentation)
        
        self.doUsePatchDiffPriorBox.stateChanged.connect(self.refreshSegmentation)
        self.diffPatchImportanceSpinBox.editingFinished.connect(self.refreshSegmentation)
        self.diffPatchMultiplierSpinBox.editingFinished.connect(self.refreshSegmentation)
        
        self.doUseGradientsCostBox.stateChanged.connect(self.refreshSegmentation)
        self.doUseCenterSquareBox.stateChanged.connect(self.refreshSegmentation)
        
        self.refreshSegmentationButton.clicked.connect(self.refreshSegmentation)
        self.segmentSequenceButton.clicked.connect(self.segmentSequence)
        
        self.originalImageOpacitySlider.valueChanged[int].connect(self.setOriginalImageOpacity)
        self.scribbleOpacitySlider.valueChanged[int].connect(self.setScribbleOpacity)
        
        self.scribbleBrushSizeSlider.valueChanged[int].connect(self.setBrushSizeAndCursor)
        
        
        self.semanticsSigmaSpinBox.editingFinished.connect(self.computeLabelPropagation)
        
        
        ## LAYOUTS ##
        
        self.trackingControls = QtGui.QGroupBox("Sequence Controls")
        self.trackingControls.setStyleSheet("QGroupBox { margin: 5px; border: 2px groove gray; border-radius: 3px; } QGroupBox::title {left: 15px; top: -7px; font: bold;}")
        trackingControlsLayout = QtGui.QGridLayout(); idx = 0
        trackingControlsLayout.addWidget(self.computeMedianImageButton, idx, 0, 1, 1, QtCore.Qt.AlignLeft)
        trackingControlsLayout.addWidget(self.computeDistanceMatrixButton, idx, 1, 1, 1, QtCore.Qt.AlignLeft); idx += 1
        trackingControlsLayout.addWidget(self.isMovingSpriteBox, idx, 0, 1, 1, QtCore.Qt.AlignLeft)
        trackingControlsLayout.addWidget(self.sigmaMultiplierSlider, idx, 1, 1, 1, QtCore.Qt.AlignLeft); idx += 1
        trackingControlsLayout.addWidget(self.newSemSequenceButton, idx, 0, 1, 1, QtCore.Qt.AlignLeft)
        trackingControlsLayout.addWidget(self.semanticsColorButton, idx, 1, 1, 1, QtCore.Qt.AlignLeft); idx += 1
        trackingControlsLayout.addWidget(self.addFramesToSequenceButton, idx, 0, 1, 1, QtCore.Qt.AlignCenter)
        trackingControlsLayout.addWidget(self.addEndFrameToSequenceButton, idx, 1, 1, 1, QtCore.Qt.AlignCenter); idx += 1
        trackingControlsLayout.addWidget(self.deleteCurrentSemSequenceBBoxButton, idx, 0, 1, 1, QtCore.Qt.AlignLeft)
        trackingControlsLayout.addWidget(self.setCurrentSemSequenceBBoxButton, idx, 1, 1, 1, QtCore.Qt.AlignLeft); idx += 1
        self.trackingControls.setLayout(trackingControlsLayout)
        
        
        self.opticalFlowPriorControls = QtGui.QGroupBox("")
        self.opticalFlowPriorControls.setStyleSheet("QGroupBox { margin: 5px; border: 2px groove gray; border-radius: 3px; } QGroupBox::title {left: 15px; top: -7px; font: bold;}")
        opticalFlowPriorControlsLayout = QtGui.QGridLayout(); idx = 0
        opticalFlowPriorControlsLayout.addWidget(QtGui.QLabel("Use Optical Flow Prior"), idx, 0, 1, 1, QtCore.Qt.AlignLeft)
        opticalFlowPriorControlsLayout.addWidget(self.doUseOpticalFlowPriorBox, idx, 1, 1, 1, QtCore.Qt.AlignLeft); idx += 1
        opticalFlowPriorControlsLayout.addWidget(QtGui.QLabel("Mask<sub>t-1</sub> Importance"), idx, 0, 1, 1, QtCore.Qt.AlignLeft)
        opticalFlowPriorControlsLayout.addWidget(self.prevMaskImportanceSpinBox, idx, 1, 1, 1, QtCore.Qt.AlignLeft); idx += 1
        opticalFlowPriorControlsLayout.addWidget(QtGui.QLabel("Mask<sub>t-1</sub> Dilation"), idx, 0, 1, 1, QtCore.Qt.AlignLeft)
        opticalFlowPriorControlsLayout.addWidget(self.prevMaskDilateSpinBox, idx, 1, 1, 1, QtCore.Qt.AlignLeft); idx += 1
        opticalFlowPriorControlsLayout.addWidget(QtGui.QLabel("Mask<sub>t-1</sub> Blur Size"), idx, 0, 1, 1, QtCore.Qt.AlignLeft)
        opticalFlowPriorControlsLayout.addWidget(self.prevMaskBlurSizeSpinBox, idx, 1, 1, 1, QtCore.Qt.AlignLeft); idx += 1
        opticalFlowPriorControlsLayout.addWidget(QtGui.QLabel("Mask<sub>t-1</sub> Blur Sigma"), idx, 0, 1, 1, QtCore.Qt.AlignLeft)
        opticalFlowPriorControlsLayout.addWidget(self.prevMaskBlurSigmaSpinBox, idx, 1, 1, 1, QtCore.Qt.AlignLeft); idx += 1
        self.opticalFlowPriorControls.setLayout(opticalFlowPriorControlsLayout)
        self.opticalFlowPriorControls.installEventFilter(self)
        
        
        self.patchDifferencePriorControls = QtGui.QGroupBox("")
        self.patchDifferencePriorControls.setStyleSheet("QGroupBox { margin: 5px; border: 2px groove gray; border-radius: 3px; } QGroupBox::title {left: 15px; top: -7px; font: bold;}")
        patchDifferencePriorControlsLayout = QtGui.QGridLayout(); idx = 0
        patchDifferencePriorControlsLayout.addWidget(QtGui.QLabel("Use Patch Diff Prior"), idx, 0, 1, 1, QtCore.Qt.AlignLeft)
        patchDifferencePriorControlsLayout.addWidget(self.doUsePatchDiffPriorBox, idx, 1, 1, 1, QtCore.Qt.AlignLeft); idx += 1
        patchDifferencePriorControlsLayout.addWidget(QtGui.QLabel("Patch Diff Importance"), idx, 0, 1, 1, QtCore.Qt.AlignLeft)
        patchDifferencePriorControlsLayout.addWidget(self.diffPatchImportanceSpinBox, idx, 1, 1, 1, QtCore.Qt.AlignLeft); idx += 1
        patchDifferencePriorControlsLayout.addWidget(QtGui.QLabel("Patch Diff Multiplier"), idx, 0, 1, 1, QtCore.Qt.AlignLeft)
        patchDifferencePriorControlsLayout.addWidget(self.diffPatchMultiplierSpinBox, idx, 1, 1, 1, QtCore.Qt.AlignLeft); idx += 1
        self.patchDifferencePriorControls.setLayout(patchDifferencePriorControlsLayout)
        self.patchDifferencePriorControls.installEventFilter(self)
        
        
        self.extraSegmentationControls = QtGui.QGroupBox("")
        self.extraSegmentationControls.setStyleSheet("QGroupBox { margin: 5px; border: 2px groove gray; border-radius: 3px; } QGroupBox::title {left: 15px; top: -7px; font: bold;}")
        extraSegmentationControlsLayout = QtGui.QGridLayout(); idx = 0
        extraSegmentationControlsLayout.addWidget(QtGui.QLabel("Use Gradients Cost"), idx, 0, 1, 1, QtCore.Qt.AlignLeft)
        extraSegmentationControlsLayout.addWidget(self.doUseGradientsCostBox, idx, 1, 1, 1, QtCore.Qt.AlignLeft); idx += 1
        extraSegmentationControlsLayout.addWidget(QtGui.QLabel("Force FG Center Square"), idx, 0, 1, 1, QtCore.Qt.AlignLeft)
        extraSegmentationControlsLayout.addWidget(self.doUseCenterSquareBox, idx, 1, 1, 1, QtCore.Qt.AlignLeft); idx += 1
        extraSegmentationControlsLayout.addWidget(QtGui.QLabel("Original Image Opacity"), idx, 0, 1, 1, QtCore.Qt.AlignLeft)
        extraSegmentationControlsLayout.addWidget(self.originalImageOpacitySlider, idx, 1, 1, 1, QtCore.Qt.AlignLeft); idx += 1
        extraSegmentationControlsLayout.addWidget(QtGui.QLabel("Scribble Opacity"), idx, 0, 1, 1, QtCore.Qt.AlignLeft)
        extraSegmentationControlsLayout.addWidget(self.scribbleOpacitySlider, idx, 1, 1, 1, QtCore.Qt.AlignLeft); idx += 1
        extraSegmentationControlsLayout.addWidget(QtGui.QLabel("Brush Size"), idx, 0, 1, 1, QtCore.Qt.AlignLeft)
        extraSegmentationControlsLayout.addWidget(self.scribbleBrushSizeSlider, idx, 1, 1, 1, QtCore.Qt.AlignLeft); idx += 1
        self.extraSegmentationControls.setLayout(extraSegmentationControlsLayout)
        self.extraSegmentationControls.installEventFilter(self)
        
        
        self.segmentationControls = QtGui.QGroupBox("Segmentation Controls")
        self.segmentationControls.setStyleSheet("QGroupBox { margin: 5px; border: 2px groove gray; border-radius: 3px; } QGroupBox::title {left: 15px; top: -7px; font: bold;}")
        segmentationControlsLayout = QtGui.QGridLayout(); idx = 0
        segmentationControlsLayout.addWidget(self.opticalFlowPriorControls, idx, 0, 1, 2, QtCore.Qt.AlignLeft); idx += 1
        segmentationControlsLayout.addWidget(self.patchDifferencePriorControls, idx, 0, 1, 2, QtCore.Qt.AlignLeft); idx += 1
        segmentationControlsLayout.addWidget(self.extraSegmentationControls, idx, 0, 1, 2, QtCore.Qt.AlignLeft); idx += 1
        segmentationControlsLayout.addWidget(self.refreshSegmentationButton, idx, 0, 1, 1, QtCore.Qt.AlignLeft)
        segmentationControlsLayout.addWidget(self.segmentSequenceButton, idx, 1, 1, 1, QtCore.Qt.AlignLeft); idx += 1
        self.segmentationControls.setLayout(segmentationControlsLayout)
        
        self.semanticsControls = QtGui.QGroupBox("Action Definition Controls")
        self.semanticsControls.setStyleSheet("QGroupBox { margin: 5px; border: 2px groove gray; border-radius: 3px; } QGroupBox::title {left: 15px; top: -7px; font: bold;}")
        semanticsControlsLayout = QtGui.QGridLayout(); idx = 0
        semanticsControlsLayout.addWidget(QtGui.QLabel("Propagation Sigma (/100)"), idx, 0, 1, 1, QtCore.Qt.AlignLeft)
        semanticsControlsLayout.addWidget(self.semanticsSigmaSpinBox, idx, 1, 1, 1, QtCore.Qt.AlignLeft); idx += 1
        semanticsControlsLayout.addWidget(QtGui.QLabel("Number of extra frames"), idx, 0, 1, 1, QtCore.Qt.AlignLeft)
        semanticsControlsLayout.addWidget(self.numExtraFramesSpinBox, idx, 1, 1, 1, QtCore.Qt.AlignLeft); idx += 1
        self.semanticsControls.setLayout(semanticsControlsLayout)
        
        
        mainLayout = QtGui.QHBoxLayout()
        
        controlsLayout = QtGui.QVBoxLayout()
        controlsLayout.addWidget(self.loadedFrameSequenceLabel)
        controlsLayout.addWidget(self.loadedSequencesListTable)
        controlsLayout.addWidget(self.trackingControls)
        controlsLayout.addWidget(self.segmentationControls)
        controlsLayout.addWidget(self.semanticsControls)
        
        sliderLayout = QtGui.QHBoxLayout()
        sliderLayout.addWidget(self.frameIdxSlider)
        sliderLayout.addWidget(self.frameIdxSpinBox)
        
        self.frameLabelWidget = QtGui.QGroupBox("")
        frameHLayout = QtGui.QHBoxLayout()
        frameHLayout.addStretch()
        frameHLayout.addWidget(self.frameLabel)
        frameHLayout.addStretch()
        
        frameLabelWidgetLayout = QtGui.QVBoxLayout()
        frameLabelWidgetLayout.addStretch()
        frameLabelWidgetLayout.addWidget(self.semanticsLabel)
        frameLabelWidgetLayout.addLayout(frameHLayout)
        frameLabelWidgetLayout.addWidget(self.frameInfo)
        frameLabelWidgetLayout.addStretch()
        
        self.frameLabelWidget.setLayout(frameLabelWidgetLayout)
        self.frameLabelWidget.setStyleSheet("QGroupBox { background-color:rgba(255, 0, 0, 28); }")
        
        
        frameVLayout = QtGui.QVBoxLayout()
        frameVLayout.addWidget(self.frameLabelWidget)
        frameVLayout.addLayout(sliderLayout)
        
        mainLayout.addLayout(controlsLayout)
        mainLayout.addLayout(frameVLayout)
        self.setLayout(mainLayout)
        
        self.showOpticalFlowPriorControls(False)
        self.showPatchDifferencePriorControls(False)
        self.showExtraSegmentationControls(False)

