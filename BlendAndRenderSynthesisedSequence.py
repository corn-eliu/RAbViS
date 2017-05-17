
# coding: utf-8

# In[ ]:


import numpy as np
import sys
import scipy as sp
from IPython.display import clear_output

import cv2
import time
import os
import scipy.io as sio
import glob
import itertools

import scipy.sparse
import PIL.Image
import pyamg

from PIL import Image

import opengm
import soundfile as sf

from matplotlib.patches import Rectangle

import shutil, errno

def copyanything(src, dst):
    try:
        shutil.copytree(src, dst)
    except OSError as exc: # python >2.5
        if exc.errno == errno.ENOTDIR:
            shutil.copy(src, dst)
        else: raise


DICT_SEQUENCE_NAME = 'semantic_sequence_name'
DICT_BBOXES = 'bboxes'
DICT_FOOTPRINTS = 'footprints' ## same as bboxes but it indicates the footprint of the sprite on the ground plane
DICT_BBOX_ROTATIONS = 'bbox_rotations'
DICT_BBOX_CENTERS = 'bbox_centers'
DICT_FRAMES_LOCATIONS = 'frame_locs'
DICT_MASK_LOCATION = 'frame_masks_location'
DICT_SEQUENCE_FRAMES = 'sequence_frames'
DICT_SEQUENCE_IDX = 'semantic_sequence_idx' # index of the instantiated sem sequence in the list of all used sem sequences for a synthesised sequence
DICT_DESIRED_SEMANTICS = 'desired_semantics' # stores what the desired semantics are for a certain sprite 
#(I could index them by the frame when the toggle happened instead of using the below but maybe ordering is important and I would lose that using a dict)
DICT_FRAME_SEMANTIC_TOGGLE = 'frame_semantic_toggle'# stores the frame index in the generated sequence when the desired semantics have changed
DICT_ICON_TOP_LEFT = "icon_top_left"
DICT_ICON_FRAME_KEY = "icon_frame_key"
DICT_ICON_SIZE = "icon_size"
DICT_REPRESENTATIVE_COLOR = 'representative_color'
DICT_OFFSET = "instance_offset"
DICT_SCALE = "instance_scale"
DICT_FRAME_SEMANTICS = "semantics_per_frame"
DICT_USED_SEQUENCES = "used_semantic_sequences"
DICT_SEQUENCE_INSTANCES = "sequence_instances"
DICT_SEQUENCE_BG = "sequence_background_image"
DICT_SEQUENCE_LOCATION = "sequence_location"
DICT_PATCHES_LOCATION = "sequence_preloaded_patches_location"
DICT_TRANSITION_COSTS_LOCATION = "sequence_precomputed_transition_costs_location"

GRAPH_MAX_COST = 10000000.0

# dataPath = "/home/ilisescu/PhD/data/"
# dataSet = "havana/"

# dataPath = "/media/ilisescu/Data1/PhD/data/"
# dataSet = "clouds_subsample10/"
# dataSet = "theme_park_cloudy/"
# dataSet = "theme_park_sunny/"
# dataSet = "wave1/"
# dataSet = "wave1/"
formatString = "{:05d}.png"

TL_IDX = 0
TR_IDX = 1
BR_IDX = 2
BL_IDX = 3

# <codecell>

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

# <codecell>

## used for enlarging bbox used to decide size of patch around it (percentage)
PATCH_BORDER = 0.4
def getSpritePatch(sprite, frameKey, frameWidth, frameHeight) :
    """Computes sprite patch based on its bbox
    
        \t  sprite      : dictionary containing relevant sprite data
        \t  frameKey    : the key of the frame the sprite patch is taken from
        \t  frameWidth  : width of original image
        \t  frameHeight : height of original image
           
        return: spritePatch, offset, patchSize,
                [left, top, bottom, right] : array of booleans telling whether the expanded bbox touches the corresponding border of the image"""
    
    ## get the bbox for the current sprite frame, make it larger and find the rectangular patch to work with
    ## boundaries of the patch [min, max]
    
    ## returns sprite patch based on bbox and returns it along with the offset [x, y] and it's size [rows, cols]
    
    ## make bbox bigger
    largeBBox = sprite[DICT_BBOXES][frameKey].T
    ## move to origin
    largeBBox = np.dot(np.array([[-sprite[DICT_BBOX_CENTERS][frameKey][0], 1.0, 0.0], 
                                 [-sprite[DICT_BBOX_CENTERS][frameKey][1], 0.0, 1.0]]), 
                        np.vstack((np.ones((1, largeBBox.shape[1])), largeBBox)))
    ## make bigger
    largeBBox = np.dot(np.array([[0.0, 1.0 + PATCH_BORDER, 0.0], 
                                 [0.0, 0.0, 1.0 + PATCH_BORDER]]), 
                        np.vstack((np.ones((1, largeBBox.shape[1])), largeBBox)))
    ## move back tooriginal center
    largeBBox = np.dot(np.array([[sprite[DICT_BBOX_CENTERS][frameKey][0], 1.0, 0.0], 
                                 [sprite[DICT_BBOX_CENTERS][frameKey][1], 0.0, 1.0]]), 
                        np.vstack((np.ones((1, largeBBox.shape[1])), largeBBox)))
    
    xBounds = np.zeros(2); yBounds = np.zeros(2)
    
    ## make sure xBounds are in between 0 and width and yBounds are in between 0 and height
    xBounds[0] = np.max((0, np.min(largeBBox[0, :])))
    xBounds[1] = np.min((frameWidth, np.max(largeBBox[0, :])))
    yBounds[0] = np.max((0, np.min(largeBBox[1, :])))
    yBounds[1] = np.min((frameHeight, np.max(largeBBox[1, :])))
    
    offset = np.array([np.round(np.array([xBounds[0], yBounds[0]]))], dtype=int).T # [x, y]
    patchSize = np.array(np.round(np.array([yBounds[1]-yBounds[0], xBounds[1]-xBounds[0]])), dtype=int) # [rows, cols]
    
    spritePatch = np.array(Image.open(sprite[DICT_FRAMES_LOCATIONS][frameKey]))[offset[1]:offset[1]+patchSize[0], offset[0]:offset[0]+patchSize[1], :]
    
    return spritePatch, offset, patchSize, [np.min((largeBBox)[0, :]) > 0.0 ,
                                            np.min((largeBBox)[1, :]) > 0.0 ,
                                            np.max((largeBBox)[1, :]) < frameHeight,
                                            np.max((largeBBox)[0, :]) < frameWidth]


def getPatchPriors(bgPatch, spritePatch, offset, patchSize, sprite, frameKey, prevFrameKey = None, prevFrameAlphaLoc = "",
                   prevMaskImportance = 0.8, prevMaskDilate = 13, prevMaskBlurSize = 31, prevMaskBlurSigma = 2.5,
                   diffPatchImportance = 0.015, diffPatchMultiplier = 1000.0, useOpticalFlow = True, useDiffPatch = False) :
    """Computes priors for background and sprite patches
    
        \t  bgPatch             : background patch
        \t  spritePatch         : sprite patch
        \t  offset              : [x, y] position of patches in the coordinate system of the original images
        \t  patchSize           : num of [rows, cols] per patches
        \t  sprite              : dictionary containing relevant sprite data
        \t  frameKey            : the key of the frame the sprite patch is taken from
        \t  prevFrameKey        : the key of the previous frame
        \t  prevFrameAlphaLoc   : location of the previous frame
        \t  prevMaskImportance  : balances the importance of the prior based on the remapped mask of the previous frame
        \t  prevMaskDilate      : amount of dilation to perform on previous frame's mask
        \t  prevMaskBlurSize    : size of the blurring kernel perfomed on previous frame's mask
        \t  prevMaskBlurSigma   : variance of the gaussian blurring perfomed on previous frame's mask
        \t  diffPatchImportance : balances the importance of the prior based on difference of patch to background
        \t  diffPatchMultiplier : multiplier that changes the scaling of the difference based cost
        \t  useOpticalFlow      : modify sprite prior by the mask of the previous frame
        \t  useDiffPatch        : modify bg prior by difference of sprite to bg patch
           
        return: bgPrior, spritePrior"""
    
    ## get uniform prior for bg patch
    bgPrior = -np.log(np.ones(patchSize)/np.prod(patchSize))
    
    ## get prior for sprite patch
    spritePrior = np.zeros(patchSize)
    xs = np.ndarray.flatten(np.arange(patchSize[1], dtype=float).reshape((patchSize[1], 1)).repeat(patchSize[0], axis=-1))
    ys = np.ndarray.flatten(np.arange(patchSize[0], dtype=float).reshape((1, patchSize[0])).repeat(patchSize[1], axis=0))
    data = np.vstack((xs.reshape((1, len(xs))), ys.reshape((1, len(ys)))))
    
    ## get covariance and means of prior on patch by using the bbox
    spriteBBox = sprite[DICT_BBOXES][frameKey].T
    segment1 = spriteBBox[:, 0] - spriteBBox[:, 1]
    segment2 = spriteBBox[:, 1] - spriteBBox[:, 2]
    sigmaX = np.linalg.norm(segment1)/3.7
    sigmaY = np.linalg.norm(segment2)/3.7
    
    rotRadians = sprite[DICT_BBOX_ROTATIONS][frameKey]
    
    rotMat = np.array([[np.cos(rotRadians), -np.sin(rotRadians)], [np.sin(rotRadians), np.cos(rotRadians)]])
    
    means = np.reshape(sprite[DICT_BBOX_CENTERS][frameKey], (2, 1)) - offset
    covs = np.dot(np.dot(rotMat.T, np.array([[sigmaX**2, 0.0], [0.0, sigmaY**2]])), rotMat)
    
    spritePrior = np.reshape(minusLogMultivariateNormal(data, means, covs, True), patchSize, order='F')
    
    ## change the spritePrior using optical flow stuff
    if useOpticalFlow and prevFrameKey != None :
        prevFrameName = sprite[DICT_FRAMES_LOCATIONS][prevFrameKey].split('/')[-1]
        nextFrameName = sprite[DICT_FRAMES_LOCATIONS][frameKey].split('/')[-1]
        
        if os.path.isfile(prevFrameAlphaLoc+prevFrameName) :
            alpha = np.array(Image.open(prevFrameAlphaLoc+prevFrameName))[:, :, -1]/255.0

            flow = cv2.calcOpticalFlowFarneback(cv2.cvtColor(np.array(Image.open(dataPath+dataSet+nextFrameName)), cv2.COLOR_RGB2GRAY), 
                                                cv2.cvtColor(np.array(Image.open(dataPath+dataSet+prevFrameName)), cv2.COLOR_RGB2GRAY), 
                                                0.5, 3, 15, 3, 5, 1.1, 0)
        
            ## remap alpha according to flow
            remappedFg = cv2.remap(alpha, flow[:, :, 0]+allXs, flow[:, :, 1]+allYs, cv2.INTER_LINEAR)
            ## get patch
            remappedFgPatch = remappedFg[offset[1]:offset[1]+patchSize[0], offset[0]:offset[0]+patchSize[1]]
            remappedFgPatch = cv2.GaussianBlur(cv2.morphologyEx(remappedFgPatch, cv2.MORPH_DILATE, 
                                                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (prevMaskDilate, prevMaskDilate))), 
                                               (prevMaskBlurSize, prevMaskBlurSize), prevMaskBlurSigma)

            spritePrior = (1.0-prevMaskImportance)*spritePrior + prevMaskImportance*(-np.log((remappedFgPatch+0.01)/np.sum(remappedFgPatch+0.01)))
    
    
    if useDiffPatch :
        ## change the background prior to give higher cost for pixels to be classified as background if the difference between bgPatch and spritePatch is high
        diffPatch = np.reshape(vectorisedMinusLogMultiNormal(spritePatch.reshape((np.prod(patchSize), 3)), 
                                                             bgPatch.reshape((np.prod(patchSize), 3)), 
                                                             np.eye(3)*diffPatchMultiplier, True), patchSize)
        bgPrior = (1.0-diffPatchImportance)*bgPrior + diffPatchImportance*diffPatch
        
    
    return bgPrior, spritePrior

# <codecell>

## from https://github.com/fbessho/PyPoi/blob/master/pypoi/poissonblending.py
def blend(img_target, img_source, img_mask, offset=(0, 0)):
    # compute regions to be blended
    region_source = (
        max(-offset[0], 0),
        max(-offset[1], 0),
        min(img_target.shape[0] - offset[0], img_source.shape[0]),
        min(img_target.shape[1] - offset[1], img_source.shape[1]))
    region_target = (
        max(offset[0], 0),
        max(offset[1], 0),
        min(img_target.shape[0], img_source.shape[0] + offset[0]),
        min(img_target.shape[1], img_source.shape[1] + offset[1]))
    region_size = (region_source[2] - region_source[0], region_source[3] - region_source[1])

    # clip and normalize mask image
    img_mask = img_mask[region_source[0]:region_source[2], region_source[1]:region_source[3]]
    img_mask[img_mask == 0] = False
    img_mask[img_mask != False] = True

    # create coefficient matrix
    A = scipy.sparse.identity(np.prod(region_size), format='lil')
    for y in range(region_size[0]):
        for x in range(region_size[1]):
            if img_mask[y, x]:
                index = x + y * region_size[1]
                A[index, index] = 4
                if index + 1 < np.prod(region_size):
                    A[index, index + 1] = -1
                if index - 1 >= 0:
                    A[index, index - 1] = -1
                if index + region_size[1] < np.prod(region_size):
                    A[index, index + region_size[1]] = -1
                if index - region_size[1] >= 0:
                    A[index, index - region_size[1]] = -1
    A = A.tocsr()
    
    # create poisson matrix for b
    P = pyamg.gallery.poisson(img_mask.shape)

    startTime = time.time()
    # for each layer (ex. RGB)
    for num_layer in range(img_target.shape[2]):
        # get subimages
        t = img_target[region_target[0]:region_target[2], region_target[1]:region_target[3], num_layer]
        s = img_source[region_source[0]:region_source[2], region_source[1]:region_source[3], num_layer]
        t = t.flatten()
        s = s.flatten()

        # create b
        b = P * s
        for y in range(region_size[0]):
            for x in range(region_size[1]):
                if not img_mask[y, x]:
                    index = x + y * region_size[1]
                    b[index] = t[index]

        # solve Ax = b
        x = pyamg.solve(A, b, verb=False, tol=1e-10)

        # assign x to target image
        x = np.reshape(x, region_size)
        x[x > 255] = 255
        x[x < 0] = 0
        x = np.array(x, img_target.dtype)
        img_target[region_target[0]:region_target[2], region_target[1]:region_target[3], num_layer] = x

    return img_target

# <codecell>

def getPoissonBlended(backgroundLoc, frameLoc, maskLoc, fgOffset) :
#     print "BG", backgroundLoc, "FRAME", frameLoc, "MASK", maskLoc
    im = np.array(Image.open(maskLoc))
    imgSize = im.shape[0:2]

    visiblePixelsGlobalIndices = np.argwhere(im[:, :, -1] != 0)
    topLeftPos = np.min(visiblePixelsGlobalIndices, axis=0)
    patchSize = np.max(visiblePixelsGlobalIndices, axis=0) - topLeftPos + 1
#         topLeftPos = np.copy(preloadedSpritePatches[spriteIdx][frameIdx]['top_left_pos'])
#         patchSize = np.copy(preloadedSpritePatches[spriteIdx][frameIdx]['patch_size'])
#         visiblePixelsGlobalIndices = preloadedSpritePatches[spriteIdx][frameIdx]['visible_indices']+topLeftPos
    
    
    ## when the mask touches the border of the patch there's some weird white halos going on so I enlarge the patch slightly
    ## not sure what happens when the patch goes outside of the bounds of the original image...
    topLeftPos -= 1
    patchSize += 2
    ## make sure we're within bounds
    topLeftPos[np.argwhere(topLeftPos < 0)] = 0
    patchSize[(topLeftPos+patchSize) > imgSize] += (imgSize-(topLeftPos+patchSize))[(topLeftPos+patchSize) > imgSize]


    img_target = np.asarray(Image.open(backgroundLoc))[:, :, 0:3]
    img_target.flags.writeable = True

    img_mask = np.asarray(Image.open(maskLoc))[topLeftPos[0]:topLeftPos[0]+patchSize[0], topLeftPos[1]:topLeftPos[1]+patchSize[1], -1]
    img_mask.flags.writeable = True
    ## make sure that borders of mask are assigned to bg
    img_mask[0, :] = 0; img_mask[-1, :] = 0; img_mask[:, 0] = 0; img_mask[:, -1] = 0

    img_source = np.asarray(Image.open(frameLoc))[topLeftPos[0]:topLeftPos[0]+patchSize[0], topLeftPos[1]:topLeftPos[1]+patchSize[1], :]

#         sourceImg = np.asarray(PIL.Image.open(dataPath+dataSet+spriteName+inputFolderSuffix+"/"+frameName))[topLeftPos[0]:topLeftPos[0]+patchSize[0], 
#                                                                                                             topLeftPos[1]:topLeftPos[1]+patchSize[1], :-1]
#         mask = np.copy(img_mask.reshape((patchSize[0], patchSize[1], 1)))/255.0

#         img_source = np.array(sourceImg*mask + np.asarray(PIL.Image.open(dataPath+dataSet+"median.png"))[topLeftPos[0]:topLeftPos[0]+patchSize[0], 
#                                                                                                          topLeftPos[1]:topLeftPos[1]+patchSize[1], :]*(1.0-mask), dtype=uint8)


    img_source.flags.writeable = True

#     figure(); imshow(np.copy(img_target))
#     figure(); imshow(np.copy(img_source))
#     figure(); imshow(np.copy(img_mask))
#     print fgOffset, offset
    
    img_ret = blend(img_target, img_source, img_mask, offset=(topLeftPos[0]+fgOffset[0], topLeftPos[1]+fgOffset[1]))


    maskedFinal = np.zeros((img_target.shape[0], img_target.shape[1], 4), dtype=np.uint8)
    maskedFinal[visiblePixelsGlobalIndices[:, 0], visiblePixelsGlobalIndices[:, 1], :-1] = img_ret[visiblePixelsGlobalIndices[:, 0]+fgOffset[0], 
                                                                                                   visiblePixelsGlobalIndices[:, 1]+fgOffset[1], :]
    maskedFinal[visiblePixelsGlobalIndices[:, 0], visiblePixelsGlobalIndices[:, 1], -1] = 255

#     PIL.Image.fromarray(np.uint8(maskedFinal)).save(dataPath+dataSet+spriteName+inputFolderSuffix+"-blended/"+frameName)
#         figure(); imshow(maskedFinal)
#     figure(); imshow(img_target)
    result = np.copy(np.uint8(maskedFinal))
#     figure(); imshow(result)
    del img_mask
    del img_source
    del img_target
    del maskedFinal
    del img_ret
    return result

def compareBBoxes(x, y, verbose=False) :
    xBBoxLocs = np.argwhere(x==1)
    yBBoxLocs = np.argwhere(y==1)
    xColRange = [np.min(xBBoxLocs[:, 1]), np.max(xBBoxLocs[:, 1])]
    yColRange = [np.min(yBBoxLocs[:, 1]), np.max(yBBoxLocs[:, 1])]
    if verbose :
        print xColRange, yColRange, len(xBBoxLocs), len(yBBoxLocs)
    
    overlapColRange = list(set(range(xColRange[0], xColRange[1]+1)).intersection(range(yColRange[0], yColRange[1]+1)))
    
    if len(overlapColRange) == 0 :
        if verbose :
            print "returning not overlapping"
        return 0
    overlapColRange = [np.min(overlapColRange), np.max(overlapColRange)]
        
    if verbose :
        print "overlap", overlapColRange
        
    validXBBoxLocs = np.all([xBBoxLocs[:, 1] >= overlapColRange[0], xBBoxLocs[:, 1] <= overlapColRange[1]], axis=0)
    validYBBoxLocs = np.all([yBBoxLocs[:, 1] >= overlapColRange[0], yBBoxLocs[:, 1] <= overlapColRange[1]], axis=0)
    
    if np.max(xBBoxLocs[validXBBoxLocs, 0]) > np.max(yBBoxLocs[validYBBoxLocs, 0]) :
        if verbose :
            print "x > y"
        return 1
    elif np.max(xBBoxLocs[validXBBoxLocs, 0]) < np.max(yBBoxLocs[validYBBoxLocs, 0]) :
        if verbose :
            print "x < y"
        return -1
    else :
        if verbose :
            print "x == y"
        return 0


if len(sys.argv) != 5 :
        print "To run specify parameters: [base/directory/of/synthesised_sequence.npy threshold1 threshold2 doBlendImages]"
        print "values for threshold1 are above 1 and values for threshold2 are between 0 and 1"
        print "values for doBlendImages are 0 if False and 1 if True"
else :
        print sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    
baseLoc = np.string_(sys.argv[1])
doComputePoisson = bool(sys.argv[4])

if doComputePoisson :
    if not os.path.isdir(baseLoc+"blended/") :
        os.mkdir(baseLoc+"blended/")

    synthSeq = np.load(baseLoc+"synthesised_sequence.npy").item()
    bgLoc = synthSeq[DICT_SEQUENCE_BG]

    bgImage = np.array(Image.open(bgLoc))[:, :, 0:3]

    semanticSequences = []
    for usedSeqLoc in synthSeq[DICT_USED_SEQUENCES] :
        semanticSequences.append(np.load(usedSeqLoc).item())

    minFrames = 10000
    for i in xrange(len(synthSeq[DICT_SEQUENCE_INSTANCES])) :
        minFrames = np.min((minFrames, len(synthSeq[DICT_SEQUENCE_INSTANCES][i][DICT_SEQUENCE_FRAMES])))

    maxFramesToRender = 2000


    avgTime = 0.0
    for iterNum, f in enumerate(np.arange(minFrames)[0:]) :
        t = time.time()
        for sIdx, seq in enumerate(synthSeq[DICT_SEQUENCE_INSTANCES][0:]) :
            seq1Idx = seq[DICT_SEQUENCE_IDX]
            frame1Idx = seq[DICT_SEQUENCE_FRAMES][f]

    #         ##### HACK for tetris necessary because I mirrored the sequence #####
    #         oldLength = (len(semanticSequences[seq1Idx][DICT_BBOXES].keys())+1)/2
    #         frame1Idx = np.concatenate((arange(oldLength), arange(oldLength-2, -1, -1)))[frame1Idx]
    #         #####################################################################

            if frame1Idx < len(semanticSequences[seq1Idx][DICT_BBOXES].keys()) :
                frame1Key = np.sort(semanticSequences[seq1Idx][DICT_BBOXES].keys())[frame1Idx]
                frame1Offset = seq[DICT_OFFSET]
                frame1Scale = seq[DICT_SCALE]
            else :
                frame1Key = -1
                frame1Offset = seq[DICT_OFFSET]
                frame1Scale = seq[DICT_SCALE]
                
            if not os.path.isfile(baseLoc+"blended/frame-{0:05}-".format(frame1Key) + "{0:02}.png".format(sIdx)) :

                if frame1Key in semanticSequences[seq1Idx][DICT_FRAMES_LOCATIONS] :

                    spritePatch, offset, patchSize, touchedBorders = getSpritePatch(semanticSequences[seq1Idx], frame1Key, bgImage.shape[1], bgImage.shape[0])

                    sprite1 = getPoissonBlended(bgLoc,
                                                "/".join(semanticSequences[seq1Idx][DICT_MASK_LOCATION].split("/")[:-2])+"/frame-{0:05}.png".format(frame1Key+1),
                                                semanticSequences[seq1Idx][DICT_MASK_LOCATION]+"frame-{0:05}.png".format(frame1Key+1), frame1Offset[::-1])


                    ## dealing with scale and offsets WELL NO SCALE REALLY BUT FUCK IT
                    tmp = np.zeros_like(sprite1)
                    if tmp[offset[1]+frame1Offset[1]:offset[1]+frame1Offset[1]+patchSize[0],
                        offset[0]+frame1Offset[0]:offset[0]+frame1Offset[0]+patchSize[1], :].shape == sprite1[offset[1]:offset[1]+patchSize[0], offset[0]:offset[0]+patchSize[1], :].shape :
                        
                        tmp[offset[1]+frame1Offset[1]:offset[1]+frame1Offset[1]+patchSize[0],
                            offset[0]+frame1Offset[0]:offset[0]+frame1Offset[0]+patchSize[1], :] = sprite1[offset[1]:offset[1]+patchSize[0], offset[0]:offset[0]+patchSize[1], :]
                    sprite1 = np.copy(tmp)
                    del tmp
                    Image.fromarray(sprite1.astype(np.uint8)).save(baseLoc+"blended/frame-{0:05}-".format(frame1Key) + "{0:02}.png".format(sIdx))

        avgTime = (avgTime*iterNum + time.time()-t)/(iterNum+1)
        remainingTime = avgTime*(maxFramesToRender-iterNum-1)/60.0
        sys.stdout.write('\r' + "Done image " + np.string_(iterNum) + " of " + np.string_(maxFramesToRender) +
                         " (avg time: " + np.string_(avgTime) + " secs --- remaining: " +
                         np.string_(int(np.floor(remainingTime))) + ":" + np.string_(int((remainingTime - np.floor(remainingTime))*60)) + ")")
        sys.stdout.flush()
    print 
    print "done"


synthSeq = np.load(baseLoc+"synthesised_sequence.npy").item()
bgLoc = synthSeq[DICT_SEQUENCE_BG]

bgImage = np.array(Image.open(bgLoc))[:, :, 0:3]

semanticSequences = []
for usedSeqLoc in synthSeq[DICT_USED_SEQUENCES] :
    semanticSequences.append(np.load(usedSeqLoc).item())

minFrames = 10000
for i in xrange(len(synthSeq[DICT_SEQUENCE_INSTANCES])) :
    minFrames = np.min((minFrames, len(synthSeq[DICT_SEQUENCE_INSTANCES][i][DICT_SEQUENCE_FRAMES])))


## super mario planes
# thresh = 4.5
# alpha = 0.15

## wave
# thresh = 4.0
# alpha = 0.25

thresh = float(sys.argv[2])
alpha = float(sys.argv[3])


kSize = 15
sigma = 11

usePoisson = bool(sys.argv[4])

maxFramesToRender = 2000


avgTime = 0.0
for iterNum, f in enumerate(np.arange(minFrames)[:maxFramesToRender]) :
    
#     maxYs = []
#     for seqInstance in synthSeq[DICT_SEQUENCE_INSTANCES] :
#         seqIdx = seqInstance[DICT_SEQUENCE_IDX]
#         frameIdx = seqInstance[DICT_SEQUENCE_FRAMES][f]
#         if frameIdx < len(semanticSequences[seqIdx][DICT_BBOXES].keys()) :
#             frameKey = np.sort(semanticSequences[seqIdx][DICT_BBOXES].keys())[frameIdx]
#             maxYs.append(np.max(semanticSequences[seqIdx][DICT_BBOXES][frameKey][:, 1]))
#         else :
#             maxYs.append(0.0)
#     print "sequencesOrder", maxYs, np.argsort(maxYs)
#     sequencesOrder = np.argsort(maxYs)
    
    renderedBBoxes = []
    instancesToRender = []
    for i, seqInstance in enumerate(synthSeq[DICT_SEQUENCE_INSTANCES]) :
        seqIdx = seqInstance[DICT_SEQUENCE_IDX]
        frameIdx = seqInstance[DICT_SEQUENCE_FRAMES][f]
        if frameIdx >= 0 and frameIdx < len(semanticSequences[seqIdx][DICT_FRAMES_LOCATIONS].keys()) :
            frameKey = np.sort(semanticSequences[seqIdx][DICT_FRAMES_LOCATIONS].keys())[frameIdx]
            if frameKey in semanticSequences[seqIdx][DICT_BBOXES].keys() :
                img = np.zeros((bgImage.shape[0], bgImage.shape[1]), np.uint8)
                cv2.fillConvexPoly(img, semanticSequences[seqIdx][DICT_BBOXES][frameKey].astype(int)[[0, 1, 2, 3, 0], :], 1)
                renderedBBoxes.append(np.copy(img))
                instancesToRender.append(i)
#                 gwv.showCustomGraph(img)
                
    sequencesOrder = [i[0] for i in sorted(enumerate(renderedBBoxes), key=lambda x:x[1], cmp=compareBBoxes)]
    sequencesOrder = np.array(instancesToRender)[sequencesOrder]
#     print sequencesOrder
    
    t = time.time()
    if len(sequencesOrder) > 0 :
    #     outputFrame = np.zeros((bgImage.shape[0], bgImage.shape[1], 4), np.uint8)
        seq1Idx = synthSeq[DICT_SEQUENCE_INSTANCES][sequencesOrder[0]][DICT_SEQUENCE_IDX]
        frame1Idx = synthSeq[DICT_SEQUENCE_INSTANCES][sequencesOrder[0]][DICT_SEQUENCE_FRAMES][f]    

        ##### HACK for tetris necessary because I mirrored the sequence #####
    #     oldLength = (len(semanticSequences[seq1Idx][DICT_BBOXES].keys())+1)/2
    # #     print "la", frame1Idx, np.concatenate((arange(oldLength), arange(oldLength-2, -1, -1))),
    #     frame1Idx = np.concatenate((arange(oldLength), arange(oldLength-2, -1, -1)))[frame1Idx]
    # #     print frame1Idx
        #####################################################################

        if frame1Idx >= 0 and frame1Idx < len(semanticSequences[seq1Idx][DICT_FRAMES_LOCATIONS].keys()) :
            frame1Key = np.sort(semanticSequences[seq1Idx][DICT_FRAMES_LOCATIONS].keys())[frame1Idx]
            if frame1Key not in semanticSequences[seq1Idx][DICT_BBOXES].keys() :
                frame1Key = -1
        else :
            frame1Key = -1

        frame1Offset = synthSeq[DICT_SEQUENCE_INSTANCES][sequencesOrder[0]][DICT_OFFSET]
        frame1Scale = synthSeq[DICT_SEQUENCE_INSTANCES][sequencesOrder[0]][DICT_SCALE]

        if frame1Key in semanticSequences[seq1Idx][DICT_FRAMES_LOCATIONS] :

            spritePatch, offset, patchSize, touchedBorders = getSpritePatch(semanticSequences[seq1Idx], frame1Key, bgImage.shape[1], bgImage.shape[0])
            bgPrior, spritePrior = getPatchPriors(bgImage[offset[1]:offset[1]+patchSize[0], offset[0]:offset[0]+patchSize[1], :], 
                                                  spritePatch, offset, patchSize, semanticSequences[seq1Idx], frame1Key)
            fullSprite1Prior = np.zeros(bgImage.shape[0:2])
            fullSprite1Prior[offset[1]:offset[1]+patchSize[0], offset[0]:offset[0]+patchSize[1]] = np.copy(1.0-spritePrior/np.max(spritePrior))
            fullSprite1Prior /= np.max(fullSprite1Prior)

            ## dealing with scale and offsets WELL NO SCALE REALLY BUT FUCK IT
            hasLoaded = False
            if usePoisson :
                if semanticSequences[seq1Idx][DICT_MASK_LOCATION]+"blended/frame-{0:05}.png".format(frame1Key+1) :
                    sprite1 = np.array(Image.open(semanticSequences[seq1Idx][DICT_MASK_LOCATION]+"blended/frame-{0:05}.png".format(frame1Key+1)))
                else :
                    if os.path.isfile(baseLoc+"blended/frame-{0:05}-".format(frame1Key) + "{0:02}.png".format(sequencesOrder[0])) :
                        sprite1 = np.array(Image.open(baseLoc+"blended/frame-{0:05}-".format(frame1Key) + "{0:02}.png".format(sequencesOrder[0])))
        #                 print "loading", 0
                        hasLoaded = True
                    else :
                        sprite1 = getPoissonBlended(bgLoc,
                                                    "/".join(semanticSequences[seq1Idx][DICT_MASK_LOCATION].split("/")[:-2])+"/frame-{0:05}.png".format(frame1Key+1),
                                                    semanticSequences[seq1Idx][DICT_MASK_LOCATION]+"frame-{0:05}.png".format(frame1Key+1), frame1Offset[::-1])
            else :
                sprite1 = np.array(Image.open(semanticSequences[seq1Idx][DICT_MASK_LOCATION]+"frame-{0:05}.png".format(frame1Key+1)))

            outputFrame = sprite1
            outputPrior = fullSprite1Prior

            if not hasLoaded :
                tmp = np.zeros_like(outputFrame)
                tmp[offset[1]+frame1Offset[1]:offset[1]+frame1Offset[1]+patchSize[0],
                    offset[0]+frame1Offset[0]:offset[0]+frame1Offset[0]+patchSize[1], :] = outputFrame[offset[1]:offset[1]+patchSize[0], offset[0]:offset[0]+patchSize[1], :]
                outputFrame = np.copy(tmp)
                del tmp

            tmp = np.zeros_like(outputPrior)
            tmp[offset[1]+frame1Offset[1]:offset[1]+frame1Offset[1]+patchSize[0],
                offset[0]+frame1Offset[0]:offset[0]+frame1Offset[0]+patchSize[1]] = outputPrior[offset[1]:offset[1]+patchSize[0], offset[0]:offset[0]+patchSize[1]]
            outputPrior = np.copy(tmp)
            del tmp

        else :
            outputFrame = np.zeros((bgImage.shape[0], bgImage.shape[1], 4), np.uint8)
            outputPrior = np.zeros(bgImage.shape[0:2])

        tmpAssignment = np.zeros(bgImage.shape[0:2])

    #     for sIdx, seq in zip(arange(1, len(synthSeq[DICT_SEQUENCE_INSTANCES])), synthSeq[DICT_SEQUENCE_INSTANCES][1:]) :
        for sIdx in sequencesOrder[1:] : #[1, 0] : #[1, 2, 3, 14, 13, 4, 5, 6, 7, 8, 9, 10, 11, 12] :
            seq = synthSeq[DICT_SEQUENCE_INSTANCES][sIdx]

            seq2Idx = seq[DICT_SEQUENCE_IDX]
            frame2Idx = seq[DICT_SEQUENCE_FRAMES][f]
            ##### HACK for tetris necessary because I mirrored the sequence #####
    #         oldLength = (len(semanticSequences[seq2Idx][DICT_BBOXES].keys())+1)/2
    # #         print "la", frame2Idx, np.concatenate((arange(oldLength), arange(oldLength-2, -1, -1))),
    #         frame2Idx = np.concatenate((arange(oldLength), arange(oldLength-2, -1, -1)))[frame2Idx]
    # #         print frame2Idx
            #####################################################################

            if frame2Idx >= 0 and frame2Idx < len(semanticSequences[seq2Idx][DICT_FRAMES_LOCATIONS].keys()) :
                frame2Key = np.sort(semanticSequences[seq2Idx][DICT_FRAMES_LOCATIONS].keys())[frame2Idx]
                if frame2Key not in semanticSequences[seq2Idx][DICT_BBOXES].keys() :
                    frame2Key = -1
            else :
                frame2Key = -1

            frame2Offset = seq[DICT_OFFSET]
            frame2Scale = seq[DICT_SCALE]

            if frame2Key in semanticSequences[seq2Idx][DICT_FRAMES_LOCATIONS] :

    #             print frame2Key, semanticSequences[seq2Idx][DICT_MASK_LOCATION]+"frame-{0:05}.png".format(frame2Key+1), frame2Offset, frame2Scale
    #             print 


                spritePatch, offset, patchSize, touchedBorders = getSpritePatch(semanticSequences[seq2Idx], frame2Key, bgImage.shape[1], bgImage.shape[0])
                bgPrior, spritePrior = getPatchPriors(bgImage[offset[1]:offset[1]+patchSize[0], offset[0]:offset[0]+patchSize[1], :], 
                                                      spritePatch, offset, patchSize, semanticSequences[seq2Idx], frame2Key)
                fullSprite2Prior = np.zeros(bgImage.shape[0:2])
                fullSprite2Prior[offset[1]:offset[1]+patchSize[0], offset[0]:offset[0]+patchSize[1]] = np.copy(1.0-spritePrior/np.max(spritePrior))
                fullSprite2Prior /= np.max(fullSprite2Prior)

                hasLoaded = False
                if usePoisson :
                    if semanticSequences[seq2Idx][DICT_MASK_LOCATION]+"blended/frame-{0:05}.png".format(frame2Key+1) :
                        sprite2 = np.array(Image.open(semanticSequences[seq2Idx][DICT_MASK_LOCATION]+"blended/frame-{0:05}.png".format(frame2Key+1)))
                    else :
                        if os.path.isfile(baseLoc+"blended/frame-{0:05}-".format(frame2Key) + "{0:02}.png".format(sIdx)) :
                            sprite2 = np.array(Image.open(baseLoc+"blended/frame-{0:05}-".format(frame2Key) + "{0:02}.png".format(sIdx)))
        #                     print "loading", sIdx; sys.stdout.flush()
                            hasLoaded = True
                        else :
                            sprite2 = getPoissonBlended(bgLoc,
                                                        "/".join(semanticSequences[seq2Idx][DICT_MASK_LOCATION].split("/")[:-2])+"/frame-{0:05}.png".format(frame2Key+1),
                                                        semanticSequences[seq2Idx][DICT_MASK_LOCATION]+"frame-{0:05}.png".format(frame2Key+1), frame2Offset[::-1])
                else :
                    sprite2 = np.array(Image.open(semanticSequences[seq2Idx][DICT_MASK_LOCATION]+"frame-{0:05}.png".format(frame2Key+1)))

                ## dealing with scale and offsets

                if not hasLoaded :
                    tmp = np.zeros_like(sprite2)
                    tmp[offset[1]+frame2Offset[1]:offset[1]+frame2Offset[1]+patchSize[0],
                        offset[0]+frame2Offset[0]:offset[0]+frame2Offset[0]+patchSize[1], :] = sprite2[offset[1]:offset[1]+patchSize[0], offset[0]:offset[0]+patchSize[1], :]
                    sprite2 = np.copy(tmp)
                    del tmp

                tmp = np.zeros_like(fullSprite2Prior)
                tmp[offset[1]+frame2Offset[1]:offset[1]+frame2Offset[1]+patchSize[0],
                    offset[0]+frame2Offset[0]:offset[0]+frame2Offset[0]+patchSize[1]] = fullSprite2Prior[offset[1]:offset[1]+patchSize[0], offset[0]:offset[0]+patchSize[1]]
                fullSprite2Prior = np.copy(tmp)
                del tmp


    #             tmp = np.zeros_like(sprite2)
    #             tmp[frame1Offset[1]:, frame1Offset[0]:, :] = sprite2[:tmp.shape[0]-frame2Offset[1], :tmp.shape[1]-frame2Offset[0], :]
    #             sprite2 = np.copy(tmp)
    #             del tmp

    #             tmp = np.zeros_like(fullSprite2Prior)
    #             tmp[frame1Offset[1]:, frame1Offset[0]:] = fullSprite2Prior[:tmp.shape[0]-frame2Offset[1], :tmp.shape[1]-frame2Offset[0]]
    #             fullSprite2Prior = np.copy(tmp)
    #             del tmp


                ## only doing the checks for the overlapping pixels
                ambiguousIdxs = np.argwhere(np.all(((outputFrame[:, :, -1] != 0).reshape((outputFrame.shape[0], outputFrame.shape[1], 1)),
                                                    (sprite2[:, :, -1] != 0).reshape((sprite2.shape[0], sprite2.shape[1], 1))), axis=0)[:, :, -1])

                ## get background differences
                outputBgDiff = np.zeros(bgImage.shape[0:2])
                outputBgDiff[ambiguousIdxs[:, 0], ambiguousIdxs[:, 1]] = np.sqrt(np.sum((bgImage[ambiguousIdxs[:, 0], ambiguousIdxs[:, 1], :]-
                                                                                         outputFrame[ambiguousIdxs[:, 0], ambiguousIdxs[:, 1], :-1])**2, axis=-1))
                outputBgDiff = cv2.GaussianBlur(outputBgDiff, (kSize, kSize), sigma)#*1.2
                outputBgDiff = outputBgDiff*alpha+(1.0-alpha)*outputPrior


                diffSprite2 = np.zeros(bgImage.shape[0:2])
                diffSprite2[ambiguousIdxs[:, 0], ambiguousIdxs[:, 1]] = np.sqrt(np.sum((bgImage[ambiguousIdxs[:, 0], ambiguousIdxs[:, 1], :]-
                                                                                        sprite2[ambiguousIdxs[:, 0], ambiguousIdxs[:, 1], :-1])**2, axis=-1))
                diffSprite2 = cv2.GaussianBlur(diffSprite2, (kSize, kSize), sigma)#*1.2
                diffSprite2 = diffSprite2*alpha+(1.0-alpha)*fullSprite2Prior



                compositedImage = np.copy(outputFrame)*(outputFrame[:, :, -1].reshape((bgImage.shape[0], bgImage.shape[1], 1))/255.0)
                compositedImage = (compositedImage*(1.0-sprite2[:, :, -1].reshape((bgImage.shape[0], bgImage.shape[1], 1))/255.0) + 
                                   np.copy(sprite2)*(sprite2[:, :, -1].reshape((bgImage.shape[0], bgImage.shape[1], 1))/255.0))
                compositedImage = np.array(compositedImage, dtype=np.uint8)

                ## do the checks and update the composited image
                for (i, j) in ambiguousIdxs :
                    if diffSprite2[i, j] > thresh :
                        compositedImage[i, j, :] = sprite2[i, j, :]
                        tmpAssignment[i, j] = 1
                    elif outputBgDiff[i, j] > thresh :
                        compositedImage[i, j, :] = outputFrame[i, j, :]
                        tmpAssignment[i, j] = 2
                    elif outputBgDiff[i, j] > diffSprite2[i, j] :
                        compositedImage[i, j, :] = outputFrame[i, j, :]
                        tmpAssignment[i, j] = 2
                    else :
                        compositedImage[i, j, :] = sprite2[i, j, :]
                        tmpAssignment[i, j] = 1

                outputFrame = compositedImage
                outputPrior += fullSprite2Prior
                outputPrior /= np.max(outputPrior)
                
    else :
        outputFrame = np.zeros((bgImage.shape[0], bgImage.shape[1], 4), np.uint8)
        
    Image.fromarray(outputFrame).save(baseLoc+"frame-{0:05}.png".format(f+1 ))
#     figure(); imshow(outputFrame)
#     figure(); imshow(np.array(Image.open(baseLoc+"frame-{0:05}.png".format(f+1 ))))
    
    avgTime = (avgTime*iterNum + time.time()-t)/(iterNum+1)
    remainingTime = avgTime*(maxFramesToRender-iterNum-1)/60.0
    sys.stdout.write('\r' + "Done image " + np.string_(iterNum) + " of " + np.string_(maxFramesToRender) +
                     " (avg time: " + np.string_(avgTime) + " secs --- remaining: " +
                     np.string_(int(np.floor(remainingTime))) + ":" + np.string_(int((remainingTime - np.floor(remainingTime))*60)) + ")")
    sys.stdout.flush()
print 
print "done"


