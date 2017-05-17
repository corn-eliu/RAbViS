
# coding: utf-8

# In[1]:

# Imports and defines
import numpy as np
import sys

import cv2
import time
import os
import glob

import opengm
import networkx
import datetime

import matplotlib as mpl

from PIL import Image
from PySide import QtCore, QtGui
# from PySide.phonon import Phonon
import pyglet

import VideoTexturesUtils as vtu
from SemanticsDefinitionTabGUI import computeTransitionMatrix

DICT_SEQUENCE_NAME = 'semantic_sequence_name'
DICT_BBOXES = 'bboxes'
DICT_FOOTPRINTS = 'footprints' ## same as bboxes but it indicates the footprint of the tracked object on the ground plane
DICT_BBOX_ROTATIONS = 'bbox_rotations'
DICT_BBOX_CENTERS = 'bbox_centers'
DICT_FRAMES_LOCATIONS = 'frame_locs'
DICT_MASK_LOCATION = 'frame_masks_location'
DICT_SEQUENCE_FRAMES = 'sequence_frames'
DICT_SEQUENCE_IDX = 'semantic_sequence_idx' # index of the instantiated sem sequence in the list of all used sem sequences for a synthesised sequence
DICT_DESIRED_SEMANTICS = 'desired_semantics' # stores what the desired semantics are for a certain semantic sequence 
#(I could index them by the frame when the toggle happened instead of using the below but maybe ordering is important and I would lose that using a dict)
DICT_FRAME_SEMANTIC_TOGGLE = 'frame_semantic_toggle'# stores the frame index in the generated sequence when the desired semantics have changed
DICT_ICON_TOP_LEFT = "icon_top_left"
DICT_ICON_FRAME_KEY = "icon_frame_key"
DICT_ICON_SIZE = "icon_size"
DICT_REPRESENTATIVE_COLOR = 'representative_color'
DICT_OFFSET = "instance_offset"
DICT_SCALE = "instance_scale"
DICT_FRAME_SEMANTICS = "semantics_per_frame"
DICT_SEMANTICS_NAMES = "semantics_names"
DICT_NUM_SEMANTICS = "number_of_semantic_classes"
DICT_USED_SEQUENCES = "used_semantic_sequences"
DICT_SEQUENCE_INSTANCES = "sequence_instances"
DICT_SEQUENCE_BG = "sequence_background_image"
DICT_SEQUENCE_LOCATION = "sequence_location"
DICT_PATCHES_LOCATION = "sequence_preloaded_patches_location"
DICT_TRANSITION_COSTS_LOCATION = "sequence_precomputed_transition_costs_location"
# DICT_FRAME_COMPATIBILITY_LABELS = 'compatibiliy_labels_per_frame'
DICT_LABELLED_FRAMES = 'labelled_frames' ## includes the frames labelled for the semantic labels (the first [DICT_FRAME_SEMANTICS].shape[1])
DICT_NUM_EXTRA_FRAMES = 'num_extra_frames' ## same len as DICT_LABELLED_FRAMES
DICT_CONFLICTING_SEQUENCES = 'conflicting_sequences'
DICT_COMPATIBLE_SEQUENCES = 'compatible_sequences'
DICT_DISTANCE_MATRIX_LOCATION = 'sequence_precomputed_distance_matrix_location' ## for label propagation
DICT_COMMAND_TYPE = "issue_command_type"
DICT_COMMAND_TYPE_KEY = 0
DICT_COMMAND_TYPE_COLOR = 1
DICT_COMMAND_BINDING = "issue_command_binding"

GRAPH_MAX_COST = 10000000.0

TL_IDX = 0
TR_IDX = 1
BR_IDX = 2
BL_IDX = 3

DRAW_FIRST_FRAME = 'first_frame'
DRAW_LAST_FRAME = 'last_frame'
DRAW_COLOR = 'color'
LIST_SECTION_SIZE = 60
SLIDER_SELECTED_HEIGHT = 23
SLIDER_NOT_SELECTED_HEIGHT = 9
SLIDER_PADDING = 1
SLIDER_INDICATOR_WIDTH = 4
SLIDER_MIN_HEIGHT = 42

preloadedImages = {}
# sequence = np.load("/media/ilisescu/Data1/PhD/data/toy/semantic_sequence-toy1.npy").item()
# for frameKey in sequence[DICT_FRAMES_LOCATIONS].keys() :
#     preloadedImages[frameKey] = np.array(Image.open(sequence[DICT_FRAMES_LOCATIONS][frameKey]), np.uint8)
    
DO_SAVE_LOGS = False


# In[2]:

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
        return -(n+ps)
    else :
        return -ps
# s = time.time()
# vectorisedMinusLogMultiNormal(semanticDist.reshape((len(semanticDist), 1)), np.array([0.0]).reshape((1, 1)), np.array([0.0001]).reshape((1, 1)), True)
# print time.time() - s
# s = time.time()
# vectorisedMinusLogMultiNormal(semanticLabels, np.array(desiredLabel).reshape((1, 2)), np.eye(2)*0.0001, True)

def vectorisedMinusLogMultiNormalMultipleMeans(dataPoints, means, var, normalized = True) :
    D = float(dataPoints.shape[1])
    n = -0.5*np.log(np.linalg.det(var))-(D/2.0)*np.log(2.0*np.pi)

    ## this does 0.5*dot(dot(data-mean, varInv), data-mean)
    varInv = np.linalg.inv(var)
    dataMinusMean = dataPoints.reshape((1, len(dataPoints), dataPoints.shape[1]))-means.reshape((means.shape[0], 1, means.shape[1]))

    ps = np.zeros((means.shape[0], dataPoints.shape[0], int(D)))
    
    for i in xrange(int(D)) :
        ps[:, :, i] = np.sum(dataMinusMean*varInv[:, i], axis=-1)

    ps = -0.5*np.sum(ps*(dataMinusMean), axis=-1)
    
    if normalized :
        return -(n+ps)
    else :
        return -ps
    
def vectorisedMultiNormalMultipleMeans(dataPoints, means, var, normalized = True) :
    D = float(dataPoints.shape[1])
    n = (1/(np.power(2.0*np.pi, D/2.0)*np.sqrt(np.linalg.det(var))))

    ## this does 0.5*dot(dot(data-mean, varInv), data-mean)
    varInv = np.linalg.inv(var)
    dataMinusMean = dataPoints.reshape((1, len(dataPoints), dataPoints.shape[1]))-means.reshape((means.shape[0], 1, means.shape[1]))

    ps = np.zeros((means.shape[0], dataPoints.shape[0], int(D)))
    
    for i in xrange(int(D)) :
        ps[:, :, i] = np.sum(dataMinusMean*varInv[:, i], axis=-1)

    ps = np.exp(-0.5*np.sum(ps*(dataMinusMean), axis=-1))
    
    if normalized :
        return n*ps
    else :
        return ps


# In[3]:

def getMRFCosts(semanticLabels, desiredSemantics, startFrame, sequenceLength) :
    """Computes the unary and pairwise costs for a given sprite
    
        \t  semanticLabels   : the semantic labels assigned to the frames in the sprite sequence
        \t  desiredSemantics : the desired label combination
        \t  startFrame       : starting frame for given sprite (used to constrain which frame to start from)
        \t  sequenceLength   : length of sequence to produce (i.e. number of variables to assign a label k \belongs [0, N] where N is number of frames for sprite)
           
        return: unaries  = unary costs for each node in the graph
                pairwise = pairwise costs for each edge in the graph"""
    
    maxCost = GRAPH_MAX_COST
    ## k = num of semantic labels as there should be semantics attached to each frame
    k = len(semanticLabels)
    
    ## unaries are dictated by semantic labels and by startFrame
    
    # start with uniform distribution for likelihood
    likelihood = np.ones((k, sequenceLength))/(k*sequenceLength)
    
#     # set probability of start frame to 1 and renormalize
#     if startFrame >= 0 and startFrame < k :
#         likelihood[startFrame, 0] = 1.0
#         likelihood /= np.sum(likelihood)
    
    # get the costs associated to agreement of the assigned labels to the desired semantics
    # the variance should maybe depend on k so that when there are more frames in a sprite, the variance is higher so that even if I have to follow the timeline for a long time
    # the cost deriveing from the unary cost does not become bigger than the single pairwise cost to break to go straight to the desired semantic label
    # but for now the sprite sequences are not that long and I'm not expecting them to be many orders of magnitude longer 
    # (variance would have to be 5 or 6 orders of magnitude smaller to make breaking the timeline cheaper than following it)
    distVariance = 0.001#0.001
    numSemantics = semanticLabels.shape[-1]
#     semanticsCosts = vectorisedMinusLogMultiNormal(semanticLabels, np.array(desiredSemantics).reshape((1, numSemantics)), np.eye(numSemantics)*distVariance, True)
    semanticsCosts = np.zeros((k, desiredSemantics.shape[0]))
    for i in xrange(desiredSemantics.shape[0]) :
        semanticsCosts[:, i] = vectorisedMinusLogMultiNormal(semanticLabels, desiredSemantics[i, :].reshape((1, numSemantics)), np.eye(numSemantics)*distVariance, True)
    
    if desiredSemantics.shape[0] < sequenceLength :
        semanticsCosts = semanticsCosts.reshape((k, 1)).repeat(sequenceLength, axis=-1)
    
    # set unaries to minus log of the likelihood + minus log of the semantic labels' distance to the 
    unaries = -np.log(likelihood) + semanticsCosts#.reshape((k, 1)).repeat(sequenceLength, axis=-1)
#     unaries = semanticsCosts.reshape((k, 1)).repeat(sequenceLength, axis=-1)
    
# #     # set cost of start frame to 0 NOTE: not sure if I should use this or the above with the renormalization
#     if startFrame >= 0 and startFrame < k :
#         unaries[startFrame, 0] = 0.0
    if startFrame >= 0 and startFrame < k :
        unaries[:, 0] = maxCost
        unaries[startFrame, 0] = 0.0
    
    ## pairwise are dictated by time constraint and looping ability (i.e. jump probability)
    
    # first dimension is k_n, second represents k_n-1 and last dimension represents all the edges going from graph column w_n-1 to w_n
    pairwise = np.zeros([k, k, sequenceLength-1])
    
    # to enforce timeline give low cost to edge between w_n-1(k = i) and w_n(k = i+1) which can be achieved using
    # an identity matrix with diagonal shifted down by one because only edges from column i-1 and k = j to column i and k=j+1 are viable
    timeConstraint = np.eye(k, k=-1)
    # also allow the sprite to keep looping on label 0 (i.e. show only sprite frame 0 which is the empty frame) so make edge from w_n-1(k=0) to w_n(k=0) viable
    timeConstraint[0, 0] = 1.0
    # also allow the sprite to keep looping from the last frame if necessary so allow to go 
    # from last column (i.e. edge starts from w_n-1(k=last frame)) to second row because first row represents empty frame (i.e. edge goes to w_n(k=1))
    timeConstraint[1, k-1] = 1.0
    # also allow the sprite to go back to the first frame (i.e. empty frame) so allow a low cost edge 
    # from last column (i.e. edge starts from w_n-1(k=last frame)) to first row (i.e. edge goes to w_n(k=0))
    timeConstraint[0, k-1] = 1.0
    
    ## NOTE: don't do all the normal distribution wanking for now: just put very high cost to non viable edges but I'll need something more clever when I try to actually loop a video texture
    ## I would also have to set the time constraint edges' costs to something different from 0 to allow for quicker paths (but more expensive individually) to be chosen when
    ## the semantic label changes
#     timeConstraint /= np.sum(timeConstraint) ## if I normalize here then I need to set mean of gaussian below to what the new max is
#     timeConstraint = vectorisedMinusLogMultiNormal(timeConstraint.reshape((k*k, 1)), np.array([np.max(timeConstraint)]).reshape((1, 1)), np.array([distVariance]).reshape((1, 1)), True)
    timeConstraint = (1.0 - timeConstraint)*maxCost
    
    pairwise = timeConstraint
    
    return unaries.T, pairwise.T


# In[4]:

def smoothstep(delay) :
    # Scale, and clamp x to 0..1 range
    edge0 = 0.0
    edge1 = 1.0
    x = np.arange(0.0, 1.0, 1.0/(delay+1))
    x = np.clip((x - edge0)/(edge1 - edge0), 0.0, 1.0);
    return (x*x*x*(x*(x*6 - 15) + 10))[1:]

def toggleLabelsSmoothly(labels, delay) :
    newLabels = np.roll(labels, 1)
    steps = smoothstep(delay)
    result = np.zeros((delay, labels.shape[-1]))
    ## where diff is less than zero, label prob went from 0 to 1
    result[:, np.argwhere(labels-newLabels < 0)[0, 1]] = steps
    ## where diff is greater than zero, label prob went from 1 to 0
    result[:, np.argwhere(labels-newLabels > 0)[0, 1]] = 1.0 - steps
    return result

def toggleAllLabelsSmoothly(labels, desiredLabel, delay) :
    result = np.zeros((delay, len(labels)))
    for i in xrange(len(labels)) :
        if labels[i] != 0.0 or i == desiredLabel :
            a, b = [labels[i], float(i == desiredLabel)]
            ## mapping interval 0, 1 to a, b
            result[:, i] = (smoothstep(delay))*(b-a) + a
    return result

print toggleLabelsSmoothly(np.array([[1.0,  0.0]]), 8)


# In[5]:

def synchedSequence2FullOverlap(spriteSequences, spritesTotalLength) :
    ## given synched sequences and corresponding sprites sequence lengths, return the full overlapping sequences assuming I'm following 
    ## the sprites' timeline so all this will become a mess as soon as I start looping
    ## or maybe not really as long as the length of the sequence I'm generating is long enough or actually if I'm looping, I would
    ## probably have the opportunity to jump around in the sprite's timeline so maybe there's no problem if the sequence is short
    if spriteSequences.shape[0] < 1 :
        raise Exception("Empty spriteSequences")
        
    if len(np.argwhere(np.any(spriteSequences < 0, axis=0))) == spriteSequences.shape[-1] :
        return None
#         raise Exception("Invalid spriteSequences")
        
    remainingFrames = spritesTotalLength-spriteSequences[:, -1]-1
#     print remainingFrames
        
    fullSequences = np.hstack((spriteSequences, np.zeros((spriteSequences.shape[0], np.max(remainingFrames)), dtype=int)))
    
    for i in xrange(spriteSequences.shape[0]) :
        fullSequences[i, spriteSequences.shape[-1]:] = np.arange(spriteSequences[i, -1]+1, spriteSequences[i, -1]+1+np.max(remainingFrames), dtype=int)
        
    ## get rid of pairs where the frame index is larger than the sprite length
    fullSequences = fullSequences[:, np.ndarray.flatten(np.argwhere(np.all(fullSequences < np.array(spritesTotalLength).reshape(2, 1), axis=0)))]
    
    ## get rid of pairs where the frame index is negative (due to the fact that I'm showing the 0th frame i.e. invisible sprite)
    fullSequences = fullSequences[:, np.ndarray.flatten(np.argwhere(np.all(fullSequences >= 0, axis=0)))]
    
    return fullSequences

# print synchedSequence2FullOverlap(np.vstack((minCostTraversalExistingSprite.reshape((1, len(minCostTraversalExistingSprite)))-1,
#                                              minCostTraversal.reshape((1, len(minCostTraversal)))-1)), spriteTotalLength)


# In[6]:

def aabb2obbDist(aabb, obb, verbose = False) :
    if verbose :
        figure(); plot(aabb[:, 0], aabb[:, 1])
        plot(obb[:, 0], obb[:, 1])
    minDist = 100000000.0
    colors = ['r', 'g', 'b', 'y']
    for i, j in zip(np.arange(4), np.mod(np.arange(1, 5), 4)) :
        m = (obb[j, 1] - obb[i, 1]) / (obb[j, 0] - obb[i, 0])
        b = obb[i, 1] - (m * obb[i, 0]);
        ## project aabb points onto obb segment
        projPoints = np.dot(np.hstack((aabb, np.ones((len(aabb), 1)))), np.array([[1, m, -m*b], [m, m**2, b]]).T)/(m**2+1)
        if np.all(np.negative(np.isnan(projPoints))) :
            ## find distances
            dists = aabb2pointsDist(aabb, projPoints)#np.linalg.norm(projPoints-aabb, axis=-1)
            ## find closest point
            closestPoint = np.argmin(dists)
            ## if rs is between 0 and 1 the point is on the segment
            rs = np.sum((obb[j, :]-obb[i, :])*(aabb-obb[i, :]), axis=1)/(np.linalg.norm(obb[j, :]-obb[i, :])**2)
            if verbose :
                print projPoints
                scatter(projPoints[:, 0], projPoints[:, 1], c=colors[i])
                print dists
                print closestPoint
                print rs
            ## if closestPoint is on the segment
            if rs[closestPoint] > 0.0 and rs[closestPoint] < 1.0 :
#                 print "in", aabb2pointDist(aabb, projPoints[closestPoint, :])
                minDist = np.min((minDist, aabb2pointDist(aabb, projPoints[closestPoint, :])))
            else :
#                 print "out", aabb2pointDist(aabb, obb[i, :]), aabb2pointDist(aabb, obb[j, :])
                minDist = np.min((minDist, aabb2pointDist(aabb, obb[i, :]), aabb2pointDist(aabb, obb[j, :])))

    return minDist


def aabb2pointDist(aabb, point) :
    dx = np.max((np.min(aabb[:, 0]) - point[0], 0, point[0] - np.max(aabb[:, 0])))
    dy = np.max((np.min(aabb[:, 1]) - point[1], 0, point[1] - np.max(aabb[:, 1])))
    return np.sqrt(dx**2 + dy**2);

def aabb2pointsDist(aabb, points) :
    dx = np.max(np.vstack((np.min(aabb[:, 0]) - points[:, 0], np.zeros(len(points)), points[:, 0] - np.max(aabb[:, 0]))), axis=0)
    dy = np.max(np.vstack((np.min(aabb[:, 1]) - points[:, 1], np.zeros(len(points)), points[:, 1] - np.max(aabb[:, 1]))), axis=0)
    return np.sqrt(dx**2 + dy**2);


def getShiftedSpriteTrackDist(firstSprite, secondSprite, shift) :
    
    spriteTotalLength = np.zeros(2, dtype=int)
    spriteTotalLength[0] = len(firstSprite[DICT_BBOX_CENTERS])
    spriteTotalLength[1] = len(secondSprite[DICT_BBOX_CENTERS])
    
    ## find the overlapping sprite subsequences
    ## length of overlap is the minimum between length of the second sequence and length of the first sequence - the advantage it has n the second sequence
    overlapLength = np.min((spriteTotalLength[0]-shift, spriteTotalLength[1]))
    
    frameRanges = np.zeros((2, overlapLength), dtype=int)
    frameRanges[0, :] = np.arange(shift, overlapLength + shift)
    frameRanges[1, :] = np.arange(overlapLength)
    
    totalDistance, distances = getOverlappingSpriteTracksDistance(firstSprite, secondSprite, frameRanges)
    
    return totalDistance, distances, frameRanges


def getOverlappingSpriteTracksDistance(firstSprite, secondSprite, frameRanges, doEarlyOut = True, verbose = False) :
#     ## for now the distance is only given by the distance between bbox center but can add later other things like bbox overlapping region
#     bboxCenters0 = np.array([firstSprite[DICT_BBOX_CENTERS][x] for x in np.sort(firstSprite[DICT_BBOX_CENTERS].keys())[frameRanges[0, :]]])
#     bboxCenters1 = np.array([secondSprite[DICT_BBOX_CENTERS][x] for x in np.sort(secondSprite[DICT_BBOX_CENTERS].keys())[frameRanges[1, :]]])
    
#     centerDistance = np.linalg.norm(bboxCenters0-bboxCenters1, axis=1)
    
#     totDist = np.min(centerDistance)
#     allDists = centerDistance
    
    firstSpriteKeys = np.sort(firstSprite[DICT_BBOX_CENTERS].keys())
    secondSpriteKeys = np.sort(secondSprite[DICT_BBOX_CENTERS].keys())
    allDists = np.zeros(frameRanges.shape[-1])
    for i in xrange(frameRanges.shape[-1]) :            
        allDists[i] = getSpritesBBoxDist(firstSprite[DICT_BBOX_ROTATIONS][firstSpriteKeys[frameRanges[0, i]]],
                                          firstSprite[DICT_BBOXES][firstSpriteKeys[frameRanges[0, i]]], 
                                          secondSprite[DICT_BBOXES][secondSpriteKeys[frameRanges[1, i]]])
        
        if verbose and np.mod(i, frameRanges.shape[-1]/100) == 0 :
            sys.stdout.write('\r' + "Computed image pair " + np.string_(i) + " of " + np.string_(frameRanges.shape[-1]))
            sys.stdout.flush()
        
        ## early out since you can't get lower than 0
        if doEarlyOut and allDists[i] == 0.0 :
            break
            
    totDist = np.min(allDists)
#     return np.sum(centerDistance)/len(centerDistance), centerDistance    
    return totDist, allDists

def getSpritesBBoxDist(theta, bbox1, bbox2, verbose = False) :
    rotMat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    bbox1 = np.dot(rotMat, bbox1.T).T
    bbox2 = np.dot(rotMat, bbox2.T).T
    ## if the bboxes coincide then the distance is set to 0
    if np.all(np.abs(bbox1 - bbox2) <= 10**-10) :
        return 0.0
    else :
        return aabb2obbDist(bbox1, bbox2, verbose)


# In[7]:

def solveMRF(unaries, pairwise) :
    ## build graph
    numLabels = unaries.shape[1]
    chainLength = unaries.shape[0]
    gm = opengm.gm(numpy.ones(chainLength,dtype=opengm.label_type)*numLabels)
    
    # add unary functions
    fids = gm.addFunctions(unaries)
    # add first order factors
    gm.addFactors(fids, np.arange(0, chainLength, 1))
    
    ## add pairwise function
    fid = gm.addFunction(pairwise)
    pairIndices = np.hstack((np.arange(chainLength-1, dtype=int).reshape((chainLength-1, 1)), 
                             np.arange(1, chainLength, dtype=int).reshape((chainLength-1, 1))))
    # add second order factors
    gm.addFactors(fid, pairIndices)
    
    dynProg = opengm.inference.DynamicProgramming(gm)
    tic = time.time()
    dynProg.infer()
    print "bla", time.time() - tic
    
    labels = np.array(dynProg.arg(), dtype=int)
    print gm
    
    return labels, gm.evaluate(labels)

### this is done using matrices
def solveSparseDynProgMRF(unaryCosts, pairwiseCosts, nodesConnectedToLabel) :
    ## assumes unaryCosts has 1 row for each label and 1 col for each variable
    ## assumes arrow heads are rows and arrow tails are cols in pairwiseCosts
    
    ## use the unary and pairwise costs to compute the min cost paths at each node
    # each column represents point n and each row says the index of the k-state that is chosen for the min cost path
    minCostPaths = np.zeros([unaryCosts.shape[0], unaryCosts.shape[1]], dtype=int)
    # contains the min cost to reach a certain state k (i.e. row) for point n (i.e. column)
    minCosts = np.zeros([unaryCosts.shape[0], unaryCosts.shape[1]])
    # the first row of minCosts is just the unary cost
    minCosts[:, 0] = unaryCosts[:, 0]
    minCostPaths[:, 0] = np.arange(0, unaryCosts.shape[0])        
    
    k = unaryCosts.shape[0]
    for n in xrange(1, unaryCosts.shape[1]) :
        costsPerVariableLabelEdge = minCosts[nodesConnectedToLabel, n-1]
        costsPerVariableLabelEdge += pairwiseCosts[np.arange(len(pairwiseCosts)).reshape((len(pairwiseCosts), 1)).repeat(nodesConnectedToLabel.shape[-1], axis=-1), nodesConnectedToLabel]
        costsPerVariableLabelEdge += unaryCosts[:, n].reshape((len(unaryCosts), 1)).repeat(nodesConnectedToLabel.shape[-1], axis=-1)
        minCostsIdxs = np.argmin(costsPerVariableLabelEdge, axis=-1)
        ## minCosts
        minCosts[:, n] = costsPerVariableLabelEdge[np.arange(len(unaryCosts)), minCostsIdxs]
        ## minCostPaths
        minCostPaths[:, n] = nodesConnectedToLabel[np.arange(len(unaryCosts)), minCostsIdxs]
    
    
    ## now find the min cost path starting from the right most n with lowest cost
    minCostTraversal = np.zeros(unaryCosts.shape[1], dtype=np.int)
    ## last node is the node where the right most node with lowest cost
    minCostTraversal[-1] = np.argmin(minCosts[:, -1]) #minCostPaths[np.argmin(minCosts[:, -1]), -1]
    if np.min(minCosts[:, -1]) == np.inf :
        minCostTraversal[-1] = np.floor((unaryCosts.shape[0])/2)
    
    for i in xrange(len(minCostTraversal)-2, -1, -1) :
        minCostTraversal[i] = minCostPaths[ minCostTraversal[i+1], i+1]
        
    return minCostTraversal, np.min(minCosts[:, -1])


# In[8]:

class SemanticsSlider(QtGui.QSlider) :
    def __init__(self, orientation=QtCore.Qt.Horizontal, parent=None) :
        super(SemanticsSlider, self).__init__(orientation, parent)
        style = "QSlider::handle:horizontal { background: #cccccc; width: 0; border-radius: 0px; } "
        style += "QSlider::groove:horizontal { background: #dddddd; } "
        self.setStyleSheet(style)
        
        self.semanticsToDraw = []
        self.selectedSemantics = []
        self.backgroundImage = QtGui.QImage(10, 10, QtGui.QImage.Format_ARGB32)
        
    def setSelectedSemantics(self, selectedSemantics) :
        self.selectedSemantics = selectedSemantics
        
        self.updateHeight()
        self.updateBackgroundImage()
        self.update()
        
    def setSemanticsToDraw(self, semanticsToDraw) :
        self.semanticsToDraw = semanticsToDraw
        
        self.updateHeight()
        self.updateBackgroundImage()
        self.update()
        
    def updateHeight(self) :
        ## reset height
        selectionHeight = len(self.selectedSemantics)*SLIDER_SELECTED_HEIGHT
        remainingHeight = (len(self.semanticsToDraw)-len(self.selectedSemantics))*SLIDER_NOT_SELECTED_HEIGHT
        paddingHeight = (len(self.semanticsToDraw))*2*SLIDER_PADDING
        
        desiredHeight = np.max((SLIDER_MIN_HEIGHT, selectionHeight+remainingHeight+paddingHeight))
        
        self.setFixedHeight(desiredHeight)
        
    def updateBackgroundImage(self) :
        ## re-render background
        if self.backgroundImage.size() != self.size() :
            self.backgroundImage = self.backgroundImage.scaled(self.size())
            
        self.backgroundImage.fill(QtGui.QColor.fromRgb(0, 0, 0, 0))
        painter = QtGui.QPainter(self.backgroundImage)
        
        ## draw semantics
        clrs = np.arange(0.0, 1.0 + 1.0/15.0, 1.0/15.0)
        clrs = mpl.cm.Set1(clrs, bytes=True)
        
        yCoord = SLIDER_PADDING
        for i in xrange(len(self.semanticsToDraw)) :
            desiredSemantics = self.semanticsToDraw[i]
            ## make the color bar representing the requested semantics
            ## getting the colors from clrs for each frame, alpha blended based on desired semantics
            semanticsColorBar = desiredSemantics.reshape((desiredSemantics.shape[0], desiredSemantics.shape[1], 1))*clrs[:desiredSemantics.shape[1], :]
            ## summing up the colors premultiplied by their alpha value
            semanticsColorBar = np.sum(semanticsColorBar, axis=1).astype(np.uint8)[:, :3].reshape((1, len(desiredSemantics), 3))
            
            colorBarHeight = SLIDER_NOT_SELECTED_HEIGHT
            if i in self.selectedSemantics :
                colorBarHeight = SLIDER_SELECTED_HEIGHT
            ## making color bar taller
            semanticsColorBar = np.ascontiguousarray(np.repeat(semanticsColorBar, colorBarHeight, axis=0))
            
            colorBarImage = QtGui.QImage(semanticsColorBar.data, semanticsColorBar.shape[1], semanticsColorBar.shape[0], semanticsColorBar.strides[0], QtGui.QImage.Format_RGB888)
            painter.drawImage(0, yCoord, colorBarImage.scaled(self.width()*float(desiredSemantics.shape[0])/float(self.maximum()+1), colorBarHeight))
            
            
            yCoord += (colorBarHeight+2*SLIDER_PADDING)
        
        painter.end()
        
        
    def resizeEvent(self, event) :
        self.updateBackgroundImage()
        
    def mousePressEvent(self, event) :
        if event.button() == QtCore.Qt.LeftButton :
            self.setValue(event.pos().x()*(float(self.maximum())/self.width()))
        
    def paintEvent(self, event) :
        super(SemanticsSlider, self).paintEvent(event)
        
        painter = QtGui.QPainter(self)
        
        painter.drawImage(0, 0, self.backgroundImage)

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


# In[ ]:

class IndexedCheckBox(QtGui.QCheckBox) :
    indexedCheckBoxStateChanged = QtCore.Signal(int)
    
    def __init__(self, index, parent=None) :
        super(IndexedCheckBox, self).__init__(parent)
        self.index = index
        
        self.stateChanged.connect(self.emitStateChanged)
    
    def emitStateChanged(self) :
        self.indexedCheckBoxStateChanged.emit(self.index)

class InstancesShowWidget(QtGui.QWidget) :
    instanceDoShowChanged = QtCore.Signal(int)
    
    def __init__(self, parent=None):
        super(InstancesShowWidget, self).__init__(parent)
        
        self.setFixedWidth(30)
        
        self.instancesToDraw = []
        self.selectedInstances = []
        self.checkboxWidgets = []
        self.backgroundImage = QtGui.QImage(10, 10, QtGui.QImage.Format_ARGB32)
        
        self.font = QtGui.QFont()
        
    def setSelectedInstances(self, selectedInstances) :
        self.selectedInstances = selectedInstances
        
        for i in xrange(len(self.instancesToDraw)) :
            if i in self.selectedInstances :
                self.layout().itemAt(i).widget().setFixedHeight(SLIDER_SELECTED_HEIGHT)
            else :
                self.layout().itemAt(i).widget().setFixedHeight(SLIDER_NOT_SELECTED_HEIGHT)
        
        self.updateHeight()
        self.updateBackgroundImage()
        self.update()
        
    def showInstanceChanged(self, index) :
        print "INSTANCE", index, "SHOW CHANGED"; sys.stdout.flush()
        self.instanceDoShowChanged.emit(index)
        
    def setInstancesToDraw(self, instancesToDraw, instancesDoShow) :
        self.instancesToDraw = instancesToDraw
        
        if self.layout() != None :
            child = self.layout().takeAt(0)
            while child:
                if child != None and child.widget() != None :
                    child.widget().deleteLater()
                child = self.layout().takeAt(0)
                
        if self.layout() == None :
            self.setLayout(QtGui.QVBoxLayout())
            
        self.layout().setSpacing(SLIDER_PADDING*2)
        self.layout().setContentsMargins(self.width()*(1.0/3.0), SLIDER_PADDING, 0, SLIDER_PADDING)

        for i in xrange(len(instancesToDraw)) :

            checkbox = IndexedCheckBox(i)
            checkbox.setChecked(instancesDoShow[i])
            checkbox.setStyleSheet("QCheckBox::indicator:hover { background: none; }")
            checkbox.indexedCheckBoxStateChanged[int].connect(self.showInstanceChanged)
            checkbox.setFixedWidth(self.width()*(2.0/3.0))

            if i in self.selectedInstances :
                checkbox.setFixedHeight(SLIDER_SELECTED_HEIGHT)
            else :
                checkbox.setFixedHeight(SLIDER_NOT_SELECTED_HEIGHT)

            self.layout().addWidget(checkbox)
        self.layout().addStretch()
        
        self.updateHeight()
        self.updateBackgroundImage()
        self.update()
        
    def updateHeight(self) :
        selectionHeight = len(self.selectedInstances)*SLIDER_SELECTED_HEIGHT
        remainingHeight = (len(self.instancesToDraw)-len(self.selectedInstances))*SLIDER_NOT_SELECTED_HEIGHT
        paddingHeight = (len(self.instancesToDraw))*2*SLIDER_PADDING
        
        desiredHeight = np.max((SLIDER_MIN_HEIGHT, selectionHeight+remainingHeight+paddingHeight))
        
        self.setFixedHeight(desiredHeight)
        
    def updateBackgroundImage(self) :
        ## re-render background
        if self.backgroundImage.size() != self.size() :
            self.backgroundImage = self.backgroundImage.scaled(self.size())
            
        self.backgroundImage.fill(QtGui.QColor.fromRgb(0, 0, 0, 0))
        
        painter = QtGui.QPainter(self.backgroundImage)
        
        yCoord = SLIDER_PADDING
        for i in xrange(len(self.instancesToDraw)) :
            col = self.instancesToDraw[i]
            
            colorBarHeight = SLIDER_NOT_SELECTED_HEIGHT
            self.font.setBold(False)
            self.font.setPixelSize(10)
            if i in self.selectedInstances :
                colorBarHeight = SLIDER_SELECTED_HEIGHT
                self.font.setBold(True)
                self.font.setPixelSize(11)
                
            painter.setFont(self.font)
                
            
            instanceRect = QtCore.QRect(0, yCoord, self.width(), colorBarHeight)
            textRect = QtCore.QRect(0, yCoord, self.width()*(1.0/3.0), colorBarHeight)
            
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtGui.QBrush(QtGui.QColor.fromRgb(col[0], col[1], col[2], 255)))
            painter.drawRect(instanceRect)
            
            painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(0, 0, 0, 255), 1,
                                          QtCore.Qt.SolidLine, QtCore.Qt.FlatCap, QtCore.Qt.MiterJoin))
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.drawText(textRect, QtCore.Qt.AlignCenter, np.string_(i+1))
            
            
            yCoord += (colorBarHeight+2*SLIDER_PADDING)
        
        painter.end()
        
    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawImage(0, 0, self.backgroundImage)
        painter.end()
        
#         super(InstancesShowWidget, self).paintEvent(event)


# In[ ]:

class SequencesListDelegate(QtGui.QItemDelegate):
    
    def __init__(self, parent=None) :
        super(SequencesListDelegate, self).__init__(parent)
        
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
        
class InstancesListDelegate(QtGui.QItemDelegate):
    
    def __init__(self, parent=None) :
        super(InstancesListDelegate, self).__init__(parent)
        
        self.setBackgroundColor(QtGui.QColor.fromRgb(245, 245, 245))
        
        self.font = QtGui.QFont()
        self.font.setPixelSize(10)

    def setBackgroundColor(self, bgColor) :
        self.bgColor = bgColor
    
    def drawDisplay(self, painter, option, rect, text):
        painter.save()
        
        colorRect = QtCore.QRect(rect.left(), rect.top()+SLIDER_PADDING, rect.width(), rect.height()-2*SLIDER_PADDING)
        selectionRect = rect

        # draw colorRect
        painter.setBrush(QtGui.QBrush(self.bgColor))
        painter.setPen(QtCore.Qt.NoPen)
        painter.drawRect(colorRect)

        ## draw selection
        if option.state & QtGui.QStyle.State_Selected:
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(245, 245, 245), 1, 
                                              QtCore.Qt.SolidLine, QtCore.Qt.SquareCap, QtCore.Qt.MiterJoin))
            painter.drawLine(rect.left(), rect.top(), rect.right(), rect.top())
            painter.drawLine(rect.left(), rect.bottom(), rect.right(), rect.bottom())

        # set text color
        painter.setPen(QtGui.QPen(QtCore.Qt.black))
        if option.state & QtGui.QStyle.State_Selected:
            self.font.setBold(True)
            self.font.setPixelSize(11)
        else :
            self.font.setBold(False)
            self.font.setPixelSize(10)
        painter.setFont(self.font)

        painter.drawText(colorRect, QtCore.Qt.AlignVCenter | QtCore.Qt.AlignCenter, text)

        painter.restore()


# In[ ]:

class ImageLabel(QtGui.QLabel) :
    
    def __init__(self, text, parent=None):
        super(ImageLabel, self).__init__(text, parent)
        
        self.setMouseTracking(True)
        
        self.image = None
        self.qImage = None
        self.overlay = None
        
    def setImage(self, image) :
        if np.any(image != None) :
            self.image = np.ascontiguousarray(image.copy())
            self.qImage = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], self.image.strides[0], QtGui.QImage.Format_RGB888);
            self.update()
        else :
            self.image = None
            self.qImage = None

    def setOverlay(self, overlay) :
        if np.any(overlay != None) :
            self.overlay = overlay.copy()
            self.update()
        else :
            self.overlay = None
        
    def paintEvent(self, event):
        super(ImageLabel, self).paintEvent(event)
        painter = QtGui.QPainter(self)
        if np.any(self.qImage != None) :
            painter.drawImage(QtCore.QPoint(0, 0), self.qImage)
            
        if np.any(self.overlay != None) :
            painter.drawImage(QtCore.QPoint(0, 0), self.overlay)
            
        painter.end()


# In[ ]:

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


# In[ ]:

def makeStackPlot(values, height) :
    stackPlot = np.zeros((height, values.shape[0]))
#     clrs = np.arange(0.0, 1.0+1.0/(values.shape[1]-1), 1.0/(values.shape[1]-1))
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

class InfoDialog(QtGui.QDialog):
    def __init__(self, parent=None, title=""):
        super(InfoDialog, self).__init__(parent)
        
        self.cursorPosition = np.zeros(2)
        self.MAX_VIS_IMAGE_HEIGHT = 200
        self.CURSOR_SIZE = 4 ## actual size is x2
        self.cursorPositionOverlay = QtGui.QImage(QtCore.QSize(100, 100), QtGui.QImage.Format_ARGB32)
        
        self.createGUI()
        
        self.setWindowTitle(title)
        
    def setInfoImage(self, infoImage) :
        self.imageLabel.setPixmap(QtGui.QPixmap.fromImage(infoImage))
        self.setMaximumSize(infoImage.width(), infoImage.height())
        self.cursorPositionOverlay = self.cursorPositionOverlay.scaled(infoImage.size())
        self.imageLabel.setOverlay(self.cursorPositionOverlay)
        self.update()
        
        if False :
            maxHeight = QtGui.QApplication.desktop().screenGeometry().height()-100-22*3
            if infoImage.height() < maxHeight - self.MAX_VIS_IMAGE_HEIGHT :
                self.mainLayout.setDirection(QtGui.QBoxLayout.TopToBottom)
            else :
                self.mainLayout.setDirection(QtGui.QBoxLayout.LeftToRight)

    def setVisImage(self, visImage) :
        self.visLabel.setPixmap(QtGui.QPixmap.fromImage(visImage.scaledToHeight(self.MAX_VIS_IMAGE_HEIGHT)))
        self.update()
        
    def saveClicked(self):
        fileName = QtGui.QFileDialog.getSaveFileName(self, "Save Info Image", os.path.expanduser("~")+"/")[0]
        if fileName != "" :
            print "saved", fileName+".png", self.imageLabel.pixmap().save(fileName+".png")
            
    def setText(self, text) :
        self.infoLabel.setText(text)
        
    def setCursorPosition(self, cursorPosition) :
        self.cursorPosition = cursorPosition
        self.cursorPositionOverlay.fill(QtGui.QColor.fromRgb(0, 0, 0, 0))
        painter = QtGui.QPainter(self.cursorPositionOverlay)
        painter.setRenderHints(QtGui.QPainter.Antialiasing)
        
        ## draw cursor
        painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(0, 0, 0, 255), 3, QtCore.Qt.SolidLine, QtCore.Qt.FlatCap, QtCore.Qt.MiterJoin))
        painter.drawEllipse(QtCore.QPointF(self.cursorPosition[0], self.cursorPosition[1]), self.CURSOR_SIZE, self.CURSOR_SIZE)
        
        painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(255, 255, 255, 255), 1, QtCore.Qt.SolidLine, QtCore.Qt.FlatCap, QtCore.Qt.MiterJoin))
        painter.drawEllipse(QtCore.QPointF(self.cursorPosition[0], self.cursorPosition[1]), self.CURSOR_SIZE, self.CURSOR_SIZE)
        
        painter.end()
        self.imageLabel.setOverlay(self.cursorPositionOverlay)
    
    def createGUI(self):
        
        self.setSizePolicy(QtGui.QSizePolicy.Maximum, QtGui.QSizePolicy.Maximum)
        
        self.imageLabel = ImageLabel("")
        self.imageLabel.setMouseTracking(True)
        self.imageLabel.setSizePolicy(QtGui.QSizePolicy.Maximum, QtGui.QSizePolicy.Maximum)
        
        self.infoLabel = QtGui.QLabel()
        self.infoLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.infoLabel.setSizePolicy(QtGui.QSizePolicy.Maximum, QtGui.QSizePolicy.Maximum)
        
        self.saveButton = QtGui.QPushButton("Save")
        self.saveButton.setSizePolicy(QtGui.QSizePolicy.Maximum, QtGui.QSizePolicy.Maximum)
        
        self.visLabel = QtGui.QLabel()
        self.visLabel.setSizePolicy(QtGui.QSizePolicy.Maximum, QtGui.QSizePolicy.Maximum)
        
        ## SIGNALS ##
        
        self.saveButton.clicked.connect(self.saveClicked)
        
        ## LAYOUTS ##
        hLayout = QtGui.QHBoxLayout()
        hLayout.addStretch()
        hLayout.addWidget(self.imageLabel, QtCore.Qt.AlignCenter)
        hLayout.addStretch()
        hLayout.addWidget(self.visLabel, QtCore.Qt.AlignCenter)
        hLayout.addStretch()

        infoLabelHLayout = QtGui.QHBoxLayout()
        infoLabelHLayout.addStretch()
        infoLabelHLayout.addWidget(self.infoLabel, QtCore.Qt.AlignCenter)
        infoLabelHLayout.addStretch()
        saveButtonHLayout = QtGui.QHBoxLayout()
        saveButtonHLayout.addStretch()
        saveButtonHLayout.addWidget(self.saveButton, QtCore.Qt.AlignCenter)
        saveButtonHLayout.addStretch()
        
        self.mainLayout = QtGui.QBoxLayout(QtGui.QBoxLayout.TopToBottom)
        self.mainLayout.addLayout(hLayout, QtCore.Qt.AlignCenter)
        self.mainLayout.addLayout(infoLabelHLayout, QtCore.Qt.AlignCenter)
        self.mainLayout.addLayout(saveButtonHLayout, QtCore.Qt.AlignCenter)
        
        
        self.setLayout(self.mainLayout)


# In[ ]:

class IndexedComboBox(QtGui.QComboBox) :
    currentIndexChangedSignal = QtCore.Signal(int, int, int)
    
    def __init__(self, rowIndex, colIndex, parent=None):
        super(IndexedComboBox, self).__init__(parent)
        self.rowIndex = rowIndex
        self.colIndex = colIndex
        
        self.currentIndexChanged[int].connect(self.emitCurrentIndexChanged)
        
    def emitCurrentIndexChanged(self, index) :
        self.currentIndexChangedSignal.emit(index, self.rowIndex, self.colIndex)
        
class IndexedPushButton(QtGui.QPushButton) :
#     HERE KEEP TRACK OF THE BINDING FOR THIS BUTTON SO THAT THEN I CAN RETRIEVE IT WHEN THE DIALOG IS CLOSED USING THE OK BUTTON
    clickedSignal = QtCore.Signal(int, int)
    
    def __init__(self, rowIndex, colIndex, commandType, commandBindings, text="", parent=None):
        super(IndexedPushButton, self).__init__(text, parent)
        self.rowIndex = rowIndex
        self.colIndex = colIndex
        self.commandBindings = commandBindings
        self.setCommandBinding(commandType, commandBindings[commandType])
        
        self.clicked.connect(self.emitClicked)
        
#     HAVE A METHOD TO SET THE BINDING GIVEN THE TYPE OF COMMAND AND SET THE STYLE (FOR COLOR TYPE) AND TEXT (FOR KEY TYPE) IN THIS METHOD RATHER THAN IN showKey(Color)Button()
    
    def setCommandBinding(self, commandType, commandBinding) :
        self.commandBindings[commandType] = commandBinding
        self.commandType = commandType
        
        if self.commandType == DICT_COMMAND_TYPE_KEY :
            self.setText("Bind Key [{0}]".format(QtGui.QKeySequence(self.commandBindings[commandType]).toString()))
            self.setStyleSheet("")
        elif self.commandType == DICT_COMMAND_TYPE_COLOR :
            self.setText("Bind Color")
            if np.median(self.commandBindings[commandType]) < 127 :
                textColor = np.ones(3, int)*255
            else :
                textColor = np.zeros(3, int)
            self.setStyleSheet("QPushButton {border: 1px solid black; background-color: rgb("+
                               np.string_(self.commandBindings[commandType][0])+", "+np.string_(self.commandBindings[commandType][1])
                               +", "+np.string_(self.commandBindings[commandType][2])+"); color: rgb("+
                               np.string_(textColor[0])+", "+np.string_(textColor[1])
                               +", "+np.string_(textColor[2])+");}")
            
    
    def emitClicked(self) :
        self.clickedSignal.emit(self.rowIndex, self.colIndex)
    
class KeyBindingDialog(QtGui.QDialog):
    def __init__(self, parent=None, title=""):
        super(KeyBindingDialog, self).__init__(parent)
        self.setWindowTitle(title)
        
        self.keySequence = None
        
        self.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.MinimumExpanding)
        
        mainLayout = QtGui.QVBoxLayout()
        label = QtGui.QLabel("Press a Key Combination")
        label.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignHCenter)
        label.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        mainLayout.addWidget(label)
        self.setLayout(mainLayout)
        
        
    def keyPressEvent(self, e) :
        if e.key() != QtCore.Qt.Key_Shift and e.key() != QtCore.Qt.Key_Control and e.key() != QtCore.Qt.Key_Alt and e.key() != QtCore.Qt.Key_AltGr and e.key() != QtCore.Qt.Key_Meta :
            self.keySequence = e.key()
            if e.modifiers() & QtCore.Qt.Modifier.SHIFT :
                self.keySequence = QtCore.Qt.SHIFT + self.keySequence
            if e.modifiers() & QtCore.Qt.Modifier.ALT :
                self.keySequence = QtCore.Qt.ALT + self.keySequence
            if e.modifiers() & QtCore.Qt.Modifier.CTRL :
                self.keySequence = QtCore.Qt.CTRL + self.keySequence
#             self.keySequence = QtGui.QKeySequence(self.keySequence).toString()
            self.done(1)
    
class ActionsCommandBindingsDialog(QtGui.QDialog):
    def __init__(self, parent=None, title="", semanticSequences=[]):
        super(ActionsCommandBindingsDialog, self).__init__(parent)
        
        self.semanticSequences = semanticSequences
        self.indexedButtonsList = []
        
        self.createGUI()
        
        self.setWindowTitle(title)
        
    def accept(self):
        self.done(1)
        
        print "########################### ACCEPTING KEY BINDINGS ###########################"
        
        idx = 0
        for semanticSequence in self.semanticSequences :
            print "SEQUENCE", semanticSequence[DICT_SEQUENCE_NAME]
            for actionKey in np.sort(semanticSequence[DICT_SEMANTICS_NAMES].keys()) :
                print "\t ACTION", semanticSequence[DICT_SEMANTICS_NAMES][actionKey],
                if self.indexedButtonsList[idx].commandType == DICT_COMMAND_TYPE_COLOR :
                    print "COLOR COMMAND", self.indexedButtonsList[idx].commandBindings[DICT_COMMAND_TYPE_COLOR]
                    semanticSequence[DICT_COMMAND_TYPE][actionKey] = DICT_COMMAND_TYPE_COLOR
                    semanticSequence[DICT_COMMAND_BINDING][actionKey] = np.array(self.indexedButtonsList[idx].commandBindings[DICT_COMMAND_TYPE_COLOR])
                elif self.indexedButtonsList[idx].commandType == DICT_COMMAND_TYPE_KEY :
                    print "KEY COMMAND", QtGui.QKeySequence(self.indexedButtonsList[idx].commandBindings[DICT_COMMAND_TYPE_KEY]).toString()
                    semanticSequence[DICT_COMMAND_TYPE][actionKey] = DICT_COMMAND_TYPE_KEY
                    semanticSequence[DICT_COMMAND_BINDING][actionKey] = self.indexedButtonsList[idx].commandBindings[DICT_COMMAND_TYPE_KEY]
                
                idx += 1
            np.save(semanticSequence[DICT_SEQUENCE_LOCATION], semanticSequence)
            
        print "##############################################################################"
    
    def reject(self):
        self.done(0)
            
    def setCommandBinding(self, rowIndex, colIndex) :
        if self.indexedButtonsList[rowIndex].commandType == DICT_COMMAND_TYPE_COLOR :
            print "BIND COLOR", rowIndex, colIndex
            newColor = QtGui.QColorDialog.getColor(QtGui.QColor(self.indexedButtonsList[rowIndex].commandBindings[DICT_COMMAND_TYPE_COLOR][0],
                                                                self.indexedButtonsList[rowIndex].commandBindings[DICT_COMMAND_TYPE_COLOR][1],
                                                                self.indexedButtonsList[rowIndex].commandBindings[DICT_COMMAND_TYPE_COLOR][2]), self, "Choose Sequence Color")
            if newColor.isValid() :
                self.indexedButtonsList[rowIndex].setCommandBinding(DICT_COMMAND_TYPE_COLOR, np.array([newColor.red(), newColor.green(), newColor.blue()]))
                
        elif self.indexedButtonsList[rowIndex].commandType == DICT_COMMAND_TYPE_KEY :
            print "BIND KEY", rowIndex, colIndex
        
            keyBindingDialog = KeyBindingDialog(self, "Key Binding")
            exitCode = keyBindingDialog.exec_()
            
            if exitCode == 1 and keyBindingDialog.keySequence !=  None :
                self.indexedButtonsList[rowIndex].setCommandBinding(DICT_COMMAND_TYPE_KEY, keyBindingDialog.keySequence)
        
    def changeCommandType(self, index, rowIndex, colIndex) :
        print "COMMAND TYPE", index, rowIndex, colIndex
        self.indexedButtonsList[rowIndex].setCommandBinding(index, self.indexedButtonsList[rowIndex].commandBindings[index])
    
    def createGUI(self):
        self.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.MinimumExpanding)
        
        self.semanticSequencesTable = QtGui.QTableWidget()
        self.semanticSequencesTable.setEditTriggers(QtGui.QAbstractItemView.NoEditTriggers)
        self.semanticSequencesTable.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
        self.semanticSequencesTable.setSelectionMode(QtGui.QAbstractItemView.NoSelection)
        self.semanticSequencesTable.horizontalHeader().setStretchLastSection(True)
        self.semanticSequencesTable.horizontalHeader().setResizeMode(QtGui.QHeaderView.ResizeToContents)
        self.semanticSequencesTable.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.MinimumExpanding)
        self.semanticSequencesTable.verticalHeader().setVisible(False)
        self.semanticSequencesTable.verticalHeader().setDefaultSectionSize(30)
        self.semanticSequencesTable.setMinimumWidth(550)
        
        self.semanticSequencesTable.setColumnCount(4)
        self.semanticSequencesTable.setHorizontalHeaderLabels(["Sequence Name", "Action", "Command Type", "Command Value"])
        numActions = 0
        for semanticSequence in self.semanticSequences :
            numActions += semanticSequence[DICT_NUM_SEMANTICS]
        self.semanticSequencesTable.setMinimumHeight(30*(numActions+1)-2)
        
        self.semanticSequencesTable.setRowCount(numActions)
        
        idx = 0
        for semanticSequence in self.semanticSequences :
            self.semanticSequencesTable.setItem(idx, 0, QtGui.QTableWidgetItem(semanticSequence[DICT_SEQUENCE_NAME]))
            self.semanticSequencesTable.setSpan(idx, 0, semanticSequence[DICT_NUM_SEMANTICS], 1)
            
            for actionKey in np.sort(semanticSequence[DICT_SEMANTICS_NAMES].keys()) :
                self.semanticSequencesTable.setItem(idx, 1, QtGui.QTableWidgetItem(semanticSequence[DICT_SEMANTICS_NAMES][actionKey]))
                combo = IndexedComboBox(idx, 2, self.semanticSequencesTable)
                combo.addItems(["Key", "Color"])
                combo.setCurrentIndex(semanticSequence[DICT_COMMAND_TYPE][actionKey])

                self.semanticSequencesTable.setCellWidget(idx, 2, combo)
                combo.currentIndexChangedSignal[int, int, int].connect(self.changeCommandType)
                
                ## make buttons and shit
                self.indexedButtonsList.append(IndexedPushButton(idx, 3, DICT_COMMAND_TYPE_KEY, [QtCore.Qt.Key_0, np.zeros(3, int)]))
                self.indexedButtonsList[-1].clickedSignal[int, int].connect(self.setCommandBinding)

                self.semanticSequencesTable.setCellWidget(self.indexedButtonsList[-1].rowIndex, self.indexedButtonsList[-1].colIndex, self.indexedButtonsList[-1])
                self.indexedButtonsList[-1].setCommandBinding(semanticSequence[DICT_COMMAND_TYPE][actionKey], semanticSequence[DICT_COMMAND_BINDING][actionKey])
                
                idx += 1
        
        self.buttonBox = QtGui.QDialogButtonBox(QtGui.QDialogButtonBox.Ok | QtGui.QDialogButtonBox.Cancel);
         
        ## SIGNALS ##
        
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        
        ## LAYOUTS ##
        
        mainLayout = QtGui.QVBoxLayout()
        
        mainLayout.addWidget(self.semanticSequencesTable)
        mainLayout.addWidget(self.buttonBox, QtCore.Qt.AlignCenter)
        
        self.setLayout(mainLayout)

def changeActionsCommandBindings(parent=None, title="Dialog", semanticSequences= []) :
    changeActionsDialog = ActionsCommandBindingsDialog(parent, title, semanticSequences)
    exitCode = changeActionsDialog.exec_()
    
    return exitCode


# In[ ]:

def getDistToSemantics(allLabels, label, distVariance) :
    """
        Computes distance of the given labels, allLabels, to a user specified class label, label
    """
    numClasses = allLabels.shape[1]
    labelVector = np.zeros((1, numClasses))
    labelVector[0, label] = 1.0
    
    return vectorisedMultiNormalMultipleMeans(allLabels, labelVector, np.eye(numClasses)*distVariance, False).T

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


# In[ ]:

class SemanticLoopingTab(QtGui.QWidget):
    def __init__(self, extendLength, startSequenceLoc, doShowCompatibilityInfo, parent=None):
        super(SemanticLoopingTab, self).__init__(parent)
        
        self.doShowCompatibilityInfo = doShowCompatibilityInfo
        
        self.playIcon = QtGui.QIcon("play.png")
        self.pauseIcon = QtGui.QIcon("pause.png")
        self.doPlaySequence = False
        
        self.EXTEND_LENGTH = extendLength + 1 ## since I get rid of the frist frame from the generated sequence because it's forced to be the one already showing
        self.PLOT_HEIGHT = 100 ## height of label compatibility plot
        self.TEXT_SIZE = 22 ## height of sprite names in info image
        self.MAX_PLOT_SIZE = 500
         
        self.createGUI()
        
        self.isSequenceLoaded = False
        self.numbersSequenceFrames = []
        self.showNumbersSequence = False
        
        self.taggedConflicts = []
        self.compatibilityMats = {}
        self.sequencePairCompatibilityLabels = {}
        
        self.isScribbling = False
        
        self.prevPoint = None
        self.dataPath = os.path.expanduser("~")+"/"
        self.dataSet = "/"
        
        self.semanticSequences = []
        self.selectedSemSequenceIdx = -1
        self.selectedSequenceInstancesIdxes = []
        self.instancesDoShow = []
        self.infoFrameIdxs = np.array([-1, -1])
        
        self.frameIdx = 0
        self.overlayImg = QtGui.QImage(QtCore.QSize(100, 100), QtGui.QImage.Format_ARGB32)
        self.infoFrameIdxs = np.array([0, 0])
        
        self.bgImageLoc = None
        self.bgImage = None
        self.loadedSynthesisedSequence = None
        self.generatedSequence = []
        self.synthesisedSequence = {}
        self.preloadedPatches = {}
        self.preloadedTransitionCosts = {}
        self.preloadedDistanceMatrices = {}
        
        
        self.playTimer = QtCore.QTimer(self)
        self.playTimer.setInterval(1000/30)
        self.playTimer.timeout.connect(self.renderOneFrame)
        self.lastRenderTime = time.time()
        self.oldInfoText = ""
        
        self.TOGGLE_DELAY = self.toggleDelaySpinBox.value()
        self.BURST_ENTER_DELAY = 2
        self.BURST_EXIT_DELAY = 20
        
        self.DO_EXTEND = 0
        self.DO_TOGGLE = 1
        self.DO_BURST = 2
        
#         pyglet.resource.path = ['/media/ilisescu/Data1/PhD/data/drumming2']
        pyglet.resource.path = ['/media/ilisescu/Data1/PhD/data/toy']
        pyglet.resource.reindex()
        
        self.doPlaySounds = False
        self.mergeActionRequests = self.doPlaySounds
        if self.doPlaySounds :
#             self.soundLocs = ["/media/ilisescu/Data1/PhD/data/drumming2/meow.wav",   ## snare               https://www.freesound.org/people/steffcaffrey/sounds/262312/
#                               "/media/ilisescu/Data1/PhD/data/drumming2/mario.wav",   ## high tom           https://www.youtube.com/watch?v=cBY_2ABINVg
#                               "/media/ilisescu/Data1/PhD/data/drumming2/laser.wav",   ## high hat           https://www.freesound.org/people/tlwm/sounds/165825/
#                               "/media/ilisescu/Data1/PhD/data/drumming2/mirror.wav",   ## crash             http://soundbible.com/994-Mirror-Shattering.html
#                               "/media/ilisescu/Data1/PhD/data/drumming2/lightsaber.wav",   ## floor tom     http://sweetsoundeffects.com/lightsaber-sounds/
#                               "/media/ilisescu/Data1/PhD/data/drumming2/frog.wav",   ## low tom             https://www.freesound.org/people/daveincamas/sounds/71964/
#                               "/media/ilisescu/Data1/PhD/data/drumming2/splash.wav",   ## splash            https://www.freesound.org/people/InspectorJ/sounds/352098/
#                               "/media/ilisescu/Data1/PhD/data/drumming2/thunder.wav"]   ## bass             drum https://www.freesound.org/people/juskiddink/sounds/101933/
            self.soundLocs = np.sort(glob.glob("/media/ilisescu/Data1/PhD/data/toy/toy[1-8].wav"))
            self.pygletMusic = []
            for soundIdx, soundLoc in enumerate(self.soundLocs) :
                print soundLoc
                self.pygletMusic.append(pyglet.resource.media(soundLoc.split(os.sep)[-1], streaming=False))

            self.playSoundsTimes = {}
        
        
        self.loadSynthesisedSequenceAtLocation(startSequenceLoc)
        
        self.setFocus()
                        
    def playSound(self, idx) :
        self.pygletMusic[idx].play()
       
    #### RENDERING ####

    def renderOneFrame(self) :
        if DICT_SEQUENCE_INSTANCES in self.synthesisedSequence :
            idx = self.frameIdx + 1
            if idx >= 0 and len(self.synthesisedSequence[DICT_SEQUENCE_INSTANCES]) > 0 : #idx < len(self.generatedSequence[0][DICT_SEQUENCE_FRAMES]) :
                self.frameIdxSpinBox.setValue(np.mod(idx, self.frameIdxSpinBox.maximum()+1))

            self.frameInfo.setText("Rendering at " + np.string_(int(1.0/(time.time() - self.lastRenderTime))) + " FPS")
            self.lastRenderTime = time.time()
            
    def showFrame(self, idx) :
        if np.any(self.bgImage != None) and self.overlayImg.size() != QtCore.QSize(self.bgImage.shape[1], self.bgImage.shape[0]) :
            self.overlayImg = self.overlayImg.scaled(QtCore.QSize(self.bgImage.shape[1], self.bgImage.shape[0]))
        if self.frameLabel.qImage != None and self.overlayImg.size() != self.frameLabel.qImage.size() :
            self.overlayImg = self.overlayImg.scaled(self.frameLabel.qImage.size())
            
            
        ## empty image
        self.overlayImg.fill(QtGui.QColor.fromRgb(255, 255, 255, 0))
        if idx >= 0 and len(self.synthesisedSequence[DICT_SEQUENCE_INSTANCES]) > 0 :
            self.frameIdx = idx
            if self.doPlaySounds and idx in self.playSoundsTimes.keys() :
                for i in self.playSoundsTimes[idx].keys() :
                    self.playSound(self.playSoundsTimes[idx][i])
                    print "playing sound", self.playSoundsTimes[idx][i],
                print
            minFrames = len(self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][0][DICT_SEQUENCE_FRAMES])
            for synthSeq in self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][1:] :
                minFrames = np.min([minFrames, len(synthSeq[DICT_SEQUENCE_FRAMES])])
            if self.mergeActionRequests and self.frameIdx == minFrames-2 :
                print "EXTEND", self.frameIdx, minFrames
                self.extendFullSequenceNew(np.array([-1]), verbose=False)
            
            ## go through all the semantic sequence instances
            for s in np.arange(len(self.synthesisedSequence[DICT_SEQUENCE_INSTANCES]))[::-1] :
                if s >= len(self.instancesDoShow) or self.instancesDoShow[s] :
                    ## index in self.semanticSequences of current instance
                    seqIdx = int(self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][s][DICT_SEQUENCE_IDX])
                    ## if there's a frame to show and the requested frameIdx exists for current instance draw, else draw just first frame
                    if self.frameIdx < len(self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][s][DICT_SEQUENCE_FRAMES]) :
                        sequenceFrameIdx = int(self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][s][DICT_SEQUENCE_FRAMES][self.frameIdx])
                        if sequenceFrameIdx >= 0 and sequenceFrameIdx < len(self.semanticSequences[seqIdx][DICT_FRAMES_LOCATIONS].keys()) :
                            frameToShowKey = np.sort(self.semanticSequences[seqIdx][DICT_FRAMES_LOCATIONS].keys())[sequenceFrameIdx]
                        else :
                            frameToShowKey = -1
#                             print "NOT OVERLAYING 1"
                    else :
                        frameToShowKey = -1 #np.sort(self.semanticSequences[seqIdx][DICT_FRAMES_LOCATIONS].keys())[0]
#                         print "NOT OVERLAYING 2"

                    if frameToShowKey >= 0 and seqIdx >= 0 and seqIdx < len(self.semanticSequences) :
                        if np.any(self.overlayImg != None) :
                            if seqIdx in self.preloadedPatches.keys() and frameToShowKey in self.preloadedPatches[seqIdx].keys() :
                                self.overlayImg = self.drawOverlay(self.overlayImg, self.semanticSequences[seqIdx], frameToShowKey,
                                                                   self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][s][DICT_OFFSET],
                                                                   self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][s][DICT_SCALE],
                                                                   s in self.selectedSequenceInstancesIdxes, self.drawSpritesBox.isChecked(),
                                                                   self.drawBBoxBox.isChecked(), self.drawCenterBox.isChecked(),
                                                                   self.preloadedPatches[seqIdx][frameToShowKey])
                            else :
                                self.overlayImg = self.drawOverlay(self.overlayImg, self.semanticSequences[seqIdx], frameToShowKey,
                                                                   self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][s][DICT_OFFSET],
                                                                   self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][s][DICT_SCALE],
                                                                   s in self.selectedSequenceInstancesIdxes, self.drawSpritesBox.isChecked(),
                                                                   self.drawBBoxBox.isChecked(), self.drawCenterBox.isChecked())
                                
            if self.showNumbersSequence and self.showNumbersSequenceBox.isChecked() and self.frameIdx < len(self.numbersSequenceFrames) :
                if os.path.isfile(self.numbersSequenceFrames[self.frameIdx]) :
                    numbersImg = np.ascontiguousarray(Image.open(self.numbersSequenceFrames[self.frameIdx]))
                    self.numbersImage = QtGui.QImage(numbersImg.data, numbersImg.shape[1], numbersImg.shape[0], numbersImg.strides[0], QtGui.QImage.Format_ARGB32);
                    
                    painter = QtGui.QPainter(self.overlayImg)
                    painter.setOpacity(0.4)
                    painter.drawImage(QtCore.QPoint(0, 0), self.numbersImage)
                    painter.end()
        else :
            print "NOT RENDERING", idx

        self.frameLabel.setFixedSize(self.overlayImg.width(), self.overlayImg.height())
        self.frameLabel.setOverlay(self.overlayImg)
#         self.overlayImg.save("/media/ilisescu/Data1/PhD/data/synthesisedSequences/multipleCandles/frame-{0:05}.png".format(idx+1))
        
        
    def drawOverlay(self, overlayImg, sequence, frameKey, offset, scale, doDrawSelected, doDrawSprite,
                    doDrawBBox, doDrawCenter, spritePatch = None, bboxColor = QtGui.QColor.fromRgb(0, 0, 255, 255)) :
#         if np.any(self.overlayImg != None) :
#         overlayImg = QtGui.QImage(overlaySize, QtGui.QImage.Format_ARGB32)
        painter = QtGui.QPainter(overlayImg)

        scaleTransf = np.array([[scale[0], 0.0], [0.0, scale[1]]])
        offsetTransf = np.array([[offset[0]], [offset[1]]])

        ## draw sprite
        if doDrawSprite :
            frameSize = [overlayImg.height(), overlayImg.width()]#np.array(Image.open(sequence[DICT_FRAMES_LOCATIONS][sequence[DICT_FRAMES_LOCATIONS].keys()[0]])).shape[:2]
            if DICT_BBOXES in sequence.keys() and frameKey in sequence[DICT_BBOXES].keys() :
                tl = np.min(sequence[DICT_BBOXES][frameKey], axis=0)
                br = np.max(sequence[DICT_BBOXES][frameKey], axis=0)
            else :
                tl = np.array([0.0, 0.0], float)
                br = np.array([frameSize[1], frameSize[0]], float)
                
            w, h = br-tl
            aabb = np.array([tl, tl + [w, 0], br, tl + [0, h]])

            transformedAABB = (np.dot(scaleTransf, aabb.T) + offsetTransf)
            
            if np.any(spritePatch != None) :
                transformedPatchTopLeftDelta = np.dot(scaleTransf, spritePatch['top_left_pos'][::-1].reshape((2, 1))-tl.reshape((2, 1)))

                image = np.ascontiguousarray(np.zeros((spritePatch['patch_size'][0], spritePatch['patch_size'][1], 4)), dtype=np.uint8)
                image[spritePatch['visible_indices'][:, 0], spritePatch['visible_indices'][:, 1], :] = spritePatch['sprite_colors']

            else :
                transformedPatchTopLeftDelta = np.zeros((2, 1))

                frameName = sequence[DICT_FRAMES_LOCATIONS][frameKey].split(os.sep)[-1]
                if DICT_MASK_LOCATION in sequence.keys() :
                    if os.path.isfile(sequence[DICT_MASK_LOCATION]+frameName) :
                        image = np.array(Image.open(sequence[DICT_MASK_LOCATION]+frameName))[:, :, [2, 1, 0, 3]]
                    else :
                        image = np.zeros([frameSize[0], frameSize[1], 4], np.uint8)
                else :
                    if self.loadedSynthesisedSequence == "/home/ilisescu/PhD/data/synthesisedSequences/lullaby_demo/synthesised_sequence.npy" :
                        image = preloadedImages[frameKey]
                    else :
                        image = np.array(Image.open(sequence[DICT_FRAMES_LOCATIONS][frameKey]))
                image = np.ascontiguousarray(image[aabb[0, 1]:aabb[2, 1], aabb[0, 0]:aabb[2, 0], :])

            
            if image.shape[-1] == 3 :
                img = QtGui.QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QtGui.QImage.Format_RGB888)
            else :
                img = QtGui.QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QtGui.QImage.Format_ARGB32)


            topLeftPos = transformedAABB[:, :1] + transformedPatchTopLeftDelta
            painter.drawImage(QtCore.QPoint(topLeftPos[0], topLeftPos[1]), img.scaled(img.width()*scale[0], img.height()*scale[1]))

            if doDrawSelected :
                painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(255, 0, 0, 255), 1, 
                                          QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))

                for p1, p2 in zip(np.mod(np.arange(4), 4), np.mod(np.arange(1, 5), 4)) :
                    painter.drawLine(QtCore.QPointF(transformedAABB[0, p1], transformedAABB[1, p1]), QtCore.QPointF(transformedAABB[0, p2], transformedAABB[1, p2]))


        painter.setPen(QtGui.QPen(bboxColor, 3, 
                                  QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
        painter.setRenderHints(QtGui.QPainter.Antialiasing)
        ## draw bbox
        if doDrawBBox and DICT_BBOXES in sequence.keys() and frameKey in sequence[DICT_BBOXES].keys() :
#                 if DICT_FOOTPRINTS in sprite.keys() :
#                     bbox = sequence[DICT_FOOTPRINTS][frameKey]
#                 else :
            bbox = sequence[DICT_BBOXES][frameKey]
            transformedBBox = (np.dot(scaleTransf, bbox.T) + offsetTransf)

            for p1, p2 in zip(np.mod(np.arange(4), 4), np.mod(np.arange(1, 5), 4)) :
                painter.drawLine(QtCore.QPointF(transformedBBox[0, p1], transformedBBox[1, p1]), QtCore.QPointF(transformedBBox[0, p2], transformedBBox[1, p2]))

        ## draw bbox center
        if doDrawCenter and frameKey in sequence[DICT_BBOXES].keys() :
            transformedCenter = (np.dot(scaleTransf, sequence[DICT_BBOX_CENTERS][frameKey].reshape((2, 1))) + offsetTransf)
            painter.drawPoint(QtCore.QPointF(transformedCenter[0], transformedCenter[1]))

        painter.end()
        return overlayImg
           
    #### SYNTHESIS ####

    def extendFullSequence(self) :
        minFrames = self.frameIdxSlider.maximum()+1
        for i in xrange(len(self.synthesisedSequence[DICT_SEQUENCE_INSTANCES])) :
            minFrames = np.min((minFrames, len(self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][i][DICT_SEQUENCE_FRAMES])))
        ## got to min frame
        self.frameIdxSpinBox.setValue(minFrames-1)
            
        maxFrames = 0
        for i in xrange(len(self.synthesisedSequence[DICT_SEQUENCE_INSTANCES])) :
            
            availableDesiredSemantics = len(self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][i][DICT_DESIRED_SEMANTICS]) - self.frameIdx
            if availableDesiredSemantics < self.EXTEND_LENGTH :
                ## the required desired semantics by copying the last one
                print "extended desired semantics for", i,
                lastSemantics = self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][i][DICT_DESIRED_SEMANTICS][-1, :]
                self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][i][DICT_DESIRED_SEMANTICS] = np.concatenate((self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][i][DICT_DESIRED_SEMANTICS],
                                                                                                               lastSemantics.reshape((1, len(lastSemantics))).repeat(self.EXTEND_LENGTH-availableDesiredSemantics, axis=0)))
            else :
                print "didn't extend semantics for", i
            desiredSemantics = self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][i][DICT_DESIRED_SEMANTICS][self.frameIdx:self.frameIdx+self.EXTEND_LENGTH, :]
            print "num of desired semantics =", desiredSemantics.shape[0], "(", len(self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][i][DICT_DESIRED_SEMANTICS]), ")",
            
            seqIdx = self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][i][DICT_SEQUENCE_IDX]

            framesToNotUse=np.empty((0, 2))
            for j in xrange(2) :
                if seqIdx in self.preloadedTransitionCosts.keys() :
#                     print "at frame", self.frameIdx, "extending instance", i, "from sequence", seqIdx, "with semantics", desiredSemantics
                    newFrames = self.getNewFramesForSequenceInstanceQuick(i, self.semanticSequences[seqIdx],
                                                                          self.preloadedTransitionCosts[seqIdx]+self.toggleSpeedDeltaSpinBox.value(),
                                                                          desiredSemantics, self.frameIdx, framesToNotUse)
                else :
                    print "ERROR: cannot extend instance", i, "because the semantic sequence", seqIdx, "does not have preloadedTransitionCosts"
                    break
                
                if self.randomizeSequenceBox.isChecked() and np.random.randn(1) > 0 :
                    print "trying again"
                    randIndices = np.random.choice(np.arange(1, len(newFrames)), 50, replace=False)
            #     framesToNotUse = np.concatenate((framesToNotUse, np.array([minCostTraversal[randIndices], randIndices]).T))
                    framesToNotUse = np.array([newFrames[randIndices], randIndices]).T
                else :
                    break
        
            self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][i][DICT_SEQUENCE_FRAMES] = np.hstack((self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][i][DICT_SEQUENCE_FRAMES][:self.frameIdx+1], newFrames[1:]))
            print "extended to", len(self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][i][DICT_SEQUENCE_FRAMES]), ", ", len(self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][i][DICT_DESIRED_SEMANTICS])
            
            maxFrames = np.max((maxFrames, len(self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][i][DICT_SEQUENCE_FRAMES])))
            
                ## update sliders
        self.frameIdxSlider.setMaximum(maxFrames-1)
        self.frameIdxSpinBox.setRange(0, maxFrames-1)

        self.frameInfo.setText("Generated sequence length: " + np.string_(maxFrames))
#         QtCore.QCoreApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)

        self.setSemanticSliderDrawables()

    def extendFullSequenceNew(self, conflictingSequences, instanceSubset=None, verbose=True) :
        startTime = time.time()
        if verbose :
            print "############################ STARTING EXTENDING ###############################"
#         minFrames = self.frameIdxSlider.maximum()+1
#         for i in xrange(len(self.synthesisedSequence[DICT_SEQUENCE_INSTANCES])) :
#             minFrames = np.min((minFrames, len(self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][i][DICT_SEQUENCE_FRAMES])))
#         ## got to min frame
#         self.frameIdxSpinBox.setValue(minFrames-1)
            
        instancesToUse = []
        instancesLengths = []
#         maxFrames = 0
        for i in xrange(len(self.synthesisedSequence[DICT_SEQUENCE_INSTANCES])) :
            if np.any(instanceSubset == None) or i in instanceSubset :
                availableDesiredSemantics = len(self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][i][DICT_DESIRED_SEMANTICS]) - self.frameIdx
                if availableDesiredSemantics < self.EXTEND_LENGTH :
                    ## extend the required desired semantics by copying the last one
                    if verbose :
                        print "extended desired semantics for", i,
                    lastSemantics = self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][i][DICT_DESIRED_SEMANTICS][-1, :]
                    self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][i][DICT_DESIRED_SEMANTICS] = np.concatenate((self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][i][DICT_DESIRED_SEMANTICS],
                                                                                                                   lastSemantics.reshape((1, len(lastSemantics))).repeat(self.EXTEND_LENGTH-availableDesiredSemantics, axis=0)))
                else :
                    if verbose :
                        print "didn't extend semantics for", i
                desiredSemantics = self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][i][DICT_DESIRED_SEMANTICS][self.frameIdx:self.frameIdx+self.EXTEND_LENGTH, :]
                
                if verbose :
                    print "num of desired semantics =", desiredSemantics.shape[0], "(", len(self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][i][DICT_DESIRED_SEMANTICS]), ")"

                seqIdx = self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][i][DICT_SEQUENCE_IDX]

                if seqIdx in self.preloadedTransitionCosts.keys() :
                    instancesToUse.append(i)
                    instancesLengths.append(len(self.semanticSequences[seqIdx][DICT_FRAME_SEMANTICS]))
    #                 newFrames = self.getNewFramesForSequenceInstanceQuick(i, self.semanticSequences[seqIdx],
    #                                                                       self.preloadedTransitionCosts[seqIdx]+self.toggleSpeedDeltaSpinBox.value(),
    #                                                                       desiredSemantics, self.frameIdx, framesToNotUse)
                else :
                    print "ERROR: cannot extend instance", i, "because the semantic sequence", seqIdx, "does not have preloadedTransitionCosts"
                    break
            else :
                if verbose :
                    print "not synthesising instance", i
        
        ######################################
#         newFrames = self.getNewFramesForSequenceFull(self.synthesisedSequence, np.array(instancesToUse), np.array(instancesLengths), self.frameIdx)
        ######################################
    
        instancesToUse = np.array(instancesToUse)
        instancesLengths = np.array(instancesLengths)
#         print instancesToUse, instancesLengths

#         conflictingSequences = np.array([1, 2])
        ## using Peter's idea
        if True :
        #     print conflictingSequences, instancesToUse, np.array([instancesToUse != conflictingSequence for conflictingSequence in conflictingSequences]).all(axis=0)
            notConflicting = np.array([instancesToUse != conflictingSequence for conflictingSequence in conflictingSequences]).all(axis=0)
            notConflictingInstances = instancesToUse[notConflicting]
            conflictingSequences = instancesToUse[np.negative(notConflicting)]
            bestCost = GRAPH_MAX_COST*500
            bestNewFrames = 0
            bestReorderedInstances = 0
            if len(conflictingSequences) > 0 :
                for s in xrange(len(conflictingSequences)) : #permutation in itertools.permutations(conflictingSequences, len(conflictingSequences)) :
                    
                    if verbose :
                        print "resolving conflict", s
            #         print np.concatenate((notConflictingInstances, permutation)), np.concatenate((np.ones(len(notConflictingInstances), bool), np.zeros(len(permutation), bool)))
                    reorderedInstances = np.concatenate((notConflictingInstances, np.roll(conflictingSequences, s)))
                    reorderedLengths = np.concatenate((instancesLengths[notConflicting], np.roll(instancesLengths[np.negative(notConflicting)], s)))
                    lockedInstances = np.concatenate((np.ones(len(instancesToUse)-1, bool), [False]))
#                     print reorderedInstances, reorderedLengths, lockedInstances
#                     print 
                    newFrames, newCost = self.getNewFramesForSequenceIterative(self.synthesisedSequence, reorderedInstances, reorderedLengths, lockedInstances,
                                                                               self.frameIdx, True,#self.resolveCompatibilityBox.isChecked(),
                                                                               self.costsAlphaSpinBox.value(), self.compatibilityAlphaSpinBox.value())
#                     print
                    if bestCost >= newCost :
                        bestNewFrames = newFrames
                        bestCost = newCost
                        bestReorderedInstances = reorderedInstances
            else :
#                 print notConflictingInstances, instancesLengths[notConflicting], np.zeros(len(notConflictingInstances), bool)
#                 bestNewFrames, bestCost = self.getNewFramesForSequenceIterative(self.synthesisedSequence, notConflictingInstances, instancesLengths[notConflicting],
#                                                                                 np.zeros(len(notConflictingInstances), bool), self.frameIdx, self.resolveCompatibilityBox.isChecked(),
#                                                                                 self.costsAlphaSpinBox.value(), self.compatibilityAlphaSpinBox.value())
                downsampleRate = self.optimizationDownsampleRateSpinBox.value()
                bestNewFrames, bestCost, unpackedStartIdx = self.getNewFramesForSequenceIterativeDownsampled(self.synthesisedSequence, notConflictingInstances, instancesLengths[notConflicting],
                                                                                                             np.zeros(len(notConflictingInstances), bool), self.frameIdx, self.resolveCompatibilityBox.isChecked(),
                                                                                                             self.costsAlphaSpinBox.value(), self.compatibilityAlphaSpinBox.value(), downsampleRate)
                print "GENERATED SEQUENCES"
                for newSeqKey in bestNewFrames.keys() :
                    bestNewFrames[newSeqKey] = np.concatenate([np.array([newFrameIdx*downsampleRate]) if (cIdx < len(bestNewFrames[newSeqKey])-1 and
                                                                                                          bestNewFrames[newSeqKey][cIdx+1]-bestNewFrames[newSeqKey][cIdx] != 1)
                                                               else np.arange(newFrameIdx*downsampleRate, newFrameIdx*downsampleRate+downsampleRate)
                                                               for cIdx, newFrameIdx in enumerate(bestNewFrames[newSeqKey])])[unpackedStartIdx[newSeqKey]:]
                    print bestNewFrames[newSeqKey]
                
#                 bob = time.time()
#                 unpackedNewFrames = {}
#                 for newSeqKey in tmpBestNewFrames.keys() :
#                     unpackedNewFrames[newSeqKey] = np.concatenate([np.array([newFrameIdx*downsampleRate]) if (cIdx < len(tmpBestNewFrames[newSeqKey])-1 and
#                                                                    tmpBestNewFrames[newSeqKey][cIdx+1]-tmpBestNewFrames[newSeqKey][cIdx] != 1)
#                                                                    else np.arange(newFrameIdx*downsampleRate, newFrameIdx*downsampleRate+downsampleRate)
#                                                                    for cIdx, newFrameIdx in enumerate(tmpBestNewFrames[newSeqKey])])[unpackedStartIdx[newSeqKey]:]
#                 print "UNPACKING DURATION", time.time()-bob
#                 print "###################################################### README ######################################################"
#                 print bestNewFrames[0], len(bestNewFrames[0])
#                 print
#                 print tmpBestNewFrames[0]*downsampleRate
#                 print
#                 print unpackedNewFrames[0], len(unpackedNewFrames[0])
#                 print "####################################################################################################################"
                bestReorderedInstances = notConflictingInstances
                    
                    
        maxFrames = 0
        for instanceIdx in bestReorderedInstances :
            newFrames = bestNewFrames[instanceIdx]
        
            self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][instanceIdx][DICT_SEQUENCE_FRAMES] = np.hstack((self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][instanceIdx][DICT_SEQUENCE_FRAMES][:self.frameIdx+1],
                                                                                                              newFrames[1:]))
            
            if verbose :
                print "extended to", len(self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][instanceIdx][DICT_SEQUENCE_FRAMES]), ", ", len(self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][instanceIdx][DICT_DESIRED_SEMANTICS])
            
            maxFrames = np.max((maxFrames, len(self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][instanceIdx][DICT_SEQUENCE_FRAMES])))

#             framesToNotUse=np.empty((0, 2))
#             for j in xrange(2) :
#                 if seqIdx in self.preloadedTransitionCosts.keys() :
# #                     print "at frame", self.frameIdx, "extending instance", i, "from sequence", seqIdx, "with semantics", desiredSemantics
#                     newFrames = self.getNewFramesForSequenceInstanceQuick(i, self.semanticSequences[seqIdx],
#                                                                           self.preloadedTransitionCosts[seqIdx]+self.toggleSpeedDeltaSpinBox.value(),
#                                                                           desiredSemantics, self.frameIdx, framesToNotUse)
#                 else :
#                     print "ERROR: cannot extend instance", i, "because the semantic sequence", seqIdx, "does not have preloadedTransitionCosts"
#                     break
                
#                 if self.randomizeSequenceBox.isChecked() and np.random.randn(1) > 0 :
#                     print "trying again"
#                     randIndices = np.random.choice(np.arange(1, len(newFrames)), 50, replace=False)
#             #     framesToNotUse = np.concatenate((framesToNotUse, np.array([minCostTraversal[randIndices], randIndices]).T))
#                     framesToNotUse = np.array([newFrames[randIndices], randIndices]).T
#                 else :
#                     break
        
#             self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][i][DICT_SEQUENCE_FRAMES] = np.hstack((self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][i][DICT_SEQUENCE_FRAMES][:self.frameIdx+1], newFrames[1:]))
#             print "extended to", len(self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][i][DICT_SEQUENCE_FRAMES]), ", ", len(self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][i][DICT_DESIRED_SEMANTICS])
            
#             maxFrames = np.max((maxFrames, len(self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][i][DICT_SEQUENCE_FRAMES])))
            
                ## update sliders
        self.frameIdxSlider.setMaximum(maxFrames-1)
        self.frameIdxSpinBox.setRange(0, maxFrames-1)

        self.frameInfo.setText("Generated sequence length: " + np.string_(maxFrames))
#         QtCore.QCoreApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)

        self.setSemanticSliderDrawables()
    
        print "FINISHED GENERATING SEQUENCE IN", time.time()-startTime
    
    def getNewFramesForSequenceFull(self, synthesisedSequence, instancesToUse, instancesLengths, startingFrame, resolveCompatibility = False) :
        
        gm = opengm.gm(instancesLengths.repeat(self.EXTEND_LENGTH))
        
        self.allUnaries = []
        
        for i, instanceIdx in enumerate(instancesToUse) : # xrange(len(synthesisedSequence[DICT_SEQUENCE_INSTANCES])) :
            seqIdx = synthesisedSequence[DICT_SEQUENCE_INSTANCES][instanceIdx][DICT_SEQUENCE_IDX]
            desiredSemantics = synthesisedSequence[DICT_SEQUENCE_INSTANCES][instanceIdx][DICT_DESIRED_SEMANTICS][startingFrame:startingFrame+self.EXTEND_LENGTH, :]
            
            if len(desiredSemantics) != self.EXTEND_LENGTH :
                raise Exception("desiredSemantics length is not the same as EXTEND_LENGTH")
            
            ################ FIND DESIRED START FRAME ################ 
            if len(synthesisedSequence[DICT_SEQUENCE_INSTANCES][instanceIdx][DICT_SEQUENCE_FRAMES]) == 0 :
                desiredStartFrame = 0
            else :
                desiredStartFrame = synthesisedSequence[DICT_SEQUENCE_INSTANCES][instanceIdx][DICT_SEQUENCE_FRAMES][startingFrame]

            distVariance = 1.0/self.semanticsImportanceSpinBox.value() ##0.0005
            
            ################ GET UNARIES ################
            self.unaries = vectorisedMultiNormalMultipleMeans(self.semanticSequences[seqIdx][DICT_FRAME_SEMANTICS], desiredSemantics, np.eye(desiredSemantics.shape[1])*distVariance, False).T
    
            ## normalizing to turn into probabilities
            self.unaries = self.unaries / np.sum(self.unaries, axis=0).reshape((1, self.unaries.shape[1]))
            impossibleLabels = self.unaries <= 0.0
            ## cost is -log(prob)
            self.unaries[np.negative(impossibleLabels)] = -np.log(self.unaries[np.negative(impossibleLabels)])
            ## if prob == 0.0 then set maxCost
            self.unaries[impossibleLabels] = GRAPH_MAX_COST
            
            
            ## force desiredStartFrame to be the first frame of the new sequence
            self.unaries[:, 0] = GRAPH_MAX_COST
            self.unaries[desiredStartFrame, 0] = 0.0
            
            self.allUnaries.append(np.copy(self.unaries.T))
            
            ## add unaries to the graph
            fids = gm.addFunctions(self.unaries.T)
            # add first order factors
            gm.addFactors(fids, np.arange(self.EXTEND_LENGTH*i, self.EXTEND_LENGTH*i+self.EXTEND_LENGTH))
            
            
            ################ GET PAIRWISE ################
            pairIndices = np.array([np.arange(self.EXTEND_LENGTH-1), np.arange(1, self.EXTEND_LENGTH)]).T + self.EXTEND_LENGTH*i

            ## add function for row-nodes pairwise cost
#             cost = np.copy(sequenceTransitionCost)
            fid = gm.addFunction(self.preloadedTransitionCosts[seqIdx]+self.toggleSpeedDeltaSpinBox.value())
            ## add second order factors
            gm.addFactors(fid, pairIndices)
            
        ################ ADD THE PAIRWISE BETWEEN ROWS ################
        if resolveCompatibility :
            for i, j in np.argwhere(np.triu(np.ones((len(instancesToUse), len(instancesToUse))), 1)) :
                seq1Idx = synthesisedSequence[DICT_SEQUENCE_INSTANCES][i][DICT_SEQUENCE_IDX]
                seq2Idx = synthesisedSequence[DICT_SEQUENCE_INSTANCES][j][DICT_SEQUENCE_IDX]
            #     print i, j
            #     print np.string_(np.min([tracks[i], tracks[j]])) + np.string_(np.max([tracks[i], tracks[j]]))
                pairIndices = np.array([np.arange(self.EXTEND_LENGTH*i, self.EXTEND_LENGTH*i+self.EXTEND_LENGTH), 
                                        np.arange(self.EXTEND_LENGTH*j, self.EXTEND_LENGTH*j+self.EXTEND_LENGTH)]).T
                print pairIndices

                ## add function for column-nodes pairwise cost
                
                fid = gm.addFunction(self.getCompatibilityMat(np.array([seq1Idx, seq2Idx])))
                print "added vertical pairwise between", seq1Idx, "and", seq2Idx
        
                ## add second order factors
                gm.addFactors(fid, pairIndices)
        
        print gm; sys.stdout.flush()
        
        t = time.time()
        inferer = opengm.inference.TrwsExternal(gm=gm)
        inferer.infer()
        print "solved in", time.time() - t

        return np.array(inferer.arg(), dtype=int)#, gm
    
#     newFrames, newCost = self.getNewFramesForSequenceIterative(self.synthesisedSequence, reorderedInstances, reorderedLengths, lockedInstances,
#                                                                                self.frameIdx, True,#self.resolveCompatibilityBox.isChecked(),
#                                                                                self.costsAlphaSpinBox.value(), self.compatibilityAlphaSpinBox.value())

#     bestNewFrames, bestCost = self.getNewFramesForSequenceIterative(self.synthesisedSequence, notConflictingInstances, instancesLengths[notConflicting],
#                                                                     np.zeros(len(notConflictingInstances), bool), self.frameIdx, self.resolveCompatibilityBox.isChecked(),
#                                                                     self.costsAlphaSpinBox.value(), self.compatibilityAlphaSpinBox.value())
    
    def getNewFramesForSequenceIterative(self, synthesisedSequence, instancesToUse, instancesLengths, lockedInstances, startingFrame, resolveCompatibility = False, costsAlpha=0.5, compatibilityAlpha=0.5) :

        self.allUnaries = []

        self.synthesisedFrames = {}
        totalCost = 0.0
        for instanceIdx, instanceLength, lockedInstance in zip(instancesToUse, instancesLengths, lockedInstances) : # xrange(len(synthesisedSequence[DICT_SEQUENCE_INSTANCES])) :

            gm = opengm.gm(np.array([instanceLength]).repeat(self.EXTEND_LENGTH))

            seqIdx = synthesisedSequence[DICT_SEQUENCE_INSTANCES][instanceIdx][DICT_SEQUENCE_IDX]
            desiredSemantics = synthesisedSequence[DICT_SEQUENCE_INSTANCES][instanceIdx][DICT_DESIRED_SEMANTICS][startingFrame:startingFrame+self.EXTEND_LENGTH, :]

            if lockedInstance :
                if len(synthesisedSequence[DICT_SEQUENCE_INSTANCES][instanceIdx][DICT_SEQUENCE_FRAMES][startingFrame:startingFrame+self.EXTEND_LENGTH]) != self.EXTEND_LENGTH :
                    print synthesisedSequence[DICT_SEQUENCE_INSTANCES][instanceIdx][DICT_SEQUENCE_FRAMES][startingFrame:startingFrame+self.EXTEND_LENGTH],
                    print len(synthesisedSequence[DICT_SEQUENCE_INSTANCES][instanceIdx][DICT_SEQUENCE_FRAMES][startingFrame:startingFrame+self.EXTEND_LENGTH])
                    raise Exception("not enough synthesised frames")
                else :
                    self.synthesisedFrames[instanceIdx] = synthesisedSequence[DICT_SEQUENCE_INSTANCES][instanceIdx][DICT_SEQUENCE_FRAMES][startingFrame:startingFrame+self.EXTEND_LENGTH]
                    print "locked instance", instanceIdx
                    print self.synthesisedFrames[instanceIdx]
                    continue

            if len(desiredSemantics) != self.EXTEND_LENGTH :
                raise Exception("desiredSemantics length is not the same as EXTEND_LENGTH")

            ################ FIND DESIRED START FRAME ################ 
            if len(synthesisedSequence[DICT_SEQUENCE_INSTANCES][instanceIdx][DICT_SEQUENCE_FRAMES]) == 0 :
                desiredStartFrame = 0
            else :
                desiredStartFrame = synthesisedSequence[DICT_SEQUENCE_INSTANCES][instanceIdx][DICT_SEQUENCE_FRAMES][startingFrame]

            distVariance = 1.0/self.semanticsImportanceSpinBox.value() ##0.0005

            ################ GET UNARIES ################
            self.unaries = vectorisedMultiNormalMultipleMeans(self.semanticSequences[seqIdx][DICT_FRAME_SEMANTICS], desiredSemantics, np.eye(desiredSemantics.shape[1])*distVariance, False).T

            ## normalizing to turn into probabilities
            self.unaries = self.unaries / np.sum(self.unaries, axis=0).reshape((1, self.unaries.shape[1]))
            impossibleLabels = self.unaries <= 0.0
            ## cost is -log(prob)
            self.unaries[np.negative(impossibleLabels)] = -np.log(self.unaries[np.negative(impossibleLabels)])
            ## if prob == 0.0 then set maxCost
            self.unaries[impossibleLabels] = GRAPH_MAX_COST


            ## force desiredStartFrame to be the first frame of the new sequence
            self.unaries[:, 0] = GRAPH_MAX_COST
            self.unaries[desiredStartFrame, 0] = 0.0
            
            if self.loopSequenceBox.isChecked() :
                print "Looping to", desiredStartFrame
                self.unaries[:, -1] = GRAPH_MAX_COST
                self.unaries[desiredStartFrame, -1] = 0.0
                

            #### minimizing totalCost = a * unary + (1 - a) * (b * vert_link + (1-b)*horiz_link) = a*unary + (1-a)*b*sum(vert_link) + (1-a)*(1-b)*horiz_link
            #### where a = costsAlpha, b = compatibilityAlpha, 

            compatibilityCosts = np.zeros_like(self.unaries)
            if resolveCompatibility :
                seq1Idx = synthesisedSequence[DICT_SEQUENCE_INSTANCES][instanceIdx][DICT_SEQUENCE_IDX]
                for instance2Idx in np.sort(self.synthesisedFrames.keys()) :
                    seq2Idx = synthesisedSequence[DICT_SEQUENCE_INSTANCES][instance2Idx][DICT_SEQUENCE_IDX]
                    print "considering sequences", seq1Idx, seq2Idx, self.synthesisedFrames.keys(), compatibilityCosts.shape

                    compatibilityCosts += (1.0-costsAlpha)*compatibilityAlpha*self.getCompatibilityMat(np.array([seq1Idx, seq2Idx]))[:, self.synthesisedFrames[instance2Idx]]
                    print "added vertical pairwise between", seq1Idx, "and", seq2Idx
                    
                    
    #         ## doing the alpha*unaries + (1-alpha)*pairwise thingy
    #         self.unaries *= costsAlpha
            self.unaries = costsAlpha*self.unaries + compatibilityCosts


            self.allUnaries.append(np.copy(self.unaries.T))


            ## add unaries to the graph
            fids = gm.addFunctions(self.unaries.T)
            # add first order factors
            gm.addFactors(fids, np.arange(self.EXTEND_LENGTH))


            ################ GET PAIRWISE ################
            pairIndices = np.array([np.arange(self.EXTEND_LENGTH-1), np.arange(1, self.EXTEND_LENGTH)]).T

    #         ## add function for row-nodes pairwise cost doing the alpha*unaries + (1-alpha)*pairwise thingy at the same time
    #         fid = gm.addFunction((1.0-costsAlpha)*(self.preloadedTransitionCosts[seqIdx]+0.1))##self.toggleSpeedDeltaSpinBox.value())
            if resolveCompatibility :
                fid = gm.addFunction((1.0-costsAlpha)*(1.0-compatibilityAlpha)*(self.preloadedTransitionCosts[seqIdx]+self.toggleSpeedDeltaSpinBox.value()))
            else :
                fid = gm.addFunction((1.0-costsAlpha)*(self.preloadedTransitionCosts[seqIdx]+self.toggleSpeedDeltaSpinBox.value()))
            ## add second order factors
            gm.addFactors(fid, pairIndices)

            print gm; sys.stdout.flush()

            t = time.time()
            inferer = opengm.inference.DynamicProgramming(gm=gm)
            inferer.infer()
            print "solved in", time.time() - t, "cost", gm.evaluate(inferer.arg())
            print np.array(inferer.arg(), dtype=int)
            totalCost += gm.evaluate(inferer.arg())
            self.synthesisedFrames[instanceIdx] = np.array(inferer.arg(), dtype=int)

        return self.synthesisedFrames, totalCost
    
    def getNewFramesForSequenceIterativeDownsampled(self, synthesisedSequence, instancesToUse, instancesLengths, lockedInstances, startingFrame,
                                                    resolveCompatibility = False, costsAlpha=0.5, compatibilityAlpha=0.5, downsampleRate=1) :
        downsampledInstanceLenghts = np.zeros_like(instancesLengths)
        for idx, instanceIdx in enumerate(instancesToUse) :
            downsampledInstanceLenghts[idx] = len(self.preloadedDistanceMatrices[synthesisedSequence[DICT_SEQUENCE_INSTANCES][instanceIdx][DICT_SEQUENCE_IDX]][::downsampleRate, ::downsampleRate])
            
        extendLength = int(np.floor(self.EXTEND_LENGTH/float(downsampleRate)))
        self.allUnaries = []

        self.synthesisedFrames = {}
        unpackedStartIdx = {}
        totalCost = 0.0
        for instanceIdx, instanceLength, lockedInstance in zip(instancesToUse, downsampledInstanceLenghts, lockedInstances) : # xrange(len(synthesisedSequence[DICT_SEQUENCE_INSTANCES])) :

            gm = opengm.gm(np.array([instanceLength]).repeat(extendLength))

            seqIdx = synthesisedSequence[DICT_SEQUENCE_INSTANCES][instanceIdx][DICT_SEQUENCE_IDX]
            desiredSemantics = synthesisedSequence[DICT_SEQUENCE_INSTANCES][instanceIdx][DICT_DESIRED_SEMANTICS][startingFrame:startingFrame+self.EXTEND_LENGTH, :]
            desiredSemantics = desiredSemantics[::downsampleRate, :]
            desiredSemantics = desiredSemantics[:extendLength, :]

            if lockedInstance :
                if len(synthesisedSequence[DICT_SEQUENCE_INSTANCES][instanceIdx][DICT_SEQUENCE_FRAMES][startingFrame:startingFrame+self.EXTEND_LENGTH]) != self.EXTEND_LENGTH :
                    print synthesisedSequence[DICT_SEQUENCE_INSTANCES][instanceIdx][DICT_SEQUENCE_FRAMES][startingFrame:startingFrame+self.EXTEND_LENGTH],
                    print len(synthesisedSequence[DICT_SEQUENCE_INSTANCES][instanceIdx][DICT_SEQUENCE_FRAMES][startingFrame:startingFrame+self.EXTEND_LENGTH])
                    raise Exception("not enough synthesised frames")
                else :
                    self.synthesisedFrames[instanceIdx] = synthesisedSequence[DICT_SEQUENCE_INSTANCES][instanceIdx][DICT_SEQUENCE_FRAMES][startingFrame:startingFrame+self.EXTEND_LENGTH]
                    print "locked instance", instanceIdx
                    print self.synthesisedFrames[instanceIdx]
                    continue

            if len(desiredSemantics) != extendLength :
                raise Exception("desiredSemantics length is not the same as EXTEND_LENGTH")

            ################ FIND DESIRED START FRAME ################ 
            if len(synthesisedSequence[DICT_SEQUENCE_INSTANCES][instanceIdx][DICT_SEQUENCE_FRAMES]) == 0 :
                desiredStartFrame = 0
            else :
                desiredStartFrame = int(np.floor(synthesisedSequence[DICT_SEQUENCE_INSTANCES][instanceIdx][DICT_SEQUENCE_FRAMES][startingFrame]/float(downsampleRate)))
                ## here I need the actual desired start frame as I'll have to start from it when I unpack the downsampled labels
                unpackedStartIdx[instanceIdx] = int(np.argwhere(np.arange(desiredStartFrame*downsampleRate, desiredStartFrame*downsampleRate+downsampleRate) ==
                                                                synthesisedSequence[DICT_SEQUENCE_INSTANCES][instanceIdx][DICT_SEQUENCE_FRAMES][startingFrame]).flatten())
#                 print "START FRAME", desiredStartFrame, synthesisedSequence[DICT_SEQUENCE_INSTANCES][instanceIdx][DICT_SEQUENCE_FRAMES][startingFrame], unpackedStartIdx[instanceIdx]

            distVariance = 1.0/self.semanticsImportanceSpinBox.value() ##0.0005

            ################ GET UNARIES ################
            self.unaries = vectorisedMultiNormalMultipleMeans(self.semanticSequences[seqIdx][DICT_FRAME_SEMANTICS][::downsampleRate, :], desiredSemantics, np.eye(desiredSemantics.shape[1])*distVariance, False).T

            ## normalizing to turn into probabilities
            self.unaries = self.unaries / np.sum(self.unaries, axis=0).reshape((1, self.unaries.shape[1]))
            impossibleLabels = self.unaries <= 0.0
            ## cost is -log(prob)
            self.unaries[np.negative(impossibleLabels)] = -np.log(self.unaries[np.negative(impossibleLabels)])
            ## if prob == 0.0 then set maxCost
            self.unaries[impossibleLabels] = GRAPH_MAX_COST


            ## force desiredStartFrame to be the first frame of the new sequence
            self.unaries[:, 0] = GRAPH_MAX_COST
            self.unaries[desiredStartFrame, 0] = 0.0
            ## also force second frame to be the subsequent frame so that we don't make a jump after the first frame
            ## this is useful because when I use downsampling, I can start from the desiredStartFrame before dividing by the downsample rate and which is not possible if I perform a jump after the first frame and
            ## desiredStartFrame is not divisible by downsampleRate
            if downsampleRate > 1 and desiredStartFrame < instanceLength-1 :
                self.unaries[:, 1] = GRAPH_MAX_COST
                self.unaries[desiredStartFrame+1, 1] = 0.0
                
            
            if self.loopSequenceBox.isChecked() :
                print "Looping to", desiredStartFrame
                self.unaries[:, -1] = GRAPH_MAX_COST
                self.unaries[desiredStartFrame, -1] = 0.0
                

            #### minimizing totalCost = a * unary + (1 - a) * (b * vert_link + (1-b)*horiz_link) = a*unary + (1-a)*b*sum(vert_link) + (1-a)*(1-b)*horiz_link
            #### where a = costsAlpha, b = compatibilityAlpha, 

            compatibilityCosts = np.zeros_like(self.unaries)
            if resolveCompatibility :
                seq1Idx = synthesisedSequence[DICT_SEQUENCE_INSTANCES][instanceIdx][DICT_SEQUENCE_IDX]
                for instance2Idx in np.sort(self.synthesisedFrames.keys()) :
                    seq2Idx = synthesisedSequence[DICT_SEQUENCE_INSTANCES][instance2Idx][DICT_SEQUENCE_IDX]
                    print "considering sequences", seq1Idx, seq2Idx, self.synthesisedFrames.keys(), compatibilityCosts.shape

                    compatibilityCosts += (1.0-costsAlpha)*compatibilityAlpha*(self.getCompatibilityMat(np.array([seq1Idx, seq2Idx]))[::downsampleRate, ::downsampleRate])[:, self.synthesisedFrames[instance2Idx]]
                    print "added vertical pairwise between", seq1Idx, "and", seq2Idx
                    
                    
    #         ## doing the alpha*unaries + (1-alpha)*pairwise thingy
    #         self.unaries *= costsAlpha
            self.unaries = costsAlpha*self.unaries + compatibilityCosts


            self.allUnaries.append(np.copy(self.unaries.T))


            ## add unaries to the graph
            fids = gm.addFunctions(self.unaries.T)
            # add first order factors
            gm.addFactors(fids, np.arange(extendLength))


            ################ GET PAIRWISE ################
            pairIndices = np.array([np.arange(extendLength-1), np.arange(1, extendLength)]).T

    #         ## add function for row-nodes pairwise cost doing the alpha*unaries + (1-alpha)*pairwise thingy at the same time
    #         fid = gm.addFunction((1.0-costsAlpha)*(self.preloadedTransitionCosts[seqIdx]+0.1))##self.toggleSpeedDeltaSpinBox.value())
            if resolveCompatibility :
                fid = gm.addFunction((1.0-costsAlpha)*(1.0-compatibilityAlpha)*(self.preloadedTransitionCosts[seqIdx]+self.toggleSpeedDeltaSpinBox.value()))
            else :
                fid = gm.addFunction((1.0-costsAlpha)*(self.preloadedTransitionCosts[seqIdx]+self.toggleSpeedDeltaSpinBox.value()))
            ## add second order factors
            gm.addFactors(fid, pairIndices)

            print gm; sys.stdout.flush()

            t = time.time()
            inferer = opengm.inference.DynamicProgramming(gm=gm)
            inferer.infer()
            print "solved in", time.time() - t, "cost", gm.evaluate(inferer.arg())
            print np.array(inferer.arg(), dtype=int)
            totalCost += gm.evaluate(inferer.arg())
            self.synthesisedFrames[instanceIdx] = np.array(inferer.arg(), dtype=int)

        return self.synthesisedFrames, totalCost, unpackedStartIdx
        
    
    def getNewFramesForSequenceInstanceQuick(self, instanceIdx, semanticSequence, sequenceTransitionCost, desiredSemantics, startingFrame, framesToNotUse=np.empty((0, 2)), resolveCompatibility = False) :
        print "min cost", np.min(sequenceTransitionCost); sys.stdout.flush()
        ## set starting frame
        if len(self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][instanceIdx][DICT_SEQUENCE_FRAMES]) == 0 :
            desiredStartFrame = 0
        else :
            desiredStartFrame = self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][instanceIdx][DICT_SEQUENCE_FRAMES][startingFrame]
            
        distVariance = 1.0/self.semanticsImportanceSpinBox.value() ##0.0005
        
        ###########################################
        
        
#         self.unaries = vectorisedMinusLogMultiNormalMultipleMeans(semanticSequence[DICT_FRAME_SEMANTICS], desiredSemantics, np.eye(desiredSemantics.shape[1])*distVariance, True).T
        self.unaries = vectorisedMultiNormalMultipleMeans(semanticSequence[DICT_FRAME_SEMANTICS], desiredSemantics, np.eye(desiredSemantics.shape[1])*distVariance, False).T
    
        ## normalizing to turn into probabilities
        self.unaries = self.unaries / np.sum(self.unaries, axis=0).reshape((1, self.unaries.shape[1]))
        impossibleLabels = self.unaries <= 0.0
        ## cost is -log(prob)
        self.unaries[np.negative(impossibleLabels)] = -np.log(self.unaries[np.negative(impossibleLabels)])
        ## if prob == 0.0 then set maxCost
        self.unaries[impossibleLabels] = GRAPH_MAX_COST
        
        
        #####################################
        
# #         synthSeq = np.load("/home/ilisescu/PhD/data/synthesisedSequences/wave-tagging_bad_jumps/synthesised_sequence.npy").item()
# #         # print synthSeq[DICT_SEQUENCE_INSTANCES][1][DICT_DESIRED_SEMANTICS][68:169]
# #         usedSeq = np.load(synthSeq[DICT_USED_SEQUENCES][1]).item()
#         self.unaries = np.zeros((semanticSequence[DICT_FRAME_SEMANTICS].shape[0], len(desiredSemantics)))
#         desiredSems = np.argmax(desiredSemantics, axis=1) #np.argmax(synthSeq[DICT_SEQUENCE_INSTANCES][1][DICT_DESIRED_SEMANTICS][68:169, :], axis=1)
#         thresholdedSems = desiredSemantics[np.arange(len(desiredSemantics)), desiredSems] > 0.75
#         print thresholdedSems.shape, desiredSems.shape, self.unaries.shape
# #         (101, 101) (101,) (720, 101)
# #         print desiredSems * thresholdedSems
# #         print (1-desiredSems) * thresholdedSems

#         for i, des in enumerate(desiredSems * thresholdedSems) :
#             if des == 1 :
#                 self.unaries[:, i] = (semanticSequence[DICT_FRAME_SEMANTICS][:, 1] <= 0.75).astype(float)*GRAPH_MAX_COST

#         for i, des in enumerate((1-desiredSems) * thresholdedSems) :
#             if des == 1 :
#                 self.unaries[:, i] = (semanticSequence[DICT_FRAME_SEMANTICS][:, 0] <= 0.75).astype(float)*GRAPH_MAX_COST
        
        
        
        #######################################

        

        ## force desiredStartFrame to be the first frame of the new sequence
        self.unaries[:, 0] = GRAPH_MAX_COST
        self.unaries[desiredStartFrame, 0] = 0.0
        
        for j in xrange(len(framesToNotUse)) :
            self.unaries[framesToNotUse[j, 0], framesToNotUse[j, 1]] = GRAPH_MAX_COST
        
        numNodes = len(desiredSemantics)
        numLabels = sequenceTransitionCost.shape[0]
        gm = opengm.gm(np.ones(numNodes,dtype=opengm.label_type)*numLabels, operator='adder')


        fids = gm.addFunctions(self.unaries.T)
        # add first order factors
        gm.addFactors(fids, np.arange(numNodes))

        pairIndices = np.array([np.arange(numNodes-1), np.arange(1, numNodes)]).T

        ## add function for row-nodes pairwise cost
#         fid = gm.addFunction(sequenceTransitionCost+np.random.rand(sequenceTransitionCost.shape[0], sequenceTransitionCost.shape[1])*0.01-0.005)
#         bestTransitions = np.argsort(sequenceTransitionCost, axis=-1)
#         jumpLength = np.abs(bestTransitions-np.arange(sequenceTransitionCost.shape[0]).reshape((sequenceTransitionCost.shape[0], 1)))
#         minJumpLength = 10
#         numTop = 15
#         topBest = np.array([bestTransitions[i, jumpLength[i, :] >= minJumpLength][:numTop] for i in xrange(sequenceTransitionCost.shape[0])])
        cost = np.copy(sequenceTransitionCost)
#         for i in xrange(sequenceTransitionCost.shape[0]) :
#             cost[i, topBest[i, :]] += (np.random.rand(numTop)*0.02-0.01)
        fid = gm.addFunction(cost)
        ## add second order factors
        gm.addFactors(fid, pairIndices)
        inferer = opengm.inference.DynamicProgramming(gm=gm)
        inferer.infer()
        print gm
        
        print "computed path cost:", gm.evaluate(inferer.arg())

        return np.array(inferer.arg(), dtype=int)
    
    def getCompatibilityMat(self, seqIdxs, doOverride=False, verbose=True) :
        needsTransposed = seqIdxs[0] > seqIdxs[1]
        matKey = np.string_(np.min(seqIdxs))+"-"+np.string_(np.max(seqIdxs))
        
        DEF_USE_PROBABILITIES = False
        
        ## get precomputed one if existing
        if not doOverride and matKey in self.compatibilityMats.keys() :
            if verbose :
                print "USING PRECOMPUTED COMPATIBILITY", matKey,
        else :
            startingTime = time.time()
            if DEF_USE_PROBABILITIES :
                compatibilityMat = np.ones((self.semanticSequences[seqIdxs[0]][DICT_FRAME_SEMANTICS].shape[0],
                                            self.semanticSequences[seqIdxs[1]][DICT_FRAME_SEMANTICS].shape[0]))
                compatibilityMat /= np.sum(compatibilityMat)
            else :
                compatibilityMat = np.zeros((self.semanticSequences[seqIdxs[0]][DICT_FRAME_SEMANTICS].shape[0],
                                             self.semanticSequences[seqIdxs[1]][DICT_FRAME_SEMANTICS].shape[0]))
            ## compute
            if (DICT_LABELLED_FRAMES in self.semanticSequences[seqIdxs[0]].keys() and
                DICT_LABELLED_FRAMES in self.semanticSequences[seqIdxs[1]].keys()) :
                
                ## get compatibility labels for the 2 sequences
#                 seqCompatibilityLabels = [self.getFrameCompatibilityLabels(seqIdxs[0], self.semanticSequences[seqIdxs[1]][DICT_SEQUENCE_LOCATION]),
#                                           self.getFrameCompatibilityLabels(seqIdxs[1], self.semanticSequences[seqIdxs[0]][DICT_SEQUENCE_LOCATION])]
                print "IN GET COMPATIBILITY MAT",
                sequencePairCompatibilityLabels = self.getSequencePairCompatibilityLabels(seqIdxs)
                
                ## duplicated code when rendering info image
#                 seq1NumLabels = self.semanticSequences[seqIdxs[0]][DICT_FRAME_COMPATIBILITY_LABELS].shape[1]
#                 seq2NumLabels = self.semanticSequences[seqIdxs[1]][DICT_FRAME_COMPATIBILITY_LABELS].shape[1]
                seq1NumLabels = sequencePairCompatibilityLabels[0][0].shape[1]
                seq2NumLabels = sequencePairCompatibilityLabels[1][0].shape[1]
                labelCompatibility = np.ones((seq1NumLabels, seq2NumLabels))
                
                ## setting labelCompatibility where there are conflicts if any
                seq2Location = self.semanticSequences[seqIdxs[1]][DICT_SEQUENCE_LOCATION]
                
                ## duplicated in info image
                if (DICT_CONFLICTING_SEQUENCES in self.semanticSequences[seqIdxs[0]].keys() and
                    seq2Location in self.semanticSequences[seqIdxs[0]][DICT_CONFLICTING_SEQUENCES].keys()) :

                    for combination in self.semanticSequences[seqIdxs[0]][DICT_CONFLICTING_SEQUENCES][seq2Location] :
#                         labelCompatibility[combination[0], combination[1]] = 0
                        labelCompatibility[int(np.argwhere(np.array(sequencePairCompatibilityLabels[0][1]) == combination[0])),
                                           int(np.argwhere(np.array(sequencePairCompatibilityLabels[1][1]) == combination[1]))] = 0
    
                if DEF_USE_PROBABILITIES :
                    labelCompatibility /= np.sum(labelCompatibility)

                for incompatibleCombination in np.argwhere(labelCompatibility == 0) :
                    ## augmented semantics based incompatibility
                    if self.compatibilityTypeComboBox.currentIndex() == 0 :
                        #     print incompatibleCombination
#                         seq1Distances = getDistToSemantics(self.semanticSequences[seqIdxs[0]][DICT_FRAME_COMPATIBILITY_LABELS], incompatibleCombination[0],
#                                                            self.compatibilitySigmaSpinBox.value()/self.compatibilitySigmaDividerSpinBox.value())
#                         seq2Distances = getDistToSemantics(self.semanticSequences[seqIdxs[1]][DICT_FRAME_COMPATIBILITY_LABELS], incompatibleCombination[1],
#                                                            self.compatibilitySigmaSpinBox.value()/self.compatibilitySigmaDividerSpinBox.value())

#                         seq1Distances = getDistToSemantics(sequencePairCompatibilityLabels[0][0], incompatibleCombination[0],
#                                                            self.compatibilitySigmaSpinBox.value()/self.compatibilitySigmaDividerSpinBox.value())
#                         seq2Distances = getDistToSemantics(sequencePairCompatibilityLabels[1][0], incompatibleCombination[1],
#                                                            self.compatibilitySigmaSpinBox.value()/self.compatibilitySigmaDividerSpinBox.value())
        
                        if DEF_USE_PROBABILITIES :
                            seq1Probs = np.copy(sequencePairCompatibilityLabels[0][0][:, incompatibleCombination[0]].reshape((len(sequencePairCompatibilityLabels[0][0]), 1)))
                            seq1Probs /= np.sum(seq1Probs)
                            seq2Probs = np.copy(sequencePairCompatibilityLabels[1][0][:, incompatibleCombination[1]].reshape((len(sequencePairCompatibilityLabels[1][0]), 1)))
                            seq2Probs /= np.sum(seq2Probs)
                            
                            compatibilityMat += (seq2Probs.T * seq1Probs) * labelCompatibility[incompatibleCombination[0], incompatibleCombination[1]]
                            compatibilityMat /= np.sum(compatibilityMat)
                            
                            print "SA:DKJS", np.max(compatibilityMat), np.min(compatibilityMat), np.max(seq1Probs), np.max(seq2Probs), seq1Probs.shape, 
                            print seq2Probs.shape, incompatibleCombination[0], incompatibleCombination[1], labelCompatibility[incompatibleCombination[0], incompatibleCombination[1]],  np.sum(compatibilityMat)
                        else :
                            seq1ClassProbs = np.copy(sequencePairCompatibilityLabels[0][0][:, incompatibleCombination[0]].reshape((len(sequencePairCompatibilityLabels[0][0]), 1)))
                            seq2ClassProbs = np.copy(sequencePairCompatibilityLabels[1][0][:, incompatibleCombination[1]].reshape((len(sequencePairCompatibilityLabels[1][0]), 1)))
                            compatibilityMat += seq2ClassProbs.T * seq1ClassProbs*100

# #                         if labelCompatibility[incompatibleCombination[0], incompatibleCombination[1]] == 0 :
#                         compatibilityMat += (seq2Probs.T * seq1Probs) * labelCompatibility[incompatibleCombination[0], incompatibleCombination[1]]
                        
#                         if labelCompatibility[incompatibleCombination[0], incompatibleCombination[1]] == 0 :
#                             compatibilityMat += seq2Probs.T * seq1Probs
#                             compatibilityMat /= np.sum(compatibilityMat)
#                         else :
#                             compatibilityMat -= seq2Distances.T * seq1Distances
                    ##  L2 distance based incompatiblity
                    elif self.compatibilityTypeComboBox.currentIndex() == 1 :
                        
                        for frame1, frame2 in zip(self.semanticSequences[seqIdxs[0]][DICT_LABELLED_FRAMES][incompatibleCombination[0]],
                                                  self.semanticSequences[seqIdxs[1]][DICT_LABELLED_FRAMES][incompatibleCombination[1]]) :
#                         frame1 = self.semanticSequences[seqIdxs[0]][DICT_LABELLED_FRAMES][incompatibleCombination[0]][0]
#                         frame2 = self.semanticSequences[seqIdxs[1]][DICT_LABELLED_FRAMES][incompatibleCombination[1]][0]
                            sigma1 = np.median(self.preloadedDistanceMatrices[seqIdxs[0]])*self.compatibilitySigmaSpinBox.value()/self.compatibilitySigmaDividerSpinBox.value()
                            sigma2 = np.median(self.preloadedDistanceMatrices[seqIdxs[1]])*self.compatibilitySigmaSpinBox.value()/self.compatibilitySigmaDividerSpinBox.value()
                            print "sigma1", sigma1, "sigma2", sigma2, "frames:", frame1, frame2
                            seq1Distances = np.exp(-self.preloadedDistanceMatrices[seqIdxs[0]][frame1, :]/(2*sigma1**2))
                            seq2Distances = np.exp(-self.preloadedDistanceMatrices[seqIdxs[1]][frame2, :]/(2*sigma2**2))

                            if labelCompatibility[incompatibleCombination[0], incompatibleCombination[1]] == 0 :
                                compatibilityMat += seq2Distances.reshape((len(seq2Distances), 1)).T * seq1Distances.reshape((len(seq1Distances), 1))
#                             else :
#                                 compatibilityMat -= seq2Distances.reshape((len(seq2Distances), 1)).T * seq1Distances.reshape((len(seq1Distances), 1))
                    ## shortest path based incompatibility
                    elif self.compatibilityTypeComboBox.currentIndex() == 2 :
                        minJumpLength = self.pathCompatibilityMinJumpLengthSpinBox.value()
#                         ## first sequence
#                         gr = graph()
#                         for i in xrange(len(self.preloadedDistanceMatrices[seqIdxs[0]])) :
#                             gr.add_node(i)
#                         for i in xrange(len(self.preloadedDistanceMatrices[seqIdxs[0]])) :
#                             for j in xrange(i+minJumpLength, len(self.preloadedDistanceMatrices[seqIdxs[0]])) :
#                                 gr.add_edge((i, j), wt=self.preloadedDistanceMatrices[seqIdxs[0]][i, j])
#                         frame1 = self.semanticSequences[seqIdxs[0]][DICT_LABELLED_FRAMES][incompatibleCombination[0]][0]
#                         paths = shortest_path(gr, frame1)
#                         costs = np.array([paths[1][key] for key in np.sort(paths[1].keys())])
#                         sigma1 = np.median(self.preloadedDistanceMatrices[seqIdxs[0]])*self.compatibilitySigmaSpinBox.value()/self.compatibilitySigmaDividerSpinBox.value()
#                         seq1Distances = np.exp(-costs/(2*sigma1**2))
#                         seq1Distances /= np.max(seq1Distances)
                        
#                         ## second sequence
#                         gr = graph()
#                         for i in xrange(len(self.preloadedDistanceMatrices[seqIdxs[1]])) :
#                             gr.add_node(i)
#                         for i in xrange(len(self.preloadedDistanceMatrices[seqIdxs[1]])) :
#                             for j in xrange(i, len(self.preloadedDistanceMatrices[seqIdxs[1]])) :
#                                 gr.add_edge((i, j), wt=self.preloadedDistanceMatrices[seqIdxs[1]][i, j])
#                         frame2 = self.semanticSequences[seqIdxs[1]][DICT_LABELLED_FRAMES][incompatibleCombination[1]][0]
#                         paths = shortest_path(gr, frame2)
#                         costs = np.array([paths[1][key] for key in np.sort(paths[1].keys())])
#                         sigma2 = np.median(self.preloadedDistanceMatrices[seqIdxs[1]])*self.compatibilitySigmaSpinBox.value()/self.compatibilitySigmaDividerSpinBox.value()
#                         seq2Distances = np.exp(-costs/(2*sigma2**2))
#                         seq2Distances /= np.max(seq2Distances)
                        
                        for frame1, frame2 in zip(self.semanticSequences[seqIdxs[0]][DICT_LABELLED_FRAMES][incompatibleCombination[0]],
                                                  self.semanticSequences[seqIdxs[1]][DICT_LABELLED_FRAMES][incompatibleCombination[1]]) :

                            ## make graph for first sequence
                            G = networkx.complete_graph(len(self.preloadedDistanceMatrices[seqIdxs[0]]))
                            if minJumpLength > 1 :
                                G.remove_edges_from(np.argwhere(np.triu(np.ones(self.preloadedDistanceMatrices[seqIdxs[0]].shape), 1)-
                                                                np.triu(np.ones(self.preloadedDistanceMatrices[seqIdxs[0]].shape), minJumpLength)))
                            for i, j in G.edges_iter() :
                                G.edge[i][j]['weight'] = self.preloadedDistanceMatrices[seqIdxs[0]][i, j]

                            ## find shortest paths from frame1
#                         frame1 = self.semanticSequences[seqIdxs[0]][DICT_LABELLED_FRAMES][incompatibleCombination[0]][0]
                            pathCosts = networkx.single_source_dijkstra_path_length(G, frame1)
                            costs = np.array([pathCosts[key] for key in np.sort(pathCosts.keys())])

                            ## get compatiblity cost
                            sigma1 = np.median(self.preloadedDistanceMatrices[seqIdxs[0]])*self.compatibilitySigmaSpinBox.value()/self.compatibilitySigmaDividerSpinBox.value()
                            seq1Distances = np.exp(-costs/(2*sigma1**2))
                            seq1Distances /= np.max(seq1Distances)



                            ## make graph for second sequence
                            G = networkx.complete_graph(len(self.preloadedDistanceMatrices[seqIdxs[1]]))
                            if minJumpLength > 1 :
                                G.remove_edges_from(np.argwhere(np.triu(np.ones(self.preloadedDistanceMatrices[seqIdxs[1]].shape), 1)-
                                                                np.triu(np.ones(self.preloadedDistanceMatrices[seqIdxs[1]].shape), minJumpLength)))
                            for i, j in G.edges_iter() :
                                G.edge[i][j]['weight'] = self.preloadedDistanceMatrices[seqIdxs[1]][i, j]

                            ## find shortest paths from frame2
#                         frame2 = self.semanticSequences[seqIdxs[1]][DICT_LABELLED_FRAMES][incompatibleCombination[1]][0]
                            pathCosts = networkx.single_source_dijkstra_path_length(G, frame2)
                            costs = np.array([pathCosts[key] for key in np.sort(pathCosts.keys())])

                            ## get compatiblity cost
                            sigma2 = np.median(self.preloadedDistanceMatrices[seqIdxs[1]])*self.compatibilitySigmaSpinBox.value()/self.compatibilitySigmaDividerSpinBox.value()
                            seq2Distances = np.exp(-costs/(2*sigma2**2))
                            seq2Distances /= np.max(seq2Distances)


                            compatibilityMat += seq2Distances.reshape((len(seq2Distances), 1)).T * seq1Distances.reshape((len(seq1Distances), 1))
                ## turn probs to cost
#                 compatibilityMat = -np.log(1.0-compatibilityMat)*10000.0
#                 compatibilityMat = -np.log(compatibilityMat)
                ## norm to range [0, 1]
#                 if np.min(compatibilityMat) != np.max(compatibilityMat) :
#                     compatibilityMat = (compatibilityMat-np.min(compatibilityMat))/(np.max(compatibilityMat)-np.min(compatibilityMat))
#                 compatibilityMat *= 100
                    
            if needsTransposed :
                self.compatibilityMats[matKey] = compatibilityMat.T
            else :
                self.compatibilityMats[matKey] = compatibilityMat
                
            if verbose :
                print "COMPUTED COMPATIBILITY", matKey, "in", time.time()-startingTime, "seconds"
        
        if needsTransposed :
            if verbose :
                print "compatibilityMat:", self.compatibilityMats[matKey].shape, "returned transposed"; sys.stdout.flush()
            return self.compatibilityMats[matKey].T
        else :
            if verbose :
                print "compatibilityMat:", self.compatibilityMats[matKey].shape; sys.stdout.flush()
            return self.compatibilityMats[matKey]
        
    def updateCompatibilityMatrices(self) :
        if DICT_USED_SEQUENCES in self.synthesisedSequence :
            for pair in np.argwhere(np.triu(np.ones((len(self.synthesisedSequence[DICT_USED_SEQUENCES]), len(self.synthesisedSequence[DICT_USED_SEQUENCES]))))) :
                matKey = np.string_(pair[0])+"-"+np.string_(pair[1])
                if matKey in self.compatibilityMats.keys() :
                    self.getCompatibilityMat(pair, True)

            if len(self.selectedSequenceInstancesIdxes) == 2 :
                self.makeInfoImage([self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][self.selectedSequenceInstancesIdxes[0]][DICT_SEQUENCE_IDX],
                                    self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][self.selectedSequenceInstancesIdxes[1]][DICT_SEQUENCE_IDX]])
                
    def updateCompatibilityLabelsAndMatrices(self) :
            if len(self.selectedSequenceInstancesIdxes) == 2 :
                self.getSequencePairCompatibilityLabels([self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][self.selectedSequenceInstancesIdxes[0]][DICT_SEQUENCE_IDX],
                                                         self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][self.selectedSequenceInstancesIdxes[1]][DICT_SEQUENCE_IDX]], True)
                
                self.updateCompatibilityMatrices()
    
    def updatePreloadedTransitionMatrices(self) :## downsample the transition matrix from the distance matrix
        bob = time.time()
        for seqIdx in self.preloadedDistanceMatrices.keys() :
            ## downsample the distance matrix using opencv maybe later but for now just skip stuff
            if self.semanticSequences[seqIdx][DICT_SEQUENCE_NAME] == "left_hand1" :
                print "USING CUSTOM PARAMS 1 FOR", self.semanticSequences[seqIdx][DICT_SEQUENCE_NAME]
                filterSize = 4; minJumpSize = 12; sigmaMultiplier = 0.006; onlyBackwards = False; threshPercentile = 0.1
            elif self.semanticSequences[seqIdx][DICT_SEQUENCE_NAME] == "right_hand1" :
                print "USING CUSTOM PARAMS 2 FOR", self.semanticSequences[seqIdx][DICT_SEQUENCE_NAME]
                filterSize = 4; minJumpSize = 12; sigmaMultiplier = 0.01; onlyBackwards = True; threshPercentile = 0.1
            elif self.semanticSequences[seqIdx][DICT_SEQUENCE_NAME] == "foot1" :
                print "USING CUSTOM PARAMS 3 FOR", self.semanticSequences[seqIdx][DICT_SEQUENCE_NAME]
                filterSize = 4; minJumpSize = 12; sigmaMultiplier = 0.2; onlyBackwards = True; threshPercentile = 0.2
            else :
                print "USING DEFAULT PARAMS", self.semanticSequences[seqIdx][DICT_SEQUENCE_NAME]
                filterSize = 4; minJumpSize = 20; sigmaMultiplier = 0.002; onlyBackwards = False; threshPercentile = 0.1
                
                
            if self.semanticSequences[seqIdx][DICT_SEQUENCE_NAME] == "toy1" or self.semanticSequences[seqIdx][DICT_SEQUENCE_NAME] == "candle_wind1" :
                self.preloadedTransitionCosts[seqIdx] = np.roll(np.roll(np.load(self.semanticSequences[seqIdx][DICT_TRANSITION_COSTS_LOCATION]), -1, 1)[::self.optimizationDownsampleRateSpinBox.value(),
                                                                                                                                                        ::self.optimizationDownsampleRateSpinBox.value()], 1, 1)
            else :
                self.preloadedTransitionCosts[seqIdx] = computeTransitionMatrix(self.preloadedDistanceMatrices[seqIdx][::self.optimizationDownsampleRateSpinBox.value(), ::self.optimizationDownsampleRateSpinBox.value()],
                                                                                filterSize/self.optimizationDownsampleRateSpinBox.value(), threshPercentile, minJumpSize/self.optimizationDownsampleRateSpinBox.value(),
                                                                                onlyBackwards, False, sigmaMultiplier)
            ## this below is for the case where I don't rememeber the transition matrix parameters like for the wave sequence
#             self.preloadedTransitionCosts[seqIdx] = np.roll(np.roll(np.load(self.semanticSequences[seqIdx][DICT_TRANSITION_COSTS_LOCATION]), -1, 1)[::self.optimizationDownsampleRateSpinBox.value(),
#                                                                                                                                                     ::self.optimizationDownsampleRateSpinBox.value()], 1, 1)
            
        print "TIME FOR COMPUTING TRANSITION MATRICES", time.time() - bob
            
    def getSequenceCompatibilityLabels(self, seq1Idx, conflictingSequenceKey, verbose=True) :
        ### gets compatibility labels using only the semantic classes plus any other class a conflict is defined for seqIdxs2
        
        ## get the relevant classes for the compatibility labels
        compatibilityClasses = list(np.arange(0, self.semanticSequences[seq1Idx][DICT_NUM_SEMANTICS])) ## the first n classes are always the semantic classes
        if DICT_CONFLICTING_SEQUENCES in self.semanticSequences[seq1Idx].keys() :
            if conflictingSequenceKey in self.semanticSequences[seq1Idx][DICT_CONFLICTING_SEQUENCES].keys() :
                for classPair in self.semanticSequences[seq1Idx][DICT_CONFLICTING_SEQUENCES][conflictingSequenceKey] :
                    if classPair[0] not in compatibilityClasses :
                        compatibilityClasses.append(classPair[0])
        if DICT_COMPATIBLE_SEQUENCES in self.semanticSequences[seq1Idx].keys() :
            if conflictingSequenceKey in self.semanticSequences[seq1Idx][DICT_COMPATIBLE_SEQUENCES].keys() :
                for classPair in self.semanticSequences[seq1Idx][DICT_COMPATIBLE_SEQUENCES][conflictingSequenceKey] :
                    if classPair[0] not in compatibilityClasses :
                        compatibilityClasses.append(classPair[0])
                        
        compatibilityClasses = np.sort(compatibilityClasses)
        
        compatibilityLabels = propagateLabels(self.preloadedDistanceMatrices[seq1Idx],
                                              np.array(self.semanticSequences[seq1Idx][DICT_LABELLED_FRAMES])[compatibilityClasses],
                                              np.array(self.semanticSequences[seq1Idx][DICT_NUM_EXTRA_FRAMES])[compatibilityClasses], verbose, self.propagationSigmaSpinBox.value()/10.0)#, 0.002)
        
        return compatibilityLabels, compatibilityClasses
    
    def getSequencePairCompatibilityLabels(self, seqIdxs, doOverride=False, verbose=True) :
        needsReverse = seqIdxs[0] > seqIdxs[1]
        matKey = np.string_(np.min(seqIdxs))+"-"+np.string_(np.max(seqIdxs))
        
        ## get precomputed one if existing
        if not doOverride and matKey in self.sequencePairCompatibilityLabels.keys() :
            if verbose :
                print "USING PRECOMPUTED PAIR COMPATIBILITY LABELS", matKey,
        else :
            startingTime = time.time()
            sequencePairCompatibilityLabels = []
            sequencePairCompatibilityLabels.append(self.getSequenceCompatibilityLabels(seqIdxs[0], self.semanticSequences[seqIdxs[1]][DICT_SEQUENCE_LOCATION], verbose))
            
            if seqIdxs[0] == seqIdxs[1] :
                sequencePairCompatibilityLabels.append(sequencePairCompatibilityLabels[0])
            else :
                sequencePairCompatibilityLabels.append(self.getSequenceCompatibilityLabels(seqIdxs[1], self.semanticSequences[seqIdxs[0]][DICT_SEQUENCE_LOCATION], verbose))
            
            if needsReverse :
                self.sequencePairCompatibilityLabels[matKey] = sequencePairCompatibilityLabels[::-1]
            else :
                self.sequencePairCompatibilityLabels[matKey] = sequencePairCompatibilityLabels
                
            if verbose :
                print "COMPUTED PAIR COMPATIBILITY LABELS", matKey, "in", time.time()-startingTime, "seconds"
        
        if needsReverse :
            if verbose :
                print "sequencePairCompatibilityLabels:", self.sequencePairCompatibilityLabels[matKey][0][0].shape,
                print self.sequencePairCompatibilityLabels[matKey][1][0].shape, "returned reversed"; sys.stdout.flush()
            return self.sequencePairCompatibilityLabels[matKey][::-1]
        else :
            if verbose :
                print "sequencePairCompatibilityLabels:", self.sequencePairCompatibilityLabels[matKey][0][0].shape,
                print self.sequencePairCompatibilityLabels[matKey][1][0].shape; sys.stdout.flush()
            return self.sequencePairCompatibilityLabels[matKey]
            
       
    #### MISCELLANEOUS INTERACTION CALLBACKS ####        

    def deleteGeneratedSequence(self) :
        del self.generatedSequence
        self.generatedSequence = []
        
        ## update sliders
        self.frameIdxSlider.setMaximum(0)
        self.frameIdxSpinBox.setRange(0, 0)
        
        self.frameInfo.setText("Info text")
        
        self.frameIdxSpinBox.setValue(0)
        
    def changeSelectedSemSequence(self, row) :
        if len(self.semanticSequences) > row.row() :
            self.selectedSemSequenceIdx = row.row()
        
        print "selected sequence", self.selectedSemSequenceIdx; sys.stdout.flush()
            
        self.setFocus()                
        
    def changeSelectedInstances(self, selected, deselected) :
        self.selectedSequenceInstancesIdxes = []
        for row in self.sequenceInstancesListTable.selectionModel().selectedRows() :
            self.selectedSequenceInstancesIdxes.append(row.row())
        
        for i in xrange(len(self.synthesisedSequence[DICT_SEQUENCE_INSTANCES])) :
            if i in self.selectedSequenceInstancesIdxes :
                self.sequenceInstancesListTable.setRowHeight(i, SLIDER_SELECTED_HEIGHT+2*SLIDER_PADDING)
            else :
                self.sequenceInstancesListTable.setRowHeight(i, SLIDER_NOT_SELECTED_HEIGHT+2*SLIDER_PADDING)
        
        if len(self.selectedSequenceInstancesIdxes) > 0 :
            if self.frameIdx > len(self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][self.selectedSequenceInstancesIdxes[-1]][DICT_SEQUENCE_FRAMES]) :
                self.frameIdxSpinBox.setValue(len(self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][self.selectedSequenceInstancesIdxes[-1]][DICT_SEQUENCE_FRAMES])-1)
            else :
                self.showFrame(self.frameIdx)
                
        self.setSemanticSliderDrawables()
        self.instancesShowIndicator.setSelectedInstances(self.selectedSequenceInstancesIdxes)
        
        print "selected instances", self.selectedSequenceInstancesIdxes; sys.stdout.flush()
        self.setFocus()
        if len(self.selectedSequenceInstancesIdxes) == 2 :
            ## make info image
            seqIdxs = np.array([self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][self.selectedSequenceInstancesIdxes[0]][DICT_SEQUENCE_IDX],
                                self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][self.selectedSequenceInstancesIdxes[1]][DICT_SEQUENCE_IDX]])
            
            self.makeInfoImage(seqIdxs)
            self.updateInfoDialog(np.array([0, 0]))
            
            
        ## reset height
        selectionHeight = len(self.selectedSequenceInstancesIdxes)*SLIDER_SELECTED_HEIGHT
        remainingHeight = (len(self.synthesisedSequence[DICT_SEQUENCE_INSTANCES])-len(self.selectedSequenceInstancesIdxes))*SLIDER_NOT_SELECTED_HEIGHT
        paddingHeight = (len(self.synthesisedSequence[DICT_SEQUENCE_INSTANCES]))*2*SLIDER_PADDING
        
        desiredHeight = np.max((SLIDER_MIN_HEIGHT, selectionHeight+remainingHeight+paddingHeight))
        
        self.sequenceInstancesListTable.setFixedHeight(desiredHeight)
            
    def makeInfoImage(self, seqIdxs) :
        if self.doShowCompatibilityInfo :
#             compatibilityMat = np.log(1.0+np.copy(self.getCompatibilityMat(seqIdxs)))
            compatibilityMat = np.copy(self.getCompatibilityMat(seqIdxs))
            self.plotScaleRatio = 1.0
            if np.max(np.array(compatibilityMat.shape)) > self.MAX_PLOT_SIZE-self.PLOT_HEIGHT-self.TEXT_SIZE :
                self.plotScaleRatio = (self.MAX_PLOT_SIZE-self.PLOT_HEIGHT-self.TEXT_SIZE)/float(np.max(np.array(compatibilityMat.shape)))
            ## normalize to [0, 1]    
    #         compatibilityMat += np.min(compatibilityMat)
            print "la", np.max(compatibilityMat), np.min(compatibilityMat)
            if np.max(compatibilityMat) != 0.0 :
                compatibilityMat /= np.max(compatibilityMat)
            print "bla", np.max(compatibilityMat), np.min(compatibilityMat)

    #         ## get compatibility labels for the 2 sequences
    #         seqCompatibilityLabels = [self.getFrameCompatibilityLabels(seqIdxs[0], self.semanticSequences[seqIdxs[1]][DICT_SEQUENCE_LOCATION]),
    #                                   self.getFrameCompatibilityLabels(seqIdxs[1], self.semanticSequences[seqIdxs[0]][DICT_SEQUENCE_LOCATION])]

    #         self.currentlySelectedSeqCompatLabels = [self.getFrameCompatibilityLabels(seqIdxs[0], self.semanticSequences[seqIdxs[1]][DICT_SEQUENCE_LOCATION]),
    #                                                  self.getFrameCompatibilityLabels(seqIdxs[1], self.semanticSequences[seqIdxs[0]][DICT_SEQUENCE_LOCATION])]

    #         print "COMPAT CLASSES", self.currentlySelectedSeqCompatLabels[0][1], self.currentlySelectedSeqCompatLabels[1][1]

            print "IN MAKE INFO IMAGE", np.max(compatibilityMat), np.min(compatibilityMat)
            sequencePairCompatibilityLabels = self.getSequencePairCompatibilityLabels(seqIdxs)
            print "COMPAT CLASSES", sequencePairCompatibilityLabels[0][1], sequencePairCompatibilityLabels[1][1]

            ## first make the image showing the compatiblity matrix
#             infoImg = mpl.cm.afmhot(compatibilityMat, bytes=True)
            infoImg = mpl.cm.jet(compatibilityMat, bytes=True)
            infoImg = cv2.resize(infoImg, (int(np.round(infoImg.shape[1]*self.plotScaleRatio)), int(np.round(infoImg.shape[0]*self.plotScaleRatio))))


            ## then make the images showing the stackplot of the labels
    #         if DICT_FRAME_COMPATIBILITY_LABELS in self.semanticSequences[seqIdxs[0]].keys() :
    #             infoImg = np.concatenate((valsToImg(makeStackPlot(self.semanticSequences[seqIdxs[0]][DICT_FRAME_COMPATIBILITY_LABELS], self.PLOT_HEIGHT).T), infoImg), axis=1)
    #         else :
    #             infoImg = np.concatenate((np.zeros((len(self.semanticSequences[seqIdxs[0]][DICT_FRAME_SEMANTICS]), self.PLOT_HEIGHT, 4), np.uint8), infoImg), axis=1)
    #         infoImg = np.concatenate((valsToImg(makeStackPlot(sequencePairCompatibilityLabels[0][0], self.PLOT_HEIGHT).T), infoImg), axis=1)
            infoImg = np.concatenate((cv2.resize(mpl.cm.Set1(makeStackPlot(sequencePairCompatibilityLabels[0][0], self.PLOT_HEIGHT).T, bytes=True),
                                                 (self.PLOT_HEIGHT, infoImg.shape[0])), infoImg), axis=1)


    #         if DICT_FRAME_COMPATIBILITY_LABELS in self.semanticSequences[seqIdxs[1]].keys() :
    #             infoImg = np.concatenate((np.concatenate((np.zeros((self.PLOT_HEIGHT, self.PLOT_HEIGHT, 4), np.uint8),
    #                                                       valsToImg(makeStackPlot(self.semanticSequences[seqIdxs[1]][DICT_FRAME_COMPATIBILITY_LABELS], self.PLOT_HEIGHT))), axis=1),
    #                                       infoImg), axis=0)
    #         else :
    #             infoImg = np.concatenate((np.zeros((self.PLOT_HEIGHT, len(self.semanticSequences[seqIdxs[1]][DICT_FRAME_SEMANTICS])+self.PLOT_HEIGHT, 4), np.uint8),
    #                                       infoImg), axis=0)
    #         infoImg = np.concatenate((np.concatenate((np.zeros((self.PLOT_HEIGHT, self.PLOT_HEIGHT, 4), np.uint8),
    #                                                   valsToImg(makeStackPlot(sequencePairCompatibilityLabels[1][0], self.PLOT_HEIGHT))), axis=1), infoImg), axis=0)
            infoImg = np.concatenate((np.concatenate((np.zeros((self.PLOT_HEIGHT, self.PLOT_HEIGHT, 4), np.uint8),
                                                      cv2.resize(mpl.cm.Set1(makeStackPlot(sequencePairCompatibilityLabels[1][0], self.PLOT_HEIGHT), bytes=True),
                                                                 (infoImg.shape[1]-self.PLOT_HEIGHT, self.PLOT_HEIGHT))), axis=1), infoImg), axis=0)

            ## finally make the labelCompatibility matrix image
            if (DICT_LABELLED_FRAMES in self.semanticSequences[seqIdxs[0]].keys() and
                DICT_LABELLED_FRAMES in self.semanticSequences[seqIdxs[1]].keys()) :

                cellSize = 100
                spacing = 10
                seq1NumLabels = sequencePairCompatibilityLabels[0][0].shape[1]
                seq2NumLabels = sequencePairCompatibilityLabels[1][0].shape[1]
                labelCompatibility = np.ones((seq1NumLabels, seq2NumLabels))

                ## setting labelCompatibility where there are conflicts if any
                seq2Location = self.semanticSequences[seqIdxs[1]][DICT_SEQUENCE_LOCATION]
                if (DICT_CONFLICTING_SEQUENCES in self.semanticSequences[seqIdxs[0]].keys() and
                    seq2Location in self.semanticSequences[seqIdxs[0]][DICT_CONFLICTING_SEQUENCES].keys()) :

                    for combination in self.semanticSequences[seqIdxs[0]][DICT_CONFLICTING_SEQUENCES][seq2Location] :
                        print "COMBINATION", combination
                        labelCompatibility[int(np.argwhere(np.array(sequencePairCompatibilityLabels[0][1]) == combination[0])),
                                           int(np.argwhere(np.array(sequencePairCompatibilityLabels[1][1]) == combination[1]))] = 0

                compatImg = np.ones((seq1NumLabels*cellSize+(1+seq1NumLabels)*spacing,
                                     seq2NumLabels*cellSize+(1+seq2NumLabels)*spacing, 4), np.uint8)*255
                compatImg[:, :, :-1] = np.array([128, 27, 128]).reshape((1, 1, 3))

                for i in xrange(labelCompatibility.shape[0]) :
                    for j in xrange(labelCompatibility.shape[1]) :
                        if labelCompatibility[i, j] == 1 :
                            clr = np.array([0, 255, 0]).reshape((1, 1, 3))
                        else :
                            clr = np.array([255, 0, 0]).reshape((1, 1, 3))
                        compatImg[spacing*(i+1)+cellSize*i:spacing*(i+1)+cellSize*(i+1),
                                  spacing*(j+1)+cellSize*j:spacing*(j+1)+cellSize*(j+1), :-1] = clr

                ## make space for label color scales
                labelScalesWidth = 40
                compatImgSize = compatImg.shape[0:2]
                compatImg = np.concatenate(((np.ones((compatImg.shape[0], labelScalesWidth+spacing, 4), np.uint8)*
                                             np.array([128, 27, 128, 255], np.uint8).reshape((1, 1, 4))), compatImg), axis=1)
                compatImg = np.concatenate(((np.ones((labelScalesWidth+spacing, compatImg.shape[1], 4), np.uint8)*
                                             np.array([128, 27, 128, 255], np.uint8).reshape((1, 1, 4))), compatImg), axis=0)
                
                ## render the label color scales
                compatImg[-compatImgSize[0]+spacing:-spacing,
                          spacing:spacing+labelScalesWidth, :] = mpl.cm.Set1(makeStackPlot(np.ones((labelScalesWidth, seq1NumLabels))/seq1NumLabels,
                                                                                         compatImgSize[0]-2*spacing), bytes=True)
                compatImg[spacing:spacing+labelScalesWidth,
                          -compatImgSize[1]+spacing:-spacing, :] = mpl.cm.Set1(makeStackPlot(np.ones((labelScalesWidth, seq2NumLabels))/seq2NumLabels,
                                                                                           compatImgSize[1]-2*spacing).T, bytes=True)

                infoImg[:self.PLOT_HEIGHT, :self.PLOT_HEIGHT, :] = cv2.resize(compatImg, (self.PLOT_HEIGHT, self.PLOT_HEIGHT))

            ## render sequence names
            textSize = self.TEXT_SIZE 
            seq1NameImg = np.ones((infoImg.shape[0], infoImg.shape[0], 4), np.uint8)*255
            cv2.putText(seq1NameImg, self.semanticSequences[seqIdxs[0]][DICT_SEQUENCE_NAME], (5, 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0, 255))
            M = cv2.getRotationMatrix2D((seq1NameImg.shape[1]/2,seq1NameImg.shape[0]/2),90,1)
            seq1NameImg = cv2.warpAffine(seq1NameImg,M,(seq1NameImg.shape[1],seq1NameImg.shape[0]))

            seq2NameImg = np.ones((textSize, infoImg.shape[1], 4), np.uint8)*255
            cv2.putText(seq2NameImg, self.semanticSequences[seqIdxs[1]][DICT_SEQUENCE_NAME], (5, 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0, 255))

            infoImg = np.concatenate((seq1NameImg[:, :textSize, :], infoImg), axis=1)
            infoImg = np.concatenate((np.concatenate((np.ones((textSize, textSize, 4), np.uint8)*255,
                                                      seq2NameImg), axis=1),
                                      infoImg), axis=0)
            infoImg[:textSize+3, :textSize, :] = 255


            infoImg = np.ascontiguousarray(infoImg[:, :, [2, 1, 0, 3]])
            self.infoImage = QtGui.QImage(infoImg.data, infoImg.shape[1], infoImg.shape[0], infoImg.strides[0], QtGui.QImage.Format_ARGB32);
            self.infoDialog.setInfoImage(self.infoImage)
            if self.infoDialog.isHidden() :
                self.infoDialog.show()
    
    def playSequenceButtonPressed(self) :
        if self.doPlaySequence :
            self.doPlaySequence = False
            self.playSequenceButton.setIcon(self.playIcon)
            self.playTimer.stop()
            
            self.frameInfo.setText(self.oldInfoText)
        else :
            self.lastRenderTime = time.time()
            self.doPlaySequence = True
            self.playSequenceButton.setIcon(self.pauseIcon)
            self.playTimer.start()
            
            self.oldInfoText = self.frameInfo.text()
            
    def setRenderFps(self, value) :
        self.playTimer.setInterval(1000/value)
        
    def setToggleDelay(self, value) :
        self.TOGGLE_DELAY = value
        
    def setExtendLength(self, value) :
        self.EXTEND_LENGTH = value + 1
                    
    def toggleInstancesDoShow(self, index) :
        self.instancesDoShow[index] = not self.instancesDoShow[index]
        self.showFrame(self.frameIdx)
        
        self.setFocus()
       
    #### HANDLE INSERTION OF NEW SEMANTIC INSTANCE AND CALLBACKS ####        

    def initNewSemanticSequenceInstance(self, frameKey) :
        frameSize = np.array(Image.open(self.semanticSequences[self.selectedSemSequenceIdx][DICT_FRAMES_LOCATIONS][self.semanticSequences[self.selectedSemSequenceIdx][DICT_FRAMES_LOCATIONS].keys()[0]])).shape[:2]
        if DICT_BBOXES in self.semanticSequences[self.selectedSemSequenceIdx].keys() and frameKey in self.semanticSequences[self.selectedSemSequenceIdx][DICT_BBOXES].keys() :
            tl = np.min(self.semanticSequences[self.selectedSemSequenceIdx][DICT_BBOXES][frameKey], axis=0)
            br = np.max(self.semanticSequences[self.selectedSemSequenceIdx][DICT_BBOXES][frameKey], axis=0)
        else :
            tl = np.array([0.0, 0.0], float)
            br = np.array([frameSize[1], frameSize[0]], float)
            
        w, h = br-tl
        self.newSemanticSequenceAABB = np.array([tl, tl + [w, 0], br, tl + [0, h]])
        
        if frameKey in self.semanticSequences[self.selectedSemSequenceIdx][DICT_FRAMES_LOCATIONS].keys() :
            frameName = self.semanticSequences[self.selectedSemSequenceIdx][DICT_FRAMES_LOCATIONS][frameKey].split(os.sep)[-1]
            if DICT_MASK_LOCATION in self.semanticSequences[self.selectedSemSequenceIdx].keys() :
                if os.path.isfile(self.semanticSequences[self.selectedSemSequenceIdx][DICT_MASK_LOCATION]+frameName) :
                    self.newSemanticSequenceImg = np.array(Image.open(self.semanticSequences[self.selectedSemSequenceIdx][DICT_MASK_LOCATION]+frameName))[:, :, [2, 1, 0, 3]]
                else :
                    self.newSemanticSequenceImg = np.zeros([frameSize[0], frameSize[1], 4], np.uint8)
            else :
                self.newSemanticSequenceImg = np.array(Image.open(self.semanticSequences[self.selectedSemSequenceIdx][DICT_FRAMES_LOCATIONS][frameKey]))

            self.newSemanticSequenceImg = np.ascontiguousarray(self.newSemanticSequenceImg[self.newSemanticSequenceAABB[0, 1]:self.newSemanticSequenceAABB[2, 1],
                                                                                           self.newSemanticSequenceAABB[0, 0]:self.newSemanticSequenceAABB[2, 0], :])
        else :
            self.newSemanticSequenceImg = np.empty(0)

        self.updateNewSemanticSequenceTransformation()
        
    def currentStartingSemanticsChanged(self, index) :
        if self.selectedSemSequenceIdx >= 0 and self.selectedSemSequenceIdx < len(self.semanticSequences) :
            frameIdx = int(np.argwhere(self.semanticSequences[self.selectedSemSequenceIdx][DICT_FRAME_SEMANTICS][:, index] >= 0.9)[0])
            frameKey = np.sort(self.semanticSequences[self.selectedSemSequenceIdx][DICT_FRAMES_LOCATIONS].keys())[frameIdx]
            self.initNewSemanticSequenceInstance(frameKey)
        
        self.setFocus()
        
    def cancelNewSemanticSequence(self) :
        self.showFrame(self.frameIdx)
        
        ### UI stuff ###
#         self.addNewSemanticSequenceControls.setVisible(False)
        self.addNewSemanticSequenceControls.done(0)
        self.frameIdxSlider.setEnabled(True)
        self.frameIdxSpinBox.setEnabled(True)
        
    def updateNewSemanticSequenceTransformation(self) :
        
        aabb = self.newSemanticSequenceAABB + np.array([[self.sequenceXOffsetSpinBox.value(), self.sequenceYOffsetSpinBox.value()]])
        w, h = aabb[2, :]-aabb[0, :]
        aabb = np.array([aabb[0, :], aabb[0, :] + [w*self.sequenceXScaleSpinBox.value(), 0], 
                         aabb[0, :] + [w*self.sequenceXScaleSpinBox.value(), h*self.sequenceYScaleSpinBox.value()],
                         aabb[0, :] + [0, h*self.sequenceYScaleSpinBox.value()]], np.int32)
        
                
        self.showFrame(self.frameIdx)

        if np.any(self.overlayImg != None) :
            painter = QtGui.QPainter(self.overlayImg)
            
            if np.all(np.array(self.newSemanticSequenceImg.shape) != 0) :
                ## draw sprite
                if self.newSemanticSequenceImg.shape[-1] == 3 :
                    img = QtGui.QImage(self.newSemanticSequenceImg.data, self.newSemanticSequenceImg.shape[1],
                                       self.newSemanticSequenceImg.shape[0], self.newSemanticSequenceImg.strides[0], QtGui.QImage.Format_RGB888)
                else :
                    img = QtGui.QImage(self.newSemanticSequenceImg.data, self.newSemanticSequenceImg.shape[1],
                                       self.newSemanticSequenceImg.shape[0], self.newSemanticSequenceImg.strides[0], QtGui.QImage.Format_ARGB32)

                painter.drawImage(QtCore.QPoint(aabb[0, 0], aabb[0, 1]), img.scaled(img.width()*self.sequenceXScaleSpinBox.value(),
                                                                                    img.height()*self.sequenceYScaleSpinBox.value()))

            ## draw AABB
            painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(0, 255, 0, 255), 3, 
                                      QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))

            for p1, p2 in zip(np.mod(np.arange(4), 4), np.mod(np.arange(1, 5), 4)) :
                painter.drawLine(QtCore.QPointF(aabb[p1, 0], aabb[p1, 1]), QtCore.QPointF(aabb[p2, 0], aabb[p2, 1]))

            self.frameLabel.setOverlay(self.overlayImg)
            
            painter.end()
            
    def addNewSemanticSequence(self) :
        if self.selectedSemSequenceIdx >= 0 and self.selectedSemSequenceIdx < len(self.semanticSequences) :
            usedSequenceIdx = 0
            if self.semanticSequences[self.selectedSemSequenceIdx][DICT_SEQUENCE_LOCATION] not in self.synthesisedSequence[DICT_USED_SEQUENCES] :
                self.synthesisedSequence[DICT_USED_SEQUENCES].append(self.semanticSequences[self.selectedSemSequenceIdx][DICT_SEQUENCE_LOCATION])
                usedSequenceIdx = len(self.synthesisedSequence[DICT_USED_SEQUENCES])-1
            else :
                usedSequenceIdx = np.argwhere(np.array(self.synthesisedSequence[DICT_USED_SEQUENCES]) ==
                                              self.semanticSequences[self.selectedSemSequenceIdx][DICT_SEQUENCE_LOCATION])[0]
            
            self.synthesisedSequence[DICT_SEQUENCE_INSTANCES].append({
                                             DICT_SEQUENCE_IDX:int(usedSequenceIdx),
                                             DICT_OFFSET:np.array([self.sequenceXOffsetSpinBox.value(), self.sequenceYOffsetSpinBox.value()]),
                                             DICT_SCALE:np.array([self.sequenceXScaleSpinBox.value(), self.sequenceYScaleSpinBox.value()]),
                                             DICT_SEQUENCE_FRAMES:np.zeros(1, dtype=int),
                                             DICT_DESIRED_SEMANTICS:np.zeros((1, self.semanticSequences[self.selectedSemSequenceIdx][DICT_NUM_SEMANTICS]), dtype=float)
                                             })
            self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][-1][DICT_SEQUENCE_FRAMES][0] = int(np.argwhere((self.semanticSequences[self.selectedSemSequenceIdx][DICT_FRAME_SEMANTICS]
                                                                                                              [:, self.startingSemanticsComboBox.currentIndex()]) >= 0.9)[0])
            self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][-1][DICT_DESIRED_SEMANTICS][0, self.startingSemanticsComboBox.currentIndex()] = 1.0
            ## if there are sequence instances in the synthesised sequence I need to fill in with placeholders so that all instances have the same length
#             if len(self.synthesisedSequence[DICT_SEQUENCE_INSTANCES]) > 1 :
#                 maxFrames = len(self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][0][DICT_SEQUENCE_FRAMES])
#                 if maxFrames > 1 :
            self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][-1][DICT_SEQUENCE_FRAMES] = np.concatenate((-np.ones(self.frameIdx, dtype=int), 
                                                                                                          self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][-1][DICT_SEQUENCE_FRAMES]))
            self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][-1][DICT_DESIRED_SEMANTICS] = np.concatenate((np.zeros((self.frameIdx, self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][-1][DICT_DESIRED_SEMANTICS].shape[1])),
                                                                                                                    self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][-1][DICT_DESIRED_SEMANTICS]))
            self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][-1][DICT_DESIRED_SEMANTICS][:self.frameIdx, self.startingSemanticsComboBox.currentIndex()] = 1.0
            
            self.instancesDoShow.append(True)
            
            ### UI stuff ###
            self.selectedSequenceInstancesIdxes = [len(self.synthesisedSequence[DICT_SEQUENCE_INSTANCES])-1]
            self.setSemanticSliderDrawables()
            self.setListOfSequenceInstances()
            self.setInstanceShowIndicatorDrawables()
            self.instancesShowIndicator.setSelectedInstances(self.selectedSequenceInstancesIdxes)
            self.sequenceInstancesListTable.selectRow(self.selectedSequenceInstancesIdxes[-1])
        
            
#         self.addNewSemanticSequenceControls.setVisible(False)
        self.addNewSemanticSequenceControls.done(1)
        self.frameIdxSlider.setEnabled(True)
        self.frameIdxSpinBox.setEnabled(True)
       
    #### HANDLE KEY AND BUTTON PRESSES ####
    
    def loadNumberSequence(self) :
        for i in xrange(len(self.synthesisedSequence[DICT_SEQUENCE_INSTANCES])) :
            seqIdx = self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][i][DICT_SEQUENCE_IDX]
            
            doAskQuestion = False
            for actionKey in np.sort(self.semanticSequences[seqIdx][DICT_COMMAND_TYPE].keys()) :
                if self.semanticSequences[seqIdx][DICT_COMMAND_TYPE][actionKey] != DICT_COMMAND_TYPE_COLOR :
                    doAskQuestion = True
                    break
            
            if doAskQuestion :
                proceed = QtGui.QMessageBox.question(self, 'Wrong Command Binding', "Some actor sequences' action commands are not bound to colors.\nWould you like to open the Action Command Bindings Dialog?", 
                                                     QtGui.QMessageBox.Yes | QtGui.QMessageBox.No, QtGui.QMessageBox.No) == QtGui.QMessageBox.Yes
                if proceed :
                    changeActionsCommandBindings(self, "Change Action Command Bindings", self.semanticSequences)
                break
            
        numberSequenceDir = QtGui.QFileDialog.getExistingDirectory(self, "Load Number Sequence", self.dataPath)
        
        if numberSequenceDir != "" :
            numbersSequenceFrames = np.sort(glob.glob(numberSequenceDir+"/numbers-frame-*.png"))
            if len(numbersSequenceFrames) > 0 :
                startTime = time.time()
                frameShape = np.array(Image.open(numbersSequenceFrames[0])).shape
                allFrames = np.zeros([len(numbersSequenceFrames), frameShape[0], frameShape[1], frameShape[2]], np.uint8)
                for i, frameLoc in enumerate(numbersSequenceFrames) :
                    allFrames[i, :, :, :] = np.array(Image.open(frameLoc)).astype(np.uint8)
                    
#                 classColors = np.array([[0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
                    
                for i in xrange(len(self.synthesisedSequence[DICT_SEQUENCE_INSTANCES])) :
                    seqIdx = self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][i][DICT_SEQUENCE_IDX]
                    
                    doUseSequence = True
                    classColors = np.ones([len(self.semanticSequences[seqIdx][DICT_COMMAND_TYPE].keys()), 4])
                    for actionKey in np.sort(self.semanticSequences[seqIdx][DICT_COMMAND_TYPE].keys()) :
                        if self.semanticSequences[seqIdx][DICT_COMMAND_TYPE][actionKey] != DICT_COMMAND_TYPE_COLOR :
                            doUseSequence = False
                            break
                        else :
                            classColors[actionKey, :-1] = self.semanticSequences[seqIdx][DICT_COMMAND_BINDING][actionKey]/255.0
                            print actionKey, self.semanticSequences[seqIdx][DICT_COMMAND_BINDING][actionKey]/255.0
                    
                    if not doUseSequence :
                        print "NOT USING SEQUENCE", self.semanticSequences[seqIdx][DICT_SEQUENCE_NAME], "BECAUSE ACTION COMMANDS NOT BOUND TO COLORS"
                        continue
                    else :
                        print classColors
                    
                    frameIdx = self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][i][DICT_SEQUENCE_FRAMES][0]
                    frameKey = np.sort(self.semanticSequences[seqIdx][DICT_FRAMES_LOCATIONS].keys())[frameIdx]
                    sigma = 0.2
                    bboxCenter = np.array([0, 0])
                    if DICT_BBOX_CENTERS in self.semanticSequences[seqIdx].keys() :
                        if frameKey in self.semanticSequences[seqIdx][DICT_BBOX_CENTERS].keys() :
                            bboxCenter = np.array(self.semanticSequences[seqIdx][DICT_BBOX_CENTERS][frameKey], int)
                        else :
                            bboxCenter = np.array(self.semanticSequences[seqIdx][DICT_BBOX_CENTERS][self.semanticSequences[seqIdx][DICT_BBOX_CENTERS].keys()[0]], int)
                            
                    bboxCenter = (bboxCenter*self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][i][DICT_SCALE])+self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][i][DICT_OFFSET]
                    
                    classDists = np.sqrt(np.sum((allFrames[0, bboxCenter[1], bboxCenter[0], :]/255.0 - classColors)**2, axis=1))
                    print 0, allFrames[0, bboxCenter[1], bboxCenter[0], :]/255.0, classDists, np.exp(-classDists/sigma), np.exp(-classDists/sigma)/np.sum(np.exp(-classDists/sigma))
#                     classDists /= np.sum(classDists)
                    classDists = np.exp(-classDists/sigma)/np.sum(np.exp(-classDists/sigma))
                    ## re-set first frame and delete all the rest
                    self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][i][DICT_SEQUENCE_FRAMES] = np.zeros(1, dtype=int)
                    self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][i][DICT_SEQUENCE_FRAMES][0] = int(np.random.choice(np.argwhere((self.semanticSequences[seqIdx][DICT_FRAME_SEMANTICS]
                                                                                                                                      [:, int(np.argmax(classDists))]) > 0.99).flatten()))
                    ## re-set first desired semantics and delete all the rest
                    self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][i][DICT_DESIRED_SEMANTICS] = classDists.reshape((1, len(classColors)))
                    
                    for j in xrange(1, allFrames.shape[0]) :
                        classDists = np.sqrt(np.sum((allFrames[j, bboxCenter[1], bboxCenter[0], :]/255.0 - classColors)**2, axis=1))
#                         print j, allFrames[j, bboxCenter[1], bboxCenter[0], :]/255.0, classDists, np.exp(-classDists/sigma), np.exp(-classDists/sigma)/np.sum(np.exp(-classDists/sigma))
#                         classDists /= np.sum(classDists)
                        classDists = np.exp(-classDists/sigma)/np.sum(np.exp(-classDists/sigma))
                        self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][i][DICT_DESIRED_SEMANTICS] = np.concatenate((self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][i][DICT_DESIRED_SEMANTICS],
                                                                                                                       classDists.reshape(1, len(classColors))))
#                         print bboxCenter, allFrames[j, bboxCenter[1], bboxCenter[0],  :], classDists, self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][i][DICT_DESIRED_SEMANTICS].shape
                print "DONE IN", time.time() - startTime
                
                del allFrames
                
                self.extendLengthSpinBox.setValue(len(numbersSequenceFrames)-1)
                self.numbersSequenceFrames = numbersSequenceFrames
                self.showNumbersSequence = True
                
                self.frameIdxSlider.setMaximum(len(numbersSequenceFrames)-1)
                self.frameIdxSpinBox.setRange(0, len(numbersSequenceFrames)-1)
                if self.frameIdx == 0 :
                    self.showFrame(self.frameIdx)
                else :
                    self.frameIdxSpinBox.setValue(0)

                self.setSemanticSliderDrawables()
                self.setInstanceShowIndicatorDrawables()
                self.frameInfo.setText("Loaded sequence at " + self.loadedSynthesisedSequence)
            else :
                self.showNumbersSequence = False
                
        self.setFocus()
                
    
    def findExampleAmongLabelledFrames(self, example, seqIdx, classIdxs, numExtraFrames) :
        ## returns index of class in DICT_LABELLED_FRAMES
        if example < 0 :
            raise Exception("Tagging a frame of {0} with a negative index!\nPlease pick another one.".format(self.semanticSequences[seqIdx][DICT_SEQUENCE_NAME]))
        ## check if I already labelled the current frame and if the user says so delete it from the other classes
        if DICT_LABELLED_FRAMES in self.semanticSequences[seqIdx].keys() :
            for idx, classIdx in enumerate(classIdxs) :
                classLabelledFrames = np.array(self.semanticSequences[seqIdx][DICT_LABELLED_FRAMES][classIdx])
                classNumExtraFrames = np.array(self.semanticSequences[seqIdx][DICT_NUM_EXTRA_FRAMES][classIdx])
                targetFrames = np.arange(example-numExtraFrames/2, example+numExtraFrames/2+1).reshape((1, numExtraFrames+1))
                found = np.any(np.abs(targetFrames - classLabelledFrames.reshape((len(classLabelledFrames), 1))) <= classNumExtraFrames.reshape((len(classNumExtraFrames), 1))/2, axis=1)
#                             found = np.abs(example-classLabelledFrames) <= classNumExtraFrames/2
                if np.any(found) :
                    if classIdx < self.semanticSequences[self.selectedSemSequenceIdx][DICT_NUM_SEMANTICS] :
                        QtGui.QMessageBox.critical(self, "Error Tagging Frame", ("<p align='center'>The current frame of {0}".format(self.semanticSequences[seqIdx][DICT_SEQUENCE_NAME]) +
                                                                                 " is used for defining semantics.<br><b>Aborting...</b></p>"))
                        return False, -1

                    if np.all(found) :
                        text = "<p align='center'>The current frame of {1} has previously been labelled as <u>compatibility</u> class {0}. "
                        text += "If it is overwritten, class {0} will have no remaining examples and <b>will</b> not influence compatibility any longer.<br>Do you want to proceed?</p>"
                        text = text.format(idx, self.semanticSequences[seqIdx][DICT_SEQUENCE_NAME])
                    else :
                        text = "<p align='center'>The current frame of {1} has previously been labelled as <u>compatibility</u> class {0}.<br>"
                        text += "Do you want to override?</p>"
                        text = text.format(idx, self.semanticSequences[seqIdx][DICT_SEQUENCE_NAME])
                    return True, classIdx
#                     proceed = QtGui.QMessageBox.question(self, 'Frame already labelled', text, 
#                                                          QtGui.QMessageBox.Yes | QtGui.QMessageBox.No, QtGui.QMessageBox.No) == QtGui.QMessageBox.Yes
#                     if proceed :
#                         self.semanticSequences[seqIdx][DICT_LABELLED_FRAMES][classIdx] = [x for i, x in enumerate(classLabelledFrames) if not found[i]]
#                         self.semanticSequences[seqIdx][DICT_NUM_EXTRA_FRAMES][classIdx] = [x for i, x in enumerate(classNumExtraFrames) if not found[i]]
                        
        return True, -1
    
    def tagSequenceInstancesFrames(self, selectedInstances, selectedFrames, isCompatible) :
        ### CAREFUL THIS ASSUMES THAT THERE IS ALWAYS JUST ONE PAIR OF INSTANCES AND IT'S THE PAIR THAT THE INFO DIALOG IS SHOWING ###
        numExtraFrames = 4
        if len(selectedInstances) > 1 :
            changedSequences = []
            seqIdxs = []
            ## for every pair of sequences in selectedInstances
            for idxs in np.argwhere(np.triu(np.ones(len(selectedInstances)), k=1)) : 
                currentPair = np.array(selectedInstances).flatten()[idxs]
                currentPairFrames = selectedFrames[idxs]
                
                ## get compatibility labels for current pair
                seq1Idx = self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][currentPair[0]][DICT_SEQUENCE_IDX]
                seq2Idx = self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][currentPair[1]][DICT_SEQUENCE_IDX]
                
                print "IN TAG CONFLICTS",
                sequencePairCompatibilityLabels = self.getSequencePairCompatibilityLabels([seq1Idx, seq2Idx])
                seq1NumLabels = len(sequencePairCompatibilityLabels[0][1])
                seq2NumLabels = len(sequencePairCompatibilityLabels[1][1])
                seq1Location = self.semanticSequences[seq1Idx][DICT_SEQUENCE_LOCATION]
                seq2Location = self.semanticSequences[seq2Idx][DICT_SEQUENCE_LOCATION]
                
                ## setting labelCompatibility where there are conflicts if any
                labelCompatibility = np.ones((seq1NumLabels, seq2NumLabels))
                ## duplicated in get comaptibility mat
                if (DICT_CONFLICTING_SEQUENCES in self.semanticSequences[seq1Idx].keys() and
                    seq2Location in self.semanticSequences[seq1Idx][DICT_CONFLICTING_SEQUENCES].keys()) :
                    for combination in self.semanticSequences[seq1Idx][DICT_CONFLICTING_SEQUENCES][seq2Location] :
                        labelCompatibility[int(np.argwhere(np.array(sequencePairCompatibilityLabels[0][1]) == combination[0])),
                                           int(np.argwhere(np.array(sequencePairCompatibilityLabels[1][1]) == combination[1]))] = 0
                        
                ## check if the chosen frames are already used as examples for other classes
                try :
                    proceed1, class1Idx = self.findExampleAmongLabelledFrames(currentPairFrames[0], seq1Idx, sequencePairCompatibilityLabels[0][1], numExtraFrames)
                    proceed2, class2Idx = self.findExampleAmongLabelledFrames(currentPairFrames[1], seq2Idx, sequencePairCompatibilityLabels[1][1], numExtraFrames)
                except Exception as e :
                    QtGui.QMessageBox.critical(self, "Error", (str(e)))
                    break
                
                if proceed1 and proceed2 :
                    doAbort = False
                    ## index of class in compatibilityLabels
                    currentFrame1Class = int(np.argmax(sequencePairCompatibilityLabels[0][0][currentPairFrames[0], :]))
                    currentFrame2Class = int(np.argmax(sequencePairCompatibilityLabels[1][0][currentPairFrames[1], :]))
                    
                    doApplyChange = [True, True]
                    doSwitchStatus = False
                    doSplit = [False, False]
                    
                    ## trying to change the compatibility status of current frame
                    if ((isCompatible and labelCompatibility[currentFrame1Class, currentFrame2Class] == 0) or
                        (not isCompatible and labelCompatibility[currentFrame1Class, currentFrame2Class] == 1)) :
                        ## ask if you want to switch the compatibility status
                        if isCompatible :
                            text = "<p align='center'>Frames {0} of {1} and {2} of {3} are currently <b>Incompatible</b>."
                            text += "<br>Do you want to mark them as <b>Compatible</b>?</p>"
                            text = text.format(currentPairFrames[0], self.semanticSequences[seq1Idx][DICT_SEQUENCE_NAME],
                                               currentPairFrames[1], self.semanticSequences[seq2Idx][DICT_SEQUENCE_NAME])
                            
                            title = "Tagging Compatible Frames"
                        else :
                            text = "<p align='center'>Frames {0} of {1} and {2} of {3} are currently <b>Compatible</b>."
                            text += "<br>Do you want to mark them as <b>Incompatible</b>?</p>"
                            text = text.format(currentPairFrames[0], self.semanticSequences[seq1Idx][DICT_SEQUENCE_NAME],
                                               currentPairFrames[1], self.semanticSequences[seq2Idx][DICT_SEQUENCE_NAME])
                            
                            title = "Tagging Incompatible Frames"

                        msgBox = QtGui.QMessageBox(QtGui.QMessageBox.Question, title, text, parent=self)
                        cancelButton = msgBox.addButton("Cancel", QtGui.QMessageBox.DestructiveRole)
                        msgBox.addButton(QtGui.QMessageBox.Yes)
#                                 msgBox.addButton(QtGui.QMessageBox.No)
#                                 msgBox.setDefaultButton(QtGui.QMessageBox.No)
                        msgBox.setDefaultButton(cancelButton)
                        msgBox.setEscapeButton(cancelButton)

                        answer = msgBox.exec_()
                        ## only go on if esc, cancel or X are not pressed
                        if msgBox.buttonRole(msgBox.clickedButton()) != QtGui.QMessageBox.DestructiveRole :
                            if answer == QtGui.QMessageBox.Yes :
                                ## if it is not an example then split
                                if class1Idx < 0 :
                                    doSplit[0] = True
                                ## if it is not an example then split
                                if class2Idx < 0 :
                                    doSplit[1] = True
                                ## no matter what though need to update dicts to switch status
                                doSwitchStatus = True
                        else :
                            doAbort = True
                    ## trying to add another example of compatibility/incompatibility
                    else :
                        for idx, classIdx, seqIdx in zip([0, 1], [class1Idx, class2Idx], [seq1Idx, seq2Idx]) :
                            ## only deal with this if the current frame is not already an example
                            if not doAbort and classIdx < 0 :
                                ## ask if you want to switch the compatibility status
                                if isCompatible :
                                    text = "<p align='center'>This frame of {0} is already <b>Compatible</b>."
                                    text += "<br>Would you like to <i>Refine</i> current compatibility measure or <i>Specialize</i>?</p>"
                                    text = text.format(self.semanticSequences[seqIdx][DICT_SEQUENCE_NAME])
                                    
                                    title = "Tagging Compatible Frames"
                                else :
                                    text = "<p align='center'>This frame of {0} is already <b>Incompatible</b>."
                                    text += "<br>Would you like to <i>Refine</i> current compatibility measure or <i>Specialize</i>?</p>"
                                    text = text.format(self.semanticSequences[seqIdx][DICT_SEQUENCE_NAME])
                                    
                                    title = "Tagging Incompatible Frames"

                                msgBox = QtGui.QMessageBox(QtGui.QMessageBox.Question, title, text, parent=self)
                                cancelButton = msgBox.addButton("Cancel", QtGui.QMessageBox.DestructiveRole)
                                refineButton = msgBox.addButton("Refine", QtGui.QMessageBox.NoRole)
                                specializeButton = msgBox.addButton("Specialize", QtGui.QMessageBox.YesRole)
                                msgBox.setDefaultButton(specializeButton)
                                msgBox.setEscapeButton(cancelButton)
                                msgBox.setDetailedText("If you choose to Refine, the selected frame will be used as an example of a situation that has been seen already. "+
                                                       "If you choose to Specialize, the selected frame will be used as an example for a new situation.")

                                msgBox.exec_()
                                
                                ## only go on if esc, cancel or X are not pressed
                                if msgBox.buttonRole(msgBox.clickedButton()) != QtGui.QMessageBox.DestructiveRole :
                                    if msgBox.clickedButton() == specializeButton :
                                        doSplit[idx] = True
                                    else :
                                        doSplit[idx] = False
                                    doApplyChange[idx] = True
                                else :
                                    doAbort = True
                        

                    ## now perform the changes requested by the user
                    if not doAbort : #and np.all(doApplyChange) :
                        taggedLabels = []
                        seq1Location = self.semanticSequences[seq1Idx][DICT_SEQUENCE_LOCATION]
                        seq2Location = self.semanticSequences[seq2Idx][DICT_SEQUENCE_LOCATION]
                        print "CHANGING STUFF (apply, switch, split)", doApplyChange, doSwitchStatus, doSplit; sys.stdout.flush()
                        
                        ## deal with dict keys :
                        for seqIdx in [seq1Idx, seq2Idx] :
                            if DICT_LABELLED_FRAMES not in self.semanticSequences[seqIdx].keys() :
                                self.semanticSequences[seqIdx][DICT_LABELLED_FRAMES] = []
                            if DICT_NUM_EXTRA_FRAMES not in self.semanticSequences[seqIdx].keys() :
                                self.semanticSequences[seqIdx][DICT_NUM_EXTRA_FRAMES] = []

                            if isCompatible :
                                ## deal with compatible sequences
                                if DICT_COMPATIBLE_SEQUENCES not in self.semanticSequences[seqIdx].keys() :
                                    self.semanticSequences[seqIdx][DICT_COMPATIBLE_SEQUENCES] = {}
                            else :
                                ## deal with the conflicting sequences
                                if DICT_CONFLICTING_SEQUENCES not in self.semanticSequences[seqIdx].keys() :
                                    self.semanticSequences[seqIdx][DICT_CONFLICTING_SEQUENCES] = {}
                            
                        if doSwitchStatus :
                            print "SWITCHING"
                            if class1Idx >= 0 and class2Idx >= 0 :                                
                                for idx, seqIdx, dictKey, labels in zip([0, 1], [seq1Idx, seq2Idx], [seq2Location, seq1Location], [[class1Idx, class2Idx], [class2Idx, class1Idx]]) :
                                    keyToRemoveFrom = DICT_COMPATIBLE_SEQUENCES
                                    keyToAddTo = DICT_CONFLICTING_SEQUENCES
                                    if isCompatible :
                                        keyToRemoveFrom = DICT_CONFLICTING_SEQUENCES
                                        keyToAddTo = DICT_COMPATIBLE_SEQUENCES
                                        
                                    print "REMOVING", labels, "FROM", keyToRemoveFrom, 
                                    

                                    if (keyToRemoveFrom in self.semanticSequences[seqIdx].keys() and dictKey in self.semanticSequences[seqIdx][keyToRemoveFrom].keys() and
                                        labels in self.semanticSequences[seqIdx][keyToRemoveFrom][dictKey]) :
                                        self.semanticSequences[seqIdx][keyToRemoveFrom][dictKey] = [pair for pair in self.semanticSequences[seqIdx][keyToRemoveFrom][dictKey] if pair != labels]
                                    
                                    print "AND ADDING", labels, "TO", keyToAddTo, "OF", idx
                                    
                                    if dictKey not in self.semanticSequences[seqIdx][keyToAddTo].keys() :
                                        self.semanticSequences[seqIdx][keyToAddTo][dictKey] = []
                                    if labels not in self.semanticSequences[seqIdx][keyToAddTo][dictKey] :
                                        self.semanticSequences[seqIdx][keyToAddTo][dictKey].append(labels)
                                        
                                    ## check if I'm adding same examples to 2 different classes for the same sequence
                                    if seq1Idx == seq2Idx and np.abs(currentPairFrames[0] - currentPairFrames[1]) <= numExtraFrames :
                                        print "BREAK BECAUSE SAME EXAMPLES"
                                        break
                            else :
                                for idx, classIdx, seqIdx, currentPairFrame in zip([0, 1], [class1Idx, class2Idx], [seq1Idx, seq2Idx], currentPairFrames) :
                                    ## didn't find the frame
                                    if classIdx < 0 :
                                        print "ADDING NEW CLASS", len(self.semanticSequences[seqIdx][DICT_LABELLED_FRAMES]), "TO", idx, "AND FRAME", currentPairFrame, "TO IT"
                                        
                                        self.semanticSequences[seqIdx][DICT_LABELLED_FRAMES].append([currentPairFrame])
                                        self.semanticSequences[seqIdx][DICT_NUM_EXTRA_FRAMES].append([numExtraFrames])
                                        taggedLabels.append(len(self.semanticSequences[seqIdx][DICT_LABELLED_FRAMES])-1)
                                    else :
                                        print "USING EXISTING CLASS", classIdx, "OF", idx, "FOR FRAME", currentPairFrame
                                        taggedLabels.append(classIdx)
                                        
                                    ## check if I'm adding same examples to 2 different classes for the same sequence
                                    if seq1Idx == seq2Idx and np.abs(currentPairFrames[0] - currentPairFrames[1]) <= numExtraFrames :
                                        print "BREAK BECAUSE SAME EXAMPLES"
                                        taggedLabels.append(taggedLabels[0])
                                        break
                                        
                            ## keep track of sequences that have changed as the compatibility matrices with all the other sequences will have to change
                            if seq1Idx not in changedSequences :
                                changedSequences.append(seq1Idx)
                            if seq2Idx not in changedSequences :
                                changedSequences.append(seq2Idx)
                        else :
                            print "NOT SWITCHING"
                            if class1Idx >= 0 and class2Idx >= 0 :
                                print "BOTH FRAMES ALREADY IN EXPECTED CLASSES"
                            else :  
                                for idx, classIdx, seqIdx, currentPairFrame, [compatibilityLabels, compatibilityClasses], currentFrameClass in zip([0, 1], [class1Idx, class2Idx],
                                                                                                                                                   [seq1Idx, seq2Idx], currentPairFrames,
                                                                                                                                                   sequencePairCompatibilityLabels, 
                                                                                                                                                   [currentFrame1Class, currentFrame2Class]) :
                                    ## didn't find the frame
                                    if classIdx < 0 :
                                        ## if I'm splitting it means I need to add a new class into labelled_frames
                                        if doSplit[idx] :
                                            print "ADDING NEW CLASS", len(self.semanticSequences[seqIdx][DICT_LABELLED_FRAMES]), "TO", idx, "AND FRAME", currentPairFrame, "TO IT"
                                            
                                            self.semanticSequences[seqIdx][DICT_LABELLED_FRAMES].append([currentPairFrame])
                                            self.semanticSequences[seqIdx][DICT_NUM_EXTRA_FRAMES].append([numExtraFrames])
                                            taggedLabels.append(len(self.semanticSequences[seqIdx][DICT_LABELLED_FRAMES])-1)
                                        else :
                                            print "USING EXISTING CLASS", compatibilityClasses[currentFrameClass], "OF", idx, "AND ADDING FRAME", currentPairFrame, "TO IT"
                                            
                                            self.semanticSequences[seqIdx][DICT_LABELLED_FRAMES][compatibilityClasses[currentFrameClass]].append(currentPairFrame)
                                            self.semanticSequences[seqIdx][DICT_NUM_EXTRA_FRAMES][compatibilityClasses[currentFrameClass]].append(numExtraFrames)
                                            taggedLabels.append(compatibilityClasses[currentFrameClass])
                                    else :
                                        print "USING EXISTING CLASS", classIdx, "OF", idx, "FOR FRAME", currentPairFrame
                                        taggedLabels.append(classIdx)
                                        
                                    ## keep track of sequences that have changed as the compatibility matrices with all the other sequences will have to change
                                    if seqIdx not in changedSequences :
                                        changedSequences.append(seqIdx)
                                        
                                    
                                    ## check if I'm adding same examples to 2 different classes for the same sequence
                                    if seq1Idx == seq2Idx and np.abs(currentPairFrames[0] - currentPairFrames[1]) <= numExtraFrames :
                                        print "BREAK BECAUSE SAME EXAMPLES"
                                        taggedLabels.append(taggedLabels[0])
                                        break
                        
                        if len(taggedLabels) == 2 :
                            keyToStore = DICT_CONFLICTING_SEQUENCES
                            if isCompatible :
                                keyToStore = DICT_COMPATIBLE_SEQUENCES
                                
                            ## sequence 1
                            if seq2Location not in self.semanticSequences[seq1Idx][keyToStore].keys() :
                                self.semanticSequences[seq1Idx][keyToStore][seq2Location] = []
                            if taggedLabels not in self.semanticSequences[seq1Idx][keyToStore][seq2Location] :
                                self.semanticSequences[seq1Idx][keyToStore][seq2Location].append(taggedLabels)
                            ## sequence 2
                            if seq1Location not in self.semanticSequences[seq2Idx][keyToStore].keys() :
                                self.semanticSequences[seq2Idx][keyToStore][seq1Location] = []
                            if taggedLabels[::-1] not in self.semanticSequences[seq2Idx][keyToStore][seq1Location] :
                                self.semanticSequences[seq2Idx][keyToStore][seq1Location].append(taggedLabels[::-1])
                                
                        ## update pair compatibility pairs
                        self.getSequencePairCompatibilityLabels([seq1Idx, seq2Idx], True)
                        
                        seqIdxs = [seq1Idx, seq2Idx]
            
            ## update all the compatibility matrices
            for changedSequence in changedSequences :
                for i in xrange(len(self.synthesisedSequence[DICT_USED_SEQUENCES])) :
                    self.getCompatibilityMat([changedSequence, i], True)
                    
#             self.getCompatibilityMat(seqIdxs, True)
            if len(seqIdxs) == 2 :
                self.makeInfoImage(seqIdxs)
            sys.stdout.flush()
    
    def tagSequenceInstancesFramesOld(self, selectedInstances, selectedFrames, isCompatible) :
        ### CAREFUL THIS ASSUMES THAT THERE IS ALWAYS JUST ONE PAIR OF INSTANCES AND IT'S THE PAIR THAT THE INFO DIALOG IS SHOWING ###
        if len(selectedInstances) > 1 :
            seqIdxs = []
            changedSequences = []
            ## for every pair of sequences in selectedInstances
            for idxs in np.argwhere(np.triu(np.ones(len(selectedInstances)), k=1)) : 
                currentPair = np.array(selectedInstances).flatten()[idxs]
                currentPairFrames = selectedFrames[idxs]
                taggedLabels = []
                seqIdxs = []
                
                ## compute compatibility labels for current pair
                seq1Idx = self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][currentPair[0]][DICT_SEQUENCE_IDX]
                seq2Idx = self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][currentPair[1]][DICT_SEQUENCE_IDX]
                
#                 seqCompatibilityLabels = [self.getFrameCompatibilityLabels(seq1Idx, self.semanticSequences[seq2Idx][DICT_SEQUENCE_LOCATION]),
#                                           self.getFrameCompatibilityLabels(seq2Idx, self.semanticSequences[seq1Idx][DICT_SEQUENCE_LOCATION])]
                print "IN TAG CONFLICTS",
                sequencePairCompatibilityLabels = self.getSequencePairCompatibilityLabels([seq1Idx, seq2Idx])

                for instanceIdx, shownFrame, [compatibilityLabels, compatibilityClasses] in zip(currentPair, currentPairFrames, sequencePairCompatibilityLabels) :
#                     shownFrame = self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][instanceIdx][DICT_SEQUENCE_FRAMES][self.frameIdx]
                    if shownFrame < 0 :
                        raise Exception("Tagging a negative frame idx")
    
                    seqIdxs.append(self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][instanceIdx][DICT_SEQUENCE_IDX])
                    
                    ## UGLY HACK that checks if I'm tagging frames of the same sequence and if they are close enough then don't add the second one
                    if len(seqIdxs) == 2 and seqIdxs[0] == seqIdxs[1] and len(taggedLabels) == 1 and np.abs(currentPairFrames[0] - currentPairFrames[1]) <= 4 :
                        taggedLabels.append(taggedLabels[0])                        
                        break
        
                    proceed = True
                    ## check if I already labelled the current frame and if the user says so delete it from the other classes
                    if DICT_LABELLED_FRAMES in self.semanticSequences[seqIdxs[-1]].keys() :
                        for idx, classIdx in enumerate(compatibilityClasses) :
                            classLabelledFrames = np.array(self.semanticSequences[seqIdxs[-1]][DICT_LABELLED_FRAMES][classIdx])
                            classNumExtraFrames = np.array(self.semanticSequences[seqIdxs[-1]][DICT_NUM_EXTRA_FRAMES][classIdx])
                            numExtraFrames = 4
                            targetFrames = np.arange(shownFrame-numExtraFrames/2, shownFrame+numExtraFrames/2+1).reshape((1, numExtraFrames+1))
                            found = np.any(np.abs(targetFrames - classLabelledFrames.reshape((len(classLabelledFrames), 1))) <= classNumExtraFrames.reshape((len(classNumExtraFrames), 1))/2, axis=1)
#                             found = np.abs(shownFrame-classLabelledFrames) <= classNumExtraFrames/2
                            if np.any(found) :
                                if classIdx < self.semanticSequences[self.selectedSemSequenceIdx][DICT_NUM_SEMANTICS] :
                                    QtGui.QMessageBox.critical(self, "Error Tagging Frame", ("<p align='center'>The current frame of {0}".format(self.semanticSequences[seqIdxs[-1]][DICT_SEQUENCE_NAME]) +
                                                                                             " is used for defining semantics.<br><b>Aborting...</b></p>"))
                                    proceed = False
                                    break
                                
                                if np.all(found) :
                                    text = "<p align='center'>The current frame of {1} has previously been labelled as <u>compatibility</u> class {0}. "
                                    text += "If it is overwritten, class {0} will have no remaining examples and <b>will</b> not influence compatibility any longer.<br>Do you want to proceed?</p>"
                                    text = text.format(idx, self.semanticSequences[seqIdxs[-1]][DICT_SEQUENCE_NAME])
                                else :
                                    text = "<p align='center'>The current frame of {1} has previously been labelled as <u>compatibility</u> class {0}.<br>"
                                    text += "Do you want to override?</p>"
                                    text = text.format(idx, self.semanticSequences[seqIdxs[-1]][DICT_SEQUENCE_NAME])
                                    
                                proceed = QtGui.QMessageBox.question(self, 'Frame already labelled', text, 
                                                                     QtGui.QMessageBox.Yes | QtGui.QMessageBox.No, QtGui.QMessageBox.No) == QtGui.QMessageBox.Yes
                                if proceed :
                                    self.semanticSequences[seqIdxs[-1]][DICT_LABELLED_FRAMES][classIdx] = [x for i, x in enumerate(classLabelledFrames) if not found[i]]
                                    self.semanticSequences[seqIdxs[-1]][DICT_NUM_EXTRA_FRAMES][classIdx] = [x for i, x in enumerate(classNumExtraFrames) if not found[i]]
                    
                    if proceed :
                        currentFrameClass = int(np.argmax(compatibilityLabels[shownFrame, :]))
                        ## ask if I should add new class or add as an example of currentFrameClass
                        msgBox = QtGui.QMessageBox(QtGui.QMessageBox.Question, 'Tagging Conflicts',
                                            ("<p align='center'>Conflicting frame {2} of {0} is of class {1}.<br>Would you like to <i>Split</i> into a new class or <i>Add</i> the "+
                                            "frame to {1}?</p>").format(self.semanticSequences[seqIdxs[-1]][DICT_SEQUENCE_NAME], currentFrameClass, shownFrame), parent=self)
                        cancelButton = msgBox.addButton("Cancel", QtGui.QMessageBox.DestructiveRole)
                        splitButton = msgBox.addButton("Split", QtGui.QMessageBox.YesRole)
                        addButton = msgBox.addButton("Add", QtGui.QMessageBox.NoRole)
                        msgBox.setDefaultButton(splitButton)
                        msgBox.setEscapeButton(cancelButton)

                        msgBox.exec_()

                        doAddNew = msgBox.clickedButton() == splitButton
                        
                        ## only go on if esc, cancel or X are not pressed
                        if msgBox.buttonRole(msgBox.clickedButton()) != QtGui.QMessageBox.DestructiveRole : 
                            ## deal with the new labelled frame
                            if DICT_LABELLED_FRAMES not in self.semanticSequences[seqIdxs[-1]].keys() :
                                self.semanticSequences[seqIdxs[-1]][DICT_LABELLED_FRAMES] = []

                            if doAddNew :
                                self.semanticSequences[seqIdxs[-1]][DICT_LABELLED_FRAMES].append([shownFrame])
                                taggedLabels.append(len(self.semanticSequences[seqIdxs[-1]][DICT_LABELLED_FRAMES])-1)
                            else :
                                self.semanticSequences[seqIdxs[-1]][DICT_LABELLED_FRAMES][compatibilityClasses[currentFrameClass]].append(shownFrame)
                                taggedLabels.append(compatibilityClasses[currentFrameClass])


                            ## deal with the extra frames
                            if DICT_NUM_EXTRA_FRAMES not in self.semanticSequences[seqIdxs[-1]].keys() :
                                self.semanticSequences[seqIdxs[-1]][DICT_NUM_EXTRA_FRAMES] = []

                            if doAddNew :
                                ## hardcoded 4 extra frames for now but should have some UI way of setting this
                                self.semanticSequences[seqIdxs[-1]][DICT_NUM_EXTRA_FRAMES].append([4])
                            else :
                                self.semanticSequences[seqIdxs[-1]][DICT_NUM_EXTRA_FRAMES][compatibilityClasses[currentFrameClass]].append(4)


                            ## deal with the new augmented labels: if the pair consists of 2 instances of the same sequence, this will be overwritten by the second instance which should be the correct 1
        #                     self.semanticSequences[seqIdxs[-1]][DICT_FRAME_COMPATIBILITY_LABELS] = propagateLabels(self.preloadedDistanceMatrices[seqIdxs[-1]],
        #                                                                                                            self.semanticSequences[seqIdxs[-1]][DICT_LABELLED_FRAMES],
        #                                                                                                            self.semanticSequences[seqIdxs[-1]][DICT_NUM_EXTRA_FRAMES], True, 0.002)
        #                                                                                                            self.compatibilitySigmaSpinBox.value()/self.compatibilitySigmaDividerSpinBox.value())

                            if isCompatible :
                                ## deal with compatible sequences
                                if DICT_COMPATIBLE_SEQUENCES not in self.semanticSequences[seqIdxs[-1]].keys() :
                                    self.semanticSequences[seqIdxs[-1]][DICT_COMPATIBLE_SEQUENCES] = {}
                            else :
                                ## deal with the conflicting sequences
                                if DICT_CONFLICTING_SEQUENCES not in self.semanticSequences[seqIdxs[-1]].keys() :
                                    self.semanticSequences[seqIdxs[-1]][DICT_CONFLICTING_SEQUENCES] = {}

                            ## keep track of sequences that have changed as the compatibility matrices with all the other sequences will have to change
                            if seqIdxs[-1] not in changedSequences :
                                changedSequences.append(seqIdxs[-1])

                ## deal with the conflicting sequences
                seq1Location = self.semanticSequences[seqIdxs[0]][DICT_SEQUENCE_LOCATION]
                seq2Location = self.semanticSequences[seqIdxs[1]][DICT_SEQUENCE_LOCATION]
                
                keyToStore = DICT_CONFLICTING_SEQUENCES
                if isCompatible :
                    keyToStore = DICT_COMPATIBLE_SEQUENCES
                
                if len(seqIdxs) == 2 and len(taggedLabels) == 2 :
                    ## sequence 1
                    if seq2Location not in self.semanticSequences[seqIdxs[0]][keyToStore].keys() :
                        self.semanticSequences[seqIdxs[0]][keyToStore][seq2Location] = []
                    if taggedLabels not in self.semanticSequences[seqIdxs[0]][keyToStore][seq2Location] :
                        self.semanticSequences[seqIdxs[0]][keyToStore][seq2Location].append(taggedLabels)
                    ## sequence 2
                    if seq1Location not in self.semanticSequences[seqIdxs[1]][keyToStore].keys() :
                        self.semanticSequences[seqIdxs[1]][keyToStore][seq1Location] = []
                    if taggedLabels[::-1] not in self.semanticSequences[seqIdxs[1]][keyToStore][seq1Location] :
                        self.semanticSequences[seqIdxs[1]][keyToStore][seq1Location].append(taggedLabels[::-1])

                    ## update pair compatibility pairs
                    self.getSequencePairCompatibilityLabels(seqIdxs, True)            
            
            ## update all the compatibility matrices
            for changedSequence in changedSequences :
                for i in xrange(len(self.synthesisedSequence[DICT_USED_SEQUENCES])) :
                    self.getCompatibilityMat([changedSequence, i], True)
                    
#             self.getCompatibilityMat(seqIdxs, True)
            self.makeInfoImage(seqIdxs)
            

    def keyPressEvent(self, e) :
        if np.all(self.loadedSynthesisedSequence != None) :
            if e.key() == QtCore.Qt.Key_U or e.key() == QtCore.Qt.Key_D :
                print "moving instance",
                if len(self.selectedSequenceInstancesIdxes) == 1 :
                    print self.selectedSequenceInstancesIdxes[0],
                    ## down means up the list as I'm visualizing from the first to the last and rendering the last ones first (for consistency as the list is basically a list of layers and first one should be on top of rest)
                    if e.key() == QtCore.Qt.Key_D and self.selectedSequenceInstancesIdxes[0] < len(self.synthesisedSequence[DICT_SEQUENCE_INSTANCES]) - 1 :
                        self.synthesisedSequence[DICT_SEQUENCE_INSTANCES].insert(self.selectedSequenceInstancesIdxes[0]+1, 
                                                                                 self.synthesisedSequence[DICT_SEQUENCE_INSTANCES].pop(self.selectedSequenceInstancesIdxes[0]))
                        print "down"
                    elif e.key() == QtCore.Qt.Key_U and self.selectedSequenceInstancesIdxes[0] > 0 :
                        self.synthesisedSequence[DICT_SEQUENCE_INSTANCES].insert(self.selectedSequenceInstancesIdxes[0]-1, 
                                                                                 self.synthesisedSequence[DICT_SEQUENCE_INSTANCES].pop(self.selectedSequenceInstancesIdxes[0]))
                        print "up"

                    self.setSemanticSliderDrawables()
                    self.setListOfSequenceInstances()
                    self.setInstanceShowIndicatorDrawables()
                    self.instancesShowIndicator.setSelectedInstances(self.selectedSequenceInstancesIdxes)
                    
                print
            elif e.key() == e.key() >= QtCore.Qt.Key_0 and e.key() <= QtCore.Qt.Key_9 :
                pressedNum = np.mod(e.key()-int(QtCore.Qt.Key_0), int(QtCore.Qt.Key_9))
                doAllowRandomChoice = False
                chosenRandomly = False
                if doAllowRandomChoice and len(self.selectedSequenceInstancesIdxes) == 0 :
                    ## randomize which instance I pick
                    viableInstances = []
                    for i in xrange(len(self.synthesisedSequence[DICT_SEQUENCE_INSTANCES])) :
                        print i, pressedNum, self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][i][DICT_DESIRED_SEMANTICS][self.frameIdx, :]
                        if self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][i][DICT_DESIRED_SEMANTICS][self.frameIdx, pressedNum] == 0.0 :
                            viableInstances.append(i)
                    if len(viableInstances) > 0 :
                        self.selectedSequenceInstancesIdxes = [np.random.choice(viableInstances)]
                    chosenRandomly = True
                
                ## each pressed number is a key in the dict below which contains the index of the instance the key press refers and the action index
                ## e.g. if I press the number 5, I want to trigger action 1 of instance 0 (which for drumming_new, it should be the floor tom of right_hand1)
                self.mergedActionRequestsBindings = {1:[1, 1], 2:[1, 2], 3:[1, 3], 4:[1, 4], 5:[0, 1], 6:[0, 2], 7:[0, 3], 8:[2, 1]}
                ## key mappings for toy1
                if self.loadedSynthesisedSequence == "/home/ilisescu/PhD/data/synthesisedSequences/lullaby_demo/synthesised_sequence.npy" :
                    self.mergedActionRequestsBindings = {1:[0, 1], 2:[0, 2], 3:[0, 3], 4:[0, 4], 5:[0,  5], 6:[0, 6], 7:[0, 7], 8:[0, 8]}
                if self.mergeActionRequests and pressedNum in self.mergedActionRequestsBindings.keys() :
                    self.selectedSequenceInstancesIdxes = [self.mergedActionRequestsBindings[pressedNum][0]]
                    pressedNum = self.mergedActionRequestsBindings[pressedNum][1]
                        
                if len(self.selectedSequenceInstancesIdxes) == 1 and self.selectedSequenceInstancesIdxes[-1] >= 0 and self.selectedSequenceInstancesIdxes[-1] < len(self.synthesisedSequence[DICT_SEQUENCE_INSTANCES]) :

                    if self.frameIdx > len(self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][self.selectedSequenceInstancesIdxes[-1]][DICT_SEQUENCE_FRAMES]) :
                        self.frameIdxSpinBox.setValue(len(self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][self.selectedSequenceInstancesIdxes[-1]][DICT_SEQUENCE_FRAMES])-1)

                    currentSemanticsIdx = np.argwhere(self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][self.selectedSequenceInstancesIdxes[-1]][DICT_DESIRED_SEMANTICS][self.frameIdx, :] == 1.0)
                    numSemantics = self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][self.selectedSequenceInstancesIdxes[-1]][DICT_DESIRED_SEMANTICS].shape[1]
                    if pressedNum < numSemantics :
                        doSwitchBackToRest = (e.modifiers() & QtCore.Qt.Modifier.ALT or self.mergeActionRequests) and pressedNum != 0

                        ## take current semantics
                        desiredSemantics = self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][self.selectedSequenceInstancesIdxes[-1]][DICT_DESIRED_SEMANTICS][self.frameIdx, :].reshape((1, numSemantics))

    #                     ## toggle to new ones
    #                     toggledlabels = toggleLabelsSmoothly(np.array([[1.0, 0.0]]), self.TOGGLE_DELAY)
    #                     desiredSemantics = np.concatenate((desiredSemantics, np.zeros((self.TOGGLE_DELAY, numSemantics))))

    #                     ## len(currentSemantic) could be 0 if the current frame has no desired semantics, i.e. all the probs in the semantics vector are 0.0
    #                     if len(currentSemanticsIdx) == 1 :
    #                         desiredSemantics[1:, int(currentSemanticsIdx[0])] = toggledlabels[:, 0]

    #                     desiredSemantics[1:, pressedNum] = toggledlabels[:, 1]

                        if self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][self.selectedSequenceInstancesIdxes[-1]][DICT_DESIRED_SEMANTICS][self.frameIdx, pressedNum] != 1.0 :
                            toggledlabels = toggleAllLabelsSmoothly(desiredSemantics[-1, :], pressedNum, self.TOGGLE_DELAY) #toggleLabelsSmoothly(np.array([[1.0, 0.0]]), self.TOGGLE_DELAY)
                            desiredSemantics = np.concatenate((desiredSemantics, toggledlabels)) #np.zeros((self.TOGGLE_DELAY, numSemantics))))

                            ## if impulse, toggle back to default semantics (i.e. the 0th one for now, not sure how to set this for arbitrary sequences)
                            if False and  e.modifiers() & QtCore.Qt.Modifier.ALT and pressedNum != 0 :
                                ## pad the tip with new semantics
                                tmp = np.zeros((1, numSemantics))
                                tmp[0, pressedNum] = 1.0
                                desiredSemantics = np.concatenate((desiredSemantics, tmp.repeat(self.TOGGLE_DELAY*2, axis=0)))
                                ## toggle back to default
                                toggledlabels = toggleAllLabelsSmoothly(desiredSemantics[-1, :], 0, self.TOGGLE_DELAY)
                                desiredSemantics = np.concatenate((desiredSemantics, toggledlabels))

                                ## pad remaining with default semantics
                                tmp = np.zeros((1, numSemantics))
                                tmp[0, 0] = 1.0
                                desiredSemantics = np.concatenate((desiredSemantics, tmp.repeat(self.EXTEND_LENGTH-3*self.TOGGLE_DELAY-1, axis=0)))


                            ## otherwise just straight up toggle to pressedNum semantics and stay there
                            else :
                                ## pad with new semantics
                                tmp = np.zeros((1, numSemantics))
                                tmp[0, pressedNum] = 1.0
                                desiredSemantics = np.concatenate((desiredSemantics, tmp.repeat(self.EXTEND_LENGTH-self.TOGGLE_DELAY-1, axis=0)))
                        else :
                            desiredSemantics = desiredSemantics.repeat(self.EXTEND_LENGTH-1, axis=0)

                        ## remove synthesised frames after frameIdx
                        self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][self.selectedSequenceInstancesIdxes[-1]][DICT_SEQUENCE_FRAMES] = (self.synthesisedSequence[DICT_SEQUENCE_INSTANCES]
                                                                                                                                            [self.selectedSequenceInstancesIdxes[-1]][DICT_SEQUENCE_FRAMES][:self.frameIdx+1])
                        ## override existing desired semantics from frameIdx on
                        self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][self.selectedSequenceInstancesIdxes[-1]][DICT_DESIRED_SEMANTICS] = np.vstack(((self.synthesisedSequence[DICT_SEQUENCE_INSTANCES]
                                                                                                                                                         [self.selectedSequenceInstancesIdxes[-1]][DICT_DESIRED_SEMANTICS]
                                                                                                                                                         [:self.frameIdx]),
                                                                                                                                                 desiredSemantics))
                        
    #                     self.extendFullSequence()
                        if e.modifiers() & QtCore.Qt.Modifier.CTRL and len(self.selectedSequenceInstancesIdxes) > 0 :
                            self.extendFullSequenceNew(np.array([-1]), self.selectedSequenceInstancesIdxes, verbose=False)
                        else :
                            self.extendFullSequenceNew(np.array([-1]), verbose=False)
                        
                        ## make sure I remove the sounds to be played if I generate more stuff on top of them
                        if self.doPlaySounds :
                            ## remove keys that are bigger than self.frameIdx
                            for key in self.playSoundsTimes.keys() :
                                if key > self.frameIdx and self.selectedSequenceInstancesIdxes[-1] in self.playSoundsTimes[key].keys() :
                                    del self.playSoundsTimes[key][self.selectedSequenceInstancesIdxes[-1]]
                                if len(self.playSoundsTimes[key].keys()) == 0 :
                                    del self.playSoundsTimes[key]
                                    
                        ## sitch back to the rest position if wanted
                        if doSwitchBackToRest :
                            seqIdx = self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][self.selectedSequenceInstancesIdxes[-1]][DICT_SEQUENCE_IDX]
                            newlySynthesisedFrames = self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][self.selectedSequenceInstancesIdxes[-1]][DICT_SEQUENCE_FRAMES][self.frameIdx:]
#                             print "README", newlySynthesisedFrames, pressedNum, seqIdx
#                             print self.semanticSequences[seqIdx][DICT_FRAME_SEMANTICS][newlySynthesisedFrames, pressedNum]
                            desiredSemanticProbabilities = np.argwhere(self.semanticSequences[seqIdx][DICT_FRAME_SEMANTICS][newlySynthesisedFrames, pressedNum] > 0.99).flatten()
                            if len(desiredSemanticProbabilities) > 0 and desiredSemanticProbabilities[0] < len(newlySynthesisedFrames)-self.TOGGLE_DELAY*2-1:
                                currentFrameIdx = np.copy(self.frameIdx)
                                self.frameIdx = int(self.frameIdx + desiredSemanticProbabilities[0]+1)
                                self.EXTEND_LENGTH = len(self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][self.selectedSequenceInstancesIdxes[-1]][DICT_SEQUENCE_FRAMES])-self.frameIdx
                                
                                ## get desired semantics by switching to the rest action and padding with the rest action till the end
                                desiredSemantics = self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][self.selectedSequenceInstancesIdxes[-1]][DICT_DESIRED_SEMANTICS][self.frameIdx, :].reshape((1, numSemantics))
                                toggledlabels = toggleAllLabelsSmoothly(desiredSemantics[-1, :], 0, self.TOGGLE_DELAY)
                                desiredSemantics = np.concatenate((desiredSemantics, toggledlabels))

                                ## pad remaining with default semantics
                                tmp = np.zeros((1, numSemantics))
                                tmp[0, 0] = 1.0
                                desiredSemantics = np.concatenate((desiredSemantics, tmp.repeat(self.EXTEND_LENGTH-self.TOGGLE_DELAY-1, axis=0)))
                                
                                
#                                 print self.frameIdx, self.EXTEND_LENGTH, desiredSemantics.shape

                                ## remove synthesised frames after frameIdx
                                self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][self.selectedSequenceInstancesIdxes[-1]][DICT_SEQUENCE_FRAMES] = (self.synthesisedSequence[DICT_SEQUENCE_INSTANCES]
                                                                                                                                                    [self.selectedSequenceInstancesIdxes[-1]]
                                                                                                                                                    [DICT_SEQUENCE_FRAMES][:self.frameIdx+1])
                                ## override existing desired semantics from frameIdx on
                                self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][self.selectedSequenceInstancesIdxes[-1]][DICT_DESIRED_SEMANTICS] = np.vstack(((self.synthesisedSequence[DICT_SEQUENCE_INSTANCES]
                                                                                                                                                                 [self.selectedSequenceInstancesIdxes[-1]]
                                                                                                                                                                 [DICT_DESIRED_SEMANTICS][:self.frameIdx]), desiredSemantics))                                        
                                
                                self.extendFullSequenceNew(np.array([-1]), self.selectedSequenceInstancesIdxes, verbose=False)
                        
                                ## deal with sound playback bookeeping
                                if self.doPlaySounds:                                    
                                    soundIdxStart = 0
                                    if self.semanticSequences[seqIdx][DICT_SEQUENCE_NAME] == "left_hand1" :
                                        soundIdxStart = 0
                                    elif self.semanticSequences[seqIdx][DICT_SEQUENCE_NAME] == "right_hand1" :
                                        soundIdxStart = 4
                                    elif self.semanticSequences[seqIdx][DICT_SEQUENCE_NAME] == "foot1" :
                                        soundIdxStart = 7
                                    soundToPlayIdx = pressedNum-1+soundIdxStart
                                    
                                    ## remove keys that are bigger than self.frameIdx
                                    for key in self.playSoundsTimes.keys() :
                                        if key > self.frameIdx and self.selectedSequenceInstancesIdxes[-1] in self.playSoundsTimes[key].keys() :
                                            del self.playSoundsTimes[key][self.selectedSequenceInstancesIdxes[-1]]
                                        if len(self.playSoundsTimes[key].keys()) == 0 :
                                            del self.playSoundsTimes[key]
                                    
#                                     print "ADDING SOUND AT", self.frameIdx, self.playSoundsTimes, "#############", 
                                    if self.frameIdx not in self.playSoundsTimes.keys() :
#                                         print "LALALA", self.playSoundsTimes, type(self.frameIdx), "LALALA", 
                                        self.playSoundsTimes[self.frameIdx] = {}
#                                         print "LALALA2", self.playSoundsTimes, self.frameIdx, self.selectedSequenceInstancesIdxes[-1], "LALALA2", 
                                    self.playSoundsTimes[self.frameIdx][self.selectedSequenceInstancesIdxes[-1]] = soundToPlayIdx
#                                     print self.selectedSequenceInstancesIdxes[-1], self.playSoundsTimes
                                    
                                ## restore vars to previous values
                                self.frameIdx = np.copy(currentFrameIdx)
                                self.EXTEND_LENGTH = self.extendLengthSpinBox.value()
                            else :
                                QtGui.QMessageBox.warning(self, "Cannot switch to rest action", ("<p align='center'>Cannot switch back to rest action as the desired action has not been reached."+
                                                                                                 "\nPlease increase the value of a (i.e. the importance of showing the desired action)</p>"))
                            
                if chosenRandomly :
                    self.selectedSequenceInstancesIdxes = []

            elif e.key() == QtCore.Qt.Key_R :
                ### REFINE

                minFrames = self.frameIdxSlider.maximum()+1
                for i in xrange(len(self.synthesisedSequence[DICT_SEQUENCE_INSTANCES])) :
                    minFrames = np.min((minFrames, len(self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][i][DICT_SEQUENCE_FRAMES])))

                startFrame = np.max((0, self.frameIdx-50))
                self.frameIdx = startFrame
                tmp = np.copy(self.EXTEND_LENGTH)
                self.EXTEND_LENGTH = np.min((minFrames, self.EXTEND_LENGTH))

                ## setting 1 instance at a time as "conflicting", which means it will be the only one free to change conditionally on the remaining locked ones
                for i in xrange(len(self.synthesisedSequence[DICT_SEQUENCE_INSTANCES])) :
                    self.extendFullSequenceNew([i])

                self.EXTEND_LENGTH = np.copy(tmp)
                self.frameIdxSpinBox.setValue(startFrame)


            elif e.key() == QtCore.Qt.Key_Space :
    #             if e.modifiers() & QtCore.Qt.Modifier.SHIFT :
    #                 ## first delete for each isntance the generated frames from self.frameIdx and then extend with existing desired semantics                
    #                 for i in xrange(len(self.synthesisedSequence[DICT_SEQUENCE_INSTANCES])) :
    #                     self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][i][DICT_SEQUENCE_FRAMES] = self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][i][DICT_SEQUENCE_FRAMES][:self.frameIdx+1]
    #             self.extendFullSequence()
    #             self.extendFullSequenceNew()

                ## find the minimum amount of frames I have
                minFrames = self.frameIdxSlider.maximum()+1
                for i in xrange(len(self.synthesisedSequence[DICT_SEQUENCE_INSTANCES])) :
                    minFrames = np.min((minFrames, len(self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][i][DICT_SEQUENCE_FRAMES])))
    #             ## got to min frame
    #             self.frameIdxSpinBox.setValue(minFrames-1)

                ## if control is pressed I have to update compatibility between selected instances and synthesise from earlier in the timeline resolving the compatibilities
                if e.modifiers() & QtCore.Qt.Modifier.CTRL and len(self.selectedSequenceInstancesIdxes) > 0 : #1 :
    #                 self.frameIdx = np.max((0, self.frameIdx-50))
    #                 tmp = np.copy(self.EXTEND_LENGTH)
    #                 self.EXTEND_LENGTH = np.min((minFrames, self.EXTEND_LENGTH))

                    ## HACK : when pressing control C now, only the selected instances will be updated ###
    #                 self.extendFullSequenceNew(self.selectedSequenceInstancesIdxes)
                    self.extendFullSequenceNew(np.array([-1]), self.selectedSequenceInstancesIdxes)

    #                 self.EXTEND_LENGTH = np.copy(tmp)
    #                 self.frameIdxSpinBox.setValue(minFrames-1)
                else :
                    if e.modifiers() & QtCore.Qt.Modifier.SHIFT and minFrames > self.frameIdx :
                        ## first delete for each instance the generated frames from self.frameIdx and then extend with existing desired semantics
                        for i in xrange(len(self.synthesisedSequence[DICT_SEQUENCE_INSTANCES])) :
                            self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][i][DICT_SEQUENCE_FRAMES] = self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][i][DICT_SEQUENCE_FRAMES][:self.frameIdx+1]
                        minFrames = self.frameIdx+1

                    ## go to min frame
                    self.frameIdxSpinBox.setValue(minFrames-1)

                    self.extendFullSequenceNew(np.array([-1]))

            elif e.key() == QtCore.Qt.Key_T :
                if len(self.selectedSequenceInstancesIdxes) == 2 :
                    self.taggedConflicts.append({
                                                    'seq_idxes':np.empty(0, int),
                                                    'tagged_frames':np.empty(0, int)
                                                })
                    selectedFrames = np.zeros(len(self.selectedSequenceInstancesIdxes), int)
                    for idx, instanceIdx in enumerate(self.selectedSequenceInstancesIdxes) :
                        self.taggedConflicts[-1]['seq_idxes'] = np.concatenate((self.taggedConflicts[-1]['seq_idxes'],
                                                                                [self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][instanceIdx][DICT_SEQUENCE_IDX]]))
                        self.taggedConflicts[-1]['tagged_frames'] = np.concatenate((self.taggedConflicts[-1]['tagged_frames'],
                                                                                    [self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][instanceIdx][DICT_SEQUENCE_FRAMES][self.frameIdx]]))

                        selectedFrames[idx] = self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][instanceIdx][DICT_SEQUENCE_FRAMES][self.frameIdx]

                    ## order by sequence idx
                    newOrder = np.argsort(self.taggedConflicts[-1]['seq_idxes']).flatten()
                    self.taggedConflicts[-1]['seq_idxes'] = self.taggedConflicts[-1]['seq_idxes'][newOrder]
                    self.taggedConflicts[-1]['tagged_frames'] = self.taggedConflicts[-1]['tagged_frames'][newOrder]

                    self.tagSequenceInstancesFrames(self.selectedSequenceInstancesIdxes, selectedFrames, e.modifiers() & QtCore.Qt.Modifier.SHIFT)

                elif len(self.selectedSequenceInstancesIdxes) == 1 and self.selectedSequenceInstancesIdxes[-1] >= 0 and self.selectedSequenceInstancesIdxes[-1] < len(self.synthesisedSequence[DICT_SEQUENCE_INSTANCES]) :
                    seqIdx = self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][self.selectedSequenceInstancesIdxes[-1]][DICT_SEQUENCE_IDX]
                    if seqIdx not in self.taggedFrames.keys() :
                        self.taggedFrames[seqIdx] = {
                                                     DICT_SEQUENCE_NAME:self.semanticSequences[seqIdx][DICT_SEQUENCE_NAME],
                                                     DICT_SEQUENCE_FRAMES:np.empty((0, 2), int)
                                                     }

                    if self.frameIdx >= 0 and self.frameIdx < len(self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][self.selectedSequenceInstancesIdxes[-1]][DICT_SEQUENCE_FRAMES])-1 :
                        firstFrame = self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][self.selectedSequenceInstancesIdxes[-1]][DICT_SEQUENCE_FRAMES][self.frameIdx]
                        secondFrame = self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][self.selectedSequenceInstancesIdxes[-1]][DICT_SEQUENCE_FRAMES][self.frameIdx+1]


                        if not np.any(np.all([self.taggedFrames[seqIdx][DICT_SEQUENCE_FRAMES][:, 0] == firstFrame, self.taggedFrames[seqIdx][DICT_SEQUENCE_FRAMES][:, 1] == secondFrame], axis=0)) :
                            print "tagging", self.semanticSequences[seqIdx][DICT_SEQUENCE_NAME], firstFrame, secondFrame
                            self.taggedFrames[seqIdx][DICT_SEQUENCE_FRAMES] = np.array(np.concatenate((self.taggedFrames[seqIdx][DICT_SEQUENCE_FRAMES],
                                                                                                       np.array([[firstFrame, secondFrame]], int)), axis=0), int)

                            np.save("/".join(self.loadedSynthesisedSequence.split("/")[:-1])+"/tagged_frames.npy", self.taggedFrames)
                        else :
                            print "NOT tagging because existing", self.semanticSequences[seqIdx][DICT_SEQUENCE_NAME], firstFrame, secondFrame


            elif e.key() == QtCore.Qt.Key_Delete :
                if len(self.selectedSequenceInstancesIdxes) == 1 and self.selectedSequenceInstancesIdxes[-1] >= 0 and self.selectedSequenceInstancesIdxes[-1] < len(self.synthesisedSequence[DICT_SEQUENCE_INSTANCES]) :
                    seqIdx = self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][self.selectedSequenceInstancesIdxes[-1]][DICT_SEQUENCE_IDX]
                    proceed = QtGui.QMessageBox.question(self, 'Delete semantic sequence instance',
                                        "Do you want to delete instance "+np.string_(self.selectedSequenceInstancesIdxes[-1]+1)+
                                        " of type "+self.semanticSequences[seqIdx][DICT_SEQUENCE_NAME]+"?\nOperation is not reversible.", 
                                        QtGui.QMessageBox.Yes | QtGui.QMessageBox.No, QtGui.QMessageBox.No) == QtGui.QMessageBox.Yes
                    if proceed :
                        del self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][self.selectedSequenceInstancesIdxes[-1]]
                        del self.instancesDoShow[self.selectedSequenceInstancesIdxes[-1]]

                        ## update sliders
                        maxFrames = 0
                        for i in xrange(len(self.synthesisedSequence[DICT_SEQUENCE_INSTANCES])) :
                            maxFrames = np.max((maxFrames, len(self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][i][DICT_SEQUENCE_FRAMES])))
                        self.frameIdxSlider.setMaximum(maxFrames-1)
                        self.frameIdxSpinBox.setRange(0, maxFrames-1)

                        ## UI stuff
                        self.setListOfSequenceInstances()
                        self.setSemanticSliderDrawables()
                        self.setInstanceShowIndicatorDrawables()
                        
                        if self.frameIdx > maxFrames :
                            self.frameIdxSpinBox.setValue(maxFrames)
                        ## render
                        self.showFrame(self.frameIdx)
                        self.setFocus()

            elif e.key() == QtCore.Qt.Key_S and e.modifiers() & QtCore.Qt.Modifier.CTRL :
                self.saveSynthesisedSequence()
                print "saved"
            elif e.key() == QtCore.Qt.Key_A :
                if self.selectedSemSequenceIdx >= 0 and self.selectedSemSequenceIdx < len(self.semanticSequences) :
                    ### UI stuff ###
#                     self.addNewSemanticSequenceControls.setVisible(True)
                    self.frameIdxSlider.setEnabled(False)
                    self.frameIdxSpinBox.setEnabled(False)
                    self.sequenceXOffsetSpinBox.setValue(0)
                    self.sequenceYOffsetSpinBox.setValue(0)
                    self.sequenceXScaleSpinBox.setValue(1.0)
                    self.sequenceYScaleSpinBox.setValue(1.0)
                    self.startingSemanticsComboBox.clear()
#                     self.startingSemanticsComboBox.addItems(np.arange(self.semanticSequences[self.selectedSemSequenceIdx][DICT_NUM_SEMANTICS]).astype(np.string0))
                    self.startingSemanticsComboBox.addItems([self.semanticSequences[self.selectedSemSequenceIdx][DICT_SEMANTICS_NAMES][bob]
                                                             for bob in self.semanticSequences[self.selectedSemSequenceIdx][DICT_SEMANTICS_NAMES].keys()])
                    self.startingSemanticsComboBox.setCurrentIndex(0)

                    ## this code is replicated in currentStartingSemanticsChanged but whatevs
                    frameIdx = int(np.argwhere(self.semanticSequences[self.selectedSemSequenceIdx][DICT_FRAME_SEMANTICS][:, 0] >= 0.9)[0])
                    frameKey = np.sort(self.semanticSequences[self.selectedSemSequenceIdx][DICT_FRAMES_LOCATIONS].keys())[frameIdx]
                    self.initNewSemanticSequenceInstance(frameKey)
                    
                    self.addNewSemanticSequenceControls.setWindowTitle("Add Copy of \"{0}\"".format(self.semanticSequences[self.selectedSemSequenceIdx][DICT_SEQUENCE_NAME]))
                    self.addNewSemanticSequenceControls.exec_()
            elif e.key() == QtCore.Qt.Key_Right :
                self.frameIdxSpinBox.setValue(self.frameIdx+1)
            elif e.key() == QtCore.Qt.Key_Left :
                self.frameIdxSpinBox.setValue(self.frameIdx-1)
            elif e.key() == QtCore.Qt.Key_K :
                if len(self.semanticSequences) > 0 :
                    changeActionsCommandBindings(self, "Change Action Command Bindings", self.semanticSequences)

            sys.stdout.flush()
    
    def wheelEvent(self, e) :
        if e.delta() < 0 :
            self.frameIdxSpinBox.setValue(self.frameIdx-1)
        else :
            self.frameIdxSpinBox.setValue(self.frameIdx+1)
        time.sleep(0.01)
        
    def eventFilter(self, obj, event) :
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
        elif obj == self.infoDialog.imageLabel and event.type() == QtCore.QEvent.Type.MouseMove :
            self.updateInfoDialog(np.array([event.pos().x(), event.pos().y()]))
            return True
        elif obj == self.infoDialog and event.type() == QtCore.QEvent.Type.KeyPress :
            if event.key() == QtCore.Qt.Key_T :
                if len(self.selectedSequenceInstancesIdxes) == 2 :
                    seqIdxs = np.array([self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][self.selectedSequenceInstancesIdxes[0]][DICT_SEQUENCE_IDX],
                                        self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][self.selectedSequenceInstancesIdxes[1]][DICT_SEQUENCE_IDX]])

                    if (np.all(self.infoFrameIdxs >= 0) and self.infoFrameIdxs[0] < len(self.semanticSequences[seqIdxs[0]][DICT_FRAMES_LOCATIONS]) and
                        self.infoFrameIdxs[1] < len(self.semanticSequences[seqIdxs[1]][DICT_FRAMES_LOCATIONS])) :

                        self.tagSequenceInstancesFrames(self.selectedSequenceInstancesIdxes, self.infoFrameIdxs, event.modifiers() & QtCore.Qt.Modifier.SHIFT)

                return True
        elif obj == self.sequenceInstancesListTable and event.type() == QtCore.QEvent.Type.KeyPress :
            self.keyPressEvent(event)
            return True
            
        return QtGui.QWidget.eventFilter(self, obj, event)
    
    def mousePressed(self, event):
#         print event.pos()
#         sys.stdout.flush()
        if event.button() == QtCore.Qt.LeftButton :
            self.isScribbling = True
            print "left button clicked"
        elif event.button() == QtCore.Qt.RightButton :
            print "right button clicked"
        
        sys.stdout.flush()
                
    def mouseMoved(self, event):
        if self.isScribbling :
            print "scribbling", event.pos()
            
    def mouseReleased(self, event):
        if self.isScribbling :
            self.isScribbling = False
            
    def updateInfoDialog(self, posXY):
        if self.doShowCompatibilityInfo :
            if len(self.selectedSequenceInstancesIdxes) == 2 :
                seqIdxs = np.array([self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][self.selectedSequenceInstancesIdxes[0]][DICT_SEQUENCE_IDX],
                                    self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][self.selectedSequenceInstancesIdxes[1]][DICT_SEQUENCE_IDX]])

                if posXY[0] >= (self.PLOT_HEIGHT + self.TEXT_SIZE) and posXY[1] >= (self.PLOT_HEIGHT + self.TEXT_SIZE) :
                    self.infoFrameIdxs = (posXY - (self.PLOT_HEIGHT + self.TEXT_SIZE))[::-1]
                    self.infoFrameIdxs = np.round(self.infoFrameIdxs/self.plotScaleRatio).astype(int)
                elif posXY[0] >= self.TEXT_SIZE and posXY[1] >= (self.PLOT_HEIGHT + self.TEXT_SIZE) :
                    self.infoFrameIdxs[0] = posXY[1]-(self.PLOT_HEIGHT + self.TEXT_SIZE)
                    self.infoFrameIdxs[0] = int(np.round(self.infoFrameIdxs[0]/self.plotScaleRatio))
                elif posXY[1] >= self.TEXT_SIZE and posXY[0] >= (self.PLOT_HEIGHT + self.TEXT_SIZE) :
                    self.infoFrameIdxs[1] = posXY[0]-(self.PLOT_HEIGHT + self.TEXT_SIZE)
                    self.infoFrameIdxs[1] = int(np.round(self.infoFrameIdxs[1]/self.plotScaleRatio))
#                 else :
#                     self.infoFrameIdxs = np.array([-1, -1])

                if (np.all(self.infoFrameIdxs >= 0) and self.infoFrameIdxs[0] < len(self.semanticSequences[seqIdxs[0]][DICT_FRAMES_LOCATIONS]) and
                    self.infoFrameIdxs[1] < len(self.semanticSequences[seqIdxs[1]][DICT_FRAMES_LOCATIONS])) :

                    if (seqIdxs[0] in self.preloadedDistanceMatrices.keys() and seqIdxs[1] in self.preloadedDistanceMatrices.keys() and
                        DICT_LABELLED_FRAMES in self.semanticSequences[seqIdxs[0]].keys() and
                        DICT_LABELLED_FRAMES in self.semanticSequences[seqIdxs[1]].keys()) :
                        
                        sequencePairCompatibilityLabels = self.getSequencePairCompatibilityLabels(seqIdxs, verbose=False)
                        self.infoDialog.setText(("compatibility cost [{0:d}, {1:d}] = {2:05f}\n"+
                                                 "{3}'s augmented labels [{4:d}] = {5}\n"+
                                                 "{6}'s augmented labels [{7:d}] = {8}").format(self.infoFrameIdxs[0], self.infoFrameIdxs[1],
                                                                                                self.getCompatibilityMat(seqIdxs, verbose=False)[self.infoFrameIdxs[0], self.infoFrameIdxs[1]],
                                                                                                self.semanticSequences[seqIdxs[0]][DICT_SEQUENCE_NAME], self.infoFrameIdxs[0],
                                                                                                np.array_str(sequencePairCompatibilityLabels[0][0][self.infoFrameIdxs[0], :],
                                                                                                             precision=1),
                                                                                                self.semanticSequences[seqIdxs[1]][DICT_SEQUENCE_NAME], self.infoFrameIdxs[1],
                                                                                                np.array_str(sequencePairCompatibilityLabels[1][0][self.infoFrameIdxs[1], :],
                                                                                                             precision=1)))
                        
                        self.infoDialog.setCursorPosition((self.infoFrameIdxs*self.plotScaleRatio+self.TEXT_SIZE+self.PLOT_HEIGHT)[::-1])

                    ## update visLabel
                    if np.all(self.bgImage != None) :
                        bgImage = np.ascontiguousarray(self.bgImage.copy())
                        visImg = QtGui.QImage(bgImage.data, bgImage.shape[1], bgImage.shape[0], bgImage.strides[0], QtGui.QImage.Format_RGB888);
                    else :
                        visImg = QtGui.QImage(QtCore.QSize(1280, 720), QtGui.QImage.Format_RGB888)
                        visImg.fill(QtGui.QColor.fromRgb(255, 255, 255))

                    col = mpl.cm.jet(self.getCompatibilityMat(seqIdxs, verbose=False)[self.infoFrameIdxs[0], self.infoFrameIdxs[1]]/np.max(self.getCompatibilityMat(seqIdxs, verbose=False)), bytes=True)

                    for i in xrange(len(seqIdxs)) :
                        seqIdx = seqIdxs[i]
                        sequenceFrameIdx = self.infoFrameIdxs[i]
                        if sequenceFrameIdx >= 0 and sequenceFrameIdx < len(self.semanticSequences[seqIdx][DICT_FRAMES_LOCATIONS].keys()) :
                            frameToShowKey = np.sort(self.semanticSequences[seqIdx][DICT_FRAMES_LOCATIONS].keys())[sequenceFrameIdx]
                        else :
                            frameToShowKey = -1

                        if frameToShowKey >= 0 and seqIdx >= 0 and seqIdx < len(self.semanticSequences) :
                            if seqIdx in self.preloadedPatches.keys() and frameToShowKey in self.preloadedPatches[seqIdx] :
                                visImg = self.drawOverlay(visImg, self.semanticSequences[seqIdx], frameToShowKey,
                                                          self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][self.selectedSequenceInstancesIdxes[i]][DICT_OFFSET],
                                                          self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][self.selectedSequenceInstancesIdxes[i]][DICT_SCALE],
                                                          False, True, True, False, self.preloadedPatches[seqIdx][frameToShowKey],
                                                          bboxColor = QtGui.QColor.fromRgb(col[0], col[1], col[2], 255))
                            else :
                                visImg = self.drawOverlay(visImg, self.semanticSequences[seqIdx], frameToShowKey,
                                                          self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][self.selectedSequenceInstancesIdxes[i]][DICT_OFFSET],
                                                          self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][self.selectedSequenceInstancesIdxes[i]][DICT_SCALE],
                                                          False, True, True, False, None,
                                                          bboxColor = QtGui.QColor.fromRgb(col[0], col[1], col[2], 255))
                    self.infoDialog.setVisImage(visImg)
       
    #### SET UI LISTS AND VISUALS ####        
      
    def setListOfLoadedSemSequences(self) :
#         print "saving bob 2"
#         self.bgImage.save("bob.png")
        if len(self.semanticSequences) > 0 :
            self.loadedSequencesListModel.setRowCount(len(self.semanticSequences))
            self.loadedSequencesDelegateList = []
            
            if np.any(self.bgImageLoc != None) :
                bgImg = np.array(Image.open(self.bgImageLoc))
            else :
                bgImg = np.zeros((720, 1280, 3), dtype=np.uint8)
                
            for i in xrange(0, len(self.semanticSequences)):
                self.loadedSequencesDelegateList.append(SequencesListDelegate())
                self.loadedSequencesListTable.setItemDelegateForRow(i, self.loadedSequencesDelegateList[-1])
                
                ## set sprite name
                self.loadedSequencesListModel.setItem(i, 0, QtGui.QStandardItem(self.semanticSequences[i][DICT_SEQUENCE_NAME]))
    
                ## set sprite icon
                if (DICT_ICON_TOP_LEFT in self.semanticSequences[i].keys() and
                    DICT_ICON_SIZE in self.semanticSequences[i].keys() and
                    DICT_ICON_FRAME_KEY in self.semanticSequences[i].keys()) :
                
                    if DICT_MASK_LOCATION in self.semanticSequences[i].keys() :
                        maskDir = self.semanticSequences[i][DICT_MASK_LOCATION]
                    else :
                        maskDir = self.dataPath + self.dataSet + self.semanticSequences[i][DICT_SEQUENCE_NAME] + "-maskedFlow-blended"
                    ## means I've the icon frame and it's been masked otherwise just load the original frame and use for the icon
                    if (os.path.isdir(maskDir) and
                        self.semanticSequences[i][DICT_ICON_FRAME_KEY] in self.semanticSequences[i][DICT_FRAMES_LOCATIONS].keys()) :

                        frameName = self.semanticSequences[i][DICT_FRAMES_LOCATIONS][self.semanticSequences[i][DICT_ICON_FRAME_KEY]].split(os.sep)[-1]

                        framePatch = np.array(Image.open(maskDir+"/"+frameName))
                        framePatch = framePatch[self.semanticSequences[i][DICT_ICON_TOP_LEFT][0]:self.semanticSequences[i][DICT_ICON_TOP_LEFT][0]+self.semanticSequences[i][DICT_ICON_SIZE],
                                                self.semanticSequences[i][DICT_ICON_TOP_LEFT][1]:self.semanticSequences[i][DICT_ICON_TOP_LEFT][1]+self.semanticSequences[i][DICT_ICON_SIZE], :]

                        bgPatch = bgImg[self.semanticSequences[i][DICT_ICON_TOP_LEFT][0]:self.semanticSequences[i][DICT_ICON_TOP_LEFT][0]+self.semanticSequences[i][DICT_ICON_SIZE],
                                        self.semanticSequences[i][DICT_ICON_TOP_LEFT][1]:self.semanticSequences[i][DICT_ICON_TOP_LEFT][1]+self.semanticSequences[i][DICT_ICON_SIZE], :]

                        iconPatch = (framePatch[:, :, :3]*(framePatch[:, :, -1].reshape((framePatch.shape[0], framePatch.shape[1], 1))/255.0) + 
                                     bgPatch[:, :, :3]*(1.0-(framePatch[:, :, -1].reshape((framePatch.shape[0], framePatch.shape[1], 1)))/255.0)).astype(np.uint8)
                        self.framePatch = framePatch

                        self.iconImage = np.ascontiguousarray(cv2.resize(iconPatch, (LIST_SECTION_SIZE, LIST_SECTION_SIZE), interpolation=cv2.INTER_AREA))
                    else :
                        frameKey = self.semanticSequences[i][DICT_ICON_FRAME_KEY]
                        if frameKey not in self.semanticSequences[i][DICT_FRAMES_LOCATIONS].keys() :
                            frameKey = self.semanticSequences[i][DICT_FRAMES_LOCATIONS].keys()[0]
                        
                        framePatch = np.array(Image.open(self.semanticSequences[i][DICT_FRAMES_LOCATIONS][frameKey]))
                        framePatch = framePatch[self.semanticSequences[i][DICT_ICON_TOP_LEFT][0]:self.semanticSequences[i][DICT_ICON_TOP_LEFT][0]+self.semanticSequences[i][DICT_ICON_SIZE],
                                                self.semanticSequences[i][DICT_ICON_TOP_LEFT][1]:self.semanticSequences[i][DICT_ICON_TOP_LEFT][1]+self.semanticSequences[i][DICT_ICON_SIZE], :]
                        
                        self.iconImage = np.ascontiguousarray(cv2.resize(framePatch[:, :, :3], (LIST_SECTION_SIZE, LIST_SECTION_SIZE), interpolation=cv2.INTER_AREA))
                        
                else :
                    self.iconImage = np.ascontiguousarray(bgImg[:LIST_SECTION_SIZE, :LIST_SECTION_SIZE, :3])
                
                self.loadedSequencesListTable.itemDelegateForRow(i).setIconImage(self.iconImage)
                
                ## set sprite color
                if DICT_REPRESENTATIVE_COLOR in self.semanticSequences[i].keys() :
                    col = self.semanticSequences[i][DICT_REPRESENTATIVE_COLOR]
                    self.loadedSequencesListTable.itemDelegateForRow(i).setBackgroundColor(QtGui.QColor.fromRgb(col[0], col[1], col[2], 255))
                    
            self.selectedSemSequenceIdx = 0
            self.loadedSequencesListTable.setEnabled(True)
        else :
            self.loadedSequencesListModel.setRowCount(1)
#             self.loadedSequencesListModel.setColumnCount(1)
            
            self.loadedSequencesDelegateList = [SequencesListDelegate()]
            self.loadedSequencesListTable.setItemDelegateForRow(0, self.loadedSequencesDelegateList[-1])
            self.loadedSequencesListModel.setItem(0, 0, QtGui.QStandardItem("None"))
            self.loadedSequencesListTable.setEnabled(False)
            
    def setListOfSequenceInstances(self) :
        if len(self.synthesisedSequence[DICT_SEQUENCE_INSTANCES]) > 0 :
            tmpSelectedInstances = np.copy(self.selectedSequenceInstancesIdxes)
            self.sequenceInstancesListModel.removeRows(0, self.sequenceInstancesListModel.rowCount())
            ## remove rows resets the list of selected rows in the table which will eventually call changeSelectedInstances and set self.selectedSequenceInstancesIdxes = []
            self.selectedSequenceInstancesIdxes = list(tmpSelectedInstances)
            self.sequenceInstancesListModel.setRowCount(len(self.synthesisedSequence[DICT_SEQUENCE_INSTANCES]))
            self.sequenceInstancesListModel.setColumnCount(1)
            self.sequenceInstancesDelegateList = []
            
            if len(self.selectedSequenceInstancesIdxes) > 0 and np.max(self.selectedSequenceInstancesIdxes) == len(self.synthesisedSequence[DICT_SEQUENCE_INSTANCES]) :
                if len(self.synthesisedSequence[DICT_SEQUENCE_INSTANCES]) - 1 not in self.selectedSequenceInstancesIdxes :
                    self.selectedSequenceInstancesIdxes[int(np.argwhere(self.selectedSequenceInstancesIdxes == np.max(self.selectedSequenceInstancesIdxes)))] = len(self.synthesisedSequence[DICT_SEQUENCE_INSTANCES]) - 1
                else :
                    del self.selectedSequenceInstancesIdxes[int(np.argwhere(self.selectedSequenceInstancesIdxes == np.max(self.selectedSequenceInstancesIdxes)))]
            print "HAHAH2", self.selectedSequenceInstancesIdxes

            for i in xrange(len(self.synthesisedSequence[DICT_SEQUENCE_INSTANCES])) :
                seqIdx = self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][i][DICT_SEQUENCE_IDX]
                if seqIdx >= 0 and seqIdx < len(self.semanticSequences) :
                    self.sequenceInstancesDelegateList.append(InstancesListDelegate())
                    self.sequenceInstancesListTable.setItemDelegateForRow(i, self.sequenceInstancesDelegateList[-1])

                    ## set sprite name
                    self.sequenceInstancesListModel.setItem(i, 0, QtGui.QStandardItem(self.semanticSequences[seqIdx][DICT_SEQUENCE_NAME]))
                
                    ## set sprite color
                    if DICT_REPRESENTATIVE_COLOR in self.semanticSequences[seqIdx].keys() :
                        col = self.semanticSequences[seqIdx][DICT_REPRESENTATIVE_COLOR]
                        self.sequenceInstancesListTable.itemDelegateForRow(i).setBackgroundColor(QtGui.QColor.fromRgb(col[0], col[1], col[2], 255))
                        
                    
                    if i in self.selectedSequenceInstancesIdxes :
                        self.sequenceInstancesListTable.setRowHeight(i, SLIDER_SELECTED_HEIGHT+2*SLIDER_PADDING)
                    else :
                        self.sequenceInstancesListTable.setRowHeight(i, SLIDER_NOT_SELECTED_HEIGHT+2*SLIDER_PADDING)

            self.sequenceInstancesListTable.setEnabled(True)
                
        else :
            self.sequenceInstancesListModel.setRowCount(1)

            self.sequenceInstancesDelegateList = [InstancesListDelegate()]
            self.sequenceInstancesListTable.setItemDelegateForRow(0, self.sequenceInstancesDelegateList[-1])
            self.sequenceInstancesListModel.setItem(0, 0, QtGui.QStandardItem("None"))
            self.sequenceInstancesListTable.setEnabled(False)
            
        
        ## reset height
        selectionHeight = len(self.selectedSequenceInstancesIdxes)*SLIDER_SELECTED_HEIGHT
        remainingHeight = (len(self.synthesisedSequence[DICT_SEQUENCE_INSTANCES])-len(self.selectedSequenceInstancesIdxes))*SLIDER_NOT_SELECTED_HEIGHT
        paddingHeight = (len(self.synthesisedSequence[DICT_SEQUENCE_INSTANCES]))*2*SLIDER_PADDING
        
        desiredHeight = np.max((SLIDER_MIN_HEIGHT, selectionHeight+remainingHeight+paddingHeight))
        
        self.sequenceInstancesListTable.setFixedHeight(desiredHeight)
      
    def setSemanticSliderDrawables(self) :
        if len(self.synthesisedSequence[DICT_SEQUENCE_INSTANCES]) > 0  :
            self.semanticsToDraw = []
            for i in xrange(0, len(self.synthesisedSequence[DICT_SEQUENCE_INSTANCES])):
                seqIdx = self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][i][DICT_SEQUENCE_IDX]
                if DICT_REPRESENTATIVE_COLOR in self.semanticSequences[seqIdx].keys() :
                    col = self.semanticSequences[seqIdx][DICT_REPRESENTATIVE_COLOR]
                else :
                    col = np.array([0, 0, 0])
                    
                self.semanticsToDraw.append((self.synthesisedSequence[DICT_SEQUENCE_INSTANCES]
                                             [i][DICT_DESIRED_SEMANTICS])[:len(self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][i][DICT_SEQUENCE_FRAMES]), :])
            
            self.frameIdxSlider.setSelectedSemantics(self.selectedSequenceInstancesIdxes)
            self.frameIdxSlider.setSemanticsToDraw(self.semanticsToDraw)
            
        else :
            self.frameIdxSlider.setSelectedSemantics([])
            self.frameIdxSlider.setSemanticsToDraw([])
            
    def setInstanceShowIndicatorDrawables(self) :
        if len(self.synthesisedSequence[DICT_SEQUENCE_INSTANCES]) > 0  :
            self.instancesToDraw = []
            for i in xrange(0, len(self.synthesisedSequence[DICT_SEQUENCE_INSTANCES])):
                seqIdx = self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][i][DICT_SEQUENCE_IDX]
                if DICT_REPRESENTATIVE_COLOR in self.semanticSequences[seqIdx].keys() :
                    col = self.semanticSequences[seqIdx][DICT_REPRESENTATIVE_COLOR]
                else :
                    col = np.array([0, 0, 0])
                self.instancesToDraw.append(col)
            self.instancesShowIndicator.setInstancesToDraw(self.instancesToDraw, self.instancesDoShow)
            
        else :
            self.instancesShowIndicator.setInstancesToDraw([], [])
            
    #### BACKGROUND IMAGE ####
            
    def setBgImage(self) :
        if not self.isSequenceLoaded :
            QtGui.QMessageBox.critical(self, "No Synthesised Sequence", ("<p align='center'> A synthesised sequence has not been loaded or created yet." +
                                                                         "<br>Please create a new sequence or load an existing one.</p>"))
            return
            
        fileName = QtGui.QFileDialog.getOpenFileName(self, "Set Background Image", self.loadedSynthesisedSequence, "Image Files (*.png)")[0]
        self.updateBgImage(fileName)
        
        ## update icons 
        self.setListOfLoadedSemSequences()
    
    def updateBgImage(self, fileName) :
        if fileName != "" :
            
#             im = np.ascontiguousarray(Image.open(self.bgImageLoc))
            self.bgImage = np.array(Image.open(fileName))[:, :, 0:3]
#             self.bgImage = QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_RGB888);
            self.frameLabel.setFixedSize(self.bgImage.shape[1], self.bgImage.shape[0])
            self.frameLabel.setImage(self.bgImage)
            self.bgImageLoc = os.sep.join(self.loadedSynthesisedSequence.split(os.sep)[:-1])+os.sep+"bgImage.png"
            Image.fromarray(self.bgImage.astype(np.uint8)).save(self.bgImageLoc)
            
            self.bgImageLoc = fileName
            
            if np.any(self.loadedSynthesisedSequence != None) :
                self.synthesisedSequence[DICT_SEQUENCE_BG] = self.bgImageLoc
            
        else :
            self.bgImageLoc = None
            self.bgImage = None
            self.frameLabel.setImage(None)
            
            if np.any(self.loadedSynthesisedSequence != None) :
                self.synthesisedSequence[DICT_SEQUENCE_BG] = ""
    
    
    #### NEW / LOAD / SAVE ####
    
    def newSynthesisedSequence(self) :
        dirLoc = QtGui.QFileDialog.getExistingDirectory(self, "New Synthesised Sequence Location", self.dataPath+"synthesisedSequences/")
        if dirLoc != "" :
            ## save currently loaded sequence
            if self.autoSaveBox.isChecked() :
                self.saveSynthesisedSequence()
                
            proceed = True
            if os.path.isfile(dirLoc+"/synthesised_sequence.npy") :
                proceed = QtGui.QMessageBox.question(self, 'Override Synthesised Sequence',
                                    "There is already a synthesised sequence in the chosen folder.\nDo you want to override?", 
                                    QtGui.QMessageBox.Yes | QtGui.QMessageBox.No, QtGui.QMessageBox.No) == QtGui.QMessageBox.Yes
            if proceed :
                self.loadedSynthesisedSequence = dirLoc + "/synthesised_sequence.npy"
                np.save(self.loadedSynthesisedSequence, {
                                                         DICT_USED_SEQUENCES:[],
                                                         DICT_SEQUENCE_INSTANCES:[],
                                                         DICT_SEQUENCE_BG:""
                                                         })
                self.loadSynthesisedSequenceAtLocation(self.loadedSynthesisedSequence)
            
    def loadSynthesisedSequence(self) :
#         fileName = QtGui.QFileDialog.getOpenFileName(self, "Load Synthesised Sequence", self.dataPath+"synthesisedSequences/", "Synthesised Sequences (synthesised_sequence*.npy)")[0]
        fileName = QtGui.QFileDialog.getOpenFileName(self, "Load Synthesised Sequence", "/home/ilisescu/PhD/data/synthesisedSequences/", "Synthesised Sequences (synthesised_sequence*.npy)")[0]
        if fileName != "" :
            ## save currently loaded sequence
            if self.autoSaveBox.isChecked() :
                self.saveSynthesisedSequence()
                
            self.loadSynthesisedSequenceAtLocation(fileName)
            
        return fileName
    
    def loadSynthesisedSequenceAtLocation(self, location) :
        if os.path.isfile(location) :
            self.isSequenceLoaded = False
            try :
                del self.synthesisedSequence
                self.synthesisedSequence = np.load(location).item()
            except :
                self.synthesisedSequence = np.load(location).item()

            try :
                del self.preloadedTransitionCosts
                self.preloadedTransitionCosts = {}
            except :            
                self.preloadedTransitionCosts = {}

            try :
                del self.preloadedDistanceMatrices
                self.preloadedDistanceMatrices = {}
            except :            
                self.preloadedDistanceMatrices = {}

            try :
                del self.preloadedPatches
                self.preloadedPatches = {}
            except :
                self.preloadedPatches = {}

            self.loadedSynthesisedSequence = location
            print "#####################"
            print
            print "LOADED SEQUENCE", location
            print 
            print "#####################"

            if len(self.synthesisedSequence[DICT_USED_SEQUENCES]) > 0 :
                self.dataPath = "/".join(self.synthesisedSequence[DICT_USED_SEQUENCES][0].split("/")[:-2]) + "/"
                self.dataSet = self.synthesisedSequence[DICT_USED_SEQUENCES][0].split("/")[-2] + "/"
            else :
                self.dataPath = "/".join(self.loadedSynthesisedSequence.split("/")[:-2]) + "/"
                self.dataSet = "/"

            print "SET PATH TO", self.dataPath, " + ", self.dataSet

            ## update background
            self.updateBgImage(self.synthesisedSequence[DICT_SEQUENCE_BG])
            

            ## load used semantic sequences
            print "#####################"
            print

            self.semanticSequences = []
            for index, seq in enumerate(self.synthesisedSequence[DICT_USED_SEQUENCES]) :
                with open(seq) as f:
                    self.semanticSequences.append(np.load(f).item())
                        
                if DICT_PATCHES_LOCATION in self.semanticSequences[-1].keys() :
                    with open(self.semanticSequences[-1][DICT_PATCHES_LOCATION]) as f :
                        self.preloadedPatches[index] = np.load(f).item()
                        print "loaded patches", self.semanticSequences[-1][DICT_PATCHES_LOCATION]; sys.stdout.flush()
                if DICT_TRANSITION_COSTS_LOCATION in self.semanticSequences[-1].keys() :
                    with open(self.semanticSequences[-1][DICT_TRANSITION_COSTS_LOCATION]) as f :
                        self.preloadedTransitionCosts[index] = np.load(f)
                        print "loaded costs", self.semanticSequences[-1][DICT_TRANSITION_COSTS_LOCATION]; sys.stdout.flush()
                if DICT_DISTANCE_MATRIX_LOCATION in self.semanticSequences[-1].keys() :
                    with open(self.semanticSequences[-1][DICT_DISTANCE_MATRIX_LOCATION]) as f :
                        self.preloadedDistanceMatrices[index] = np.load(f)
                        print "loaded distances", self.semanticSequences[-1][DICT_DISTANCE_MATRIX_LOCATION]; sys.stdout.flush()
                    
                ## check that they have defined actions
                if DICT_NUM_SEMANTICS in self.semanticSequences[-1].keys() :
                    tmpDoSave = False
                    # check if they have assigned names
                    if DICT_SEMANTICS_NAMES not in self.semanticSequences[-1].keys() :
                        self.semanticSequences[-1][DICT_SEMANTICS_NAMES] = {}
                        tmpDoSave = True

                    for semIdx in xrange(self.semanticSequences[-1][DICT_NUM_SEMANTICS]) :
                        if semIdx not in self.semanticSequences[-1][DICT_SEMANTICS_NAMES].keys() :
                            self.semanticSequences[-1][DICT_SEMANTICS_NAMES][semIdx] = "action{0:d}".format(semIdx)
                            print "ACTION NAME", self.semanticSequences[-1][DICT_SEMANTICS_NAMES][semIdx], "added for", self.semanticSequences[-1][DICT_SEQUENCE_NAME]
                            tmpDoSave = True
                    
                    ## check if they have assigned keybindings (or color bindings)
                    if DICT_COMMAND_TYPE not in self.semanticSequences[-1].keys() or DICT_COMMAND_BINDING not in self.semanticSequences[-1].keys() :
                        self.semanticSequences[-1][DICT_COMMAND_TYPE] = {}
                        self.semanticSequences[-1][DICT_COMMAND_BINDING] = {}
                        tmpDoSave = True

                    for semIdx in xrange(self.semanticSequences[-1][DICT_NUM_SEMANTICS]) :
                        if semIdx not in self.semanticSequences[-1][DICT_COMMAND_TYPE].keys() :
                            self.semanticSequences[-1][DICT_COMMAND_TYPE][semIdx] = DICT_COMMAND_TYPE_KEY
                            print "COMMAND TYPE DICT_COMMAND_TYPE_KEY added for", self.semanticSequences[-1][DICT_SEQUENCE_NAME]
                            tmpDoSave = True
                        if semIdx not in self.semanticSequences[-1][DICT_COMMAND_BINDING].keys() :
                            self.semanticSequences[-1][DICT_COMMAND_BINDING][semIdx] = np.string0(semIdx)
                            print "COMMAND BINDING ", self.semanticSequences[-1][DICT_COMMAND_BINDING][semIdx], " added for", self.semanticSequences[-1][DICT_SEQUENCE_NAME]
                            tmpDoSave = True
                    if tmpDoSave :
                        np.save(self.semanticSequences[-1][DICT_SEQUENCE_LOCATION], self.semanticSequences[-1])
            print
            print "#####################"
            
            
            ## resize the label showing the frame if there is no bgImage
            if self.bgImage == None and len(self.semanticSequences) > 0:
                if len(self.semanticSequences[0][DICT_FRAMES_LOCATIONS].keys()) > 0 :
                    height, width, channels = np.array(Image.open(self.semanticSequences[0][DICT_FRAMES_LOCATIONS][self.semanticSequences[0][DICT_FRAMES_LOCATIONS].keys()[0]])).shape
                    self.frameLabel.setFixedSize(width, height)

            ## set list of loaded semantic sequences
            self.setListOfLoadedSemSequences()
            if len(self.semanticSequences) > 0 :
                self.selectedSemSequenceIdx = 0
                self.loadedSequencesListTable.selectRow(self.selectedSemSequenceIdx)
                self.changeSelectedSemSequence(self.loadedSequencesListModel.item(self.selectedSemSequenceIdx).index())
            else :
                self.selectedSemSequenceIdx = -1

            ## set list of instantiated sequences
            self.instancesDoShow = list(np.ones(len(self.synthesisedSequence[DICT_SEQUENCE_INSTANCES]), bool))
            self.setListOfSequenceInstances()
#             if len(self.semanticSequences) > 0 :
#                 self.selectedSequenceInstancesIdxes = [0]
#                 self.sequenceInstancesListTable.selectRow(self.selectedSequenceInstancesIdxes[-1])
#     #             self.changeSelectedSequenceInstance(self.sequenceInstancesListTable.indexFromItem(self.sequenceInstancesListTable.item(self.selectedSequenceInstancesIdxes[-1], 0)))
#             else :
            self.selectedSequenceInstancesIdxes = []

            ## set sliders
            maxFrames = 0
            for i in xrange(len(self.synthesisedSequence[DICT_SEQUENCE_INSTANCES])) :
                maxFrames = np.max((maxFrames, len(self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][i][DICT_SEQUENCE_FRAMES])))
                
            if self.doPlaySounds :
                if os.path.isfile(os.sep.join(self.loadedSynthesisedSequence.split(os.sep)[:-1])+os.sep+"playSoundsTimes.npy") :
                    self.playSoundsTimes = np.load(os.sep.join(self.loadedSynthesisedSequence.split(os.sep)[:-1])+os.sep+"playSoundsTimes.npy").item()
                    print self.playSoundsTimes
                else :
                    self.playSoundsTimes = {}

            if maxFrames > 0 :
                self.frameIdxSlider.setMaximum(maxFrames-1)
                self.frameIdxSpinBox.setRange(0, maxFrames-1)
            else :
                self.frameIdxSlider.setMaximum(0)
                self.frameIdxSpinBox.setRange(0, 0)

            self.setSemanticSliderDrawables()
            self.setInstanceShowIndicatorDrawables()

            ## render
            self.frameIdx = 0
            if self.frameIdxSpinBox.value() == self.frameIdx :
                self.showFrame(self.frameIdx)
            else :
                self.frameIdxSpinBox.setValue(self.frameIdx)

            ## load tagged frames
            if os.path.isfile("/".join(self.loadedSynthesisedSequence.split("/")[:-1])+"/tagged_frames.npy") :
                self.taggedFrames = np.load("/".join(self.loadedSynthesisedSequence.split("/")[:-1])+"/tagged_frames.npy").item()
            else :
                self.taggedFrames = {}
            self.deleteSequenceButton.setEnabled(True)
            self.playSequenceButton.setEnabled(True)
            self.loadNumberSequenceButton.setEnabled(True)
            self.frameInfo.setText("Loaded sequence at " + self.loadedSynthesisedSequence)
            self.isSequenceLoaded = True
            self.optimizationDownsampleRateSpinBox.setValue(1)
    
    def saveSynthesisedSequence(self) :
        if np.any(self.loadedSynthesisedSequence != None) :
            np.save(self.loadedSynthesisedSequence, self.synthesisedSequence)
            if self.doPlaySounds :
                np.save(os.sep.join(self.loadedSynthesisedSequence.split(os.sep)[:-1])+os.sep+"playSoundsTimes.npy", self.playSoundsTimes)
            
            self.frameInfo.setText("Saved sequence at " + self.loadedSynthesisedSequence)
    
    def loadSemanticSequence(self) :
        if not self.isSequenceLoaded :
            QtGui.QMessageBox.critical(self, "No Synthesised Sequence", ("<p align='center'> A synthesised sequence has not been loaded or created yet." +
                                                                         "<br>Please create a new sequence or load an existing one.</p>"))
            return
        
        fileNames = QtGui.QFileDialog.getOpenFileNames(self, "Load Actor Sequence(s)", self.dataPath, "Actor Sequences (semantic_sequence*.npy)")[0]
        for fileName in fileNames :
            semanticSequence = np.load(fileName).item()
            status, message = self.checkSemanticSequence(semanticSequence)
            if status == 0 :
                self.semanticSequences.append(semanticSequence)
                if DICT_PATCHES_LOCATION in self.semanticSequences[-1].keys() :
                    self.preloadedPatches[len(self.semanticSequences)-1] = np.load(self.semanticSequences[-1][DICT_PATCHES_LOCATION]).item()
                if DICT_TRANSITION_COSTS_LOCATION in self.semanticSequences[-1].keys() :
                    self.preloadedTransitionCosts[len(self.semanticSequences)-1] = np.load(self.semanticSequences[-1][DICT_TRANSITION_COSTS_LOCATION])
                if DICT_DISTANCE_MATRIX_LOCATION in self.semanticSequences[-1].keys() :
                    self.preloadedDistanceMatrices[len(self.semanticSequences)-1] = np.load(self.semanticSequences[-1][DICT_DISTANCE_MATRIX_LOCATION])
                    
                ## check that they have defined actions 
                if DICT_NUM_SEMANTICS in self.semanticSequences[-1].keys() :
                    tmpDoSave = False
                    # check if they have assigned names
                    if DICT_SEMANTICS_NAMES not in self.semanticSequences[-1].keys() :
                        self.semanticSequences[-1][DICT_SEMANTICS_NAMES] = {}
                        tmpDoSave = True

                    for semIdx in xrange(self.semanticSequences[-1][DICT_NUM_SEMANTICS]) :
                        if semIdx not in self.semanticSequences[-1][DICT_SEMANTICS_NAMES].keys() :
                            self.semanticSequences[-1][DICT_SEMANTICS_NAMES][semIdx] = "action{0:d}".format(semIdx)
                            print "ACTION NAME", self.semanticSequences[-1][DICT_SEMANTICS_NAMES][semIdx], "added for", self.semanticSequences[-1][DICT_SEQUENCE_NAME]
                            tmpDoSave = True
                    
                    ## check if they have assigned keybindings (or color bindings)
                    if DICT_COMMAND_TYPE not in self.semanticSequences[-1].keys() or DICT_COMMAND_BINDING not in self.semanticSequences[-1].keys() :
                        self.semanticSequences[-1][DICT_COMMAND_TYPE] = {}
                        self.semanticSequences[-1][DICT_COMMAND_BINDING] = {}
                        tmpDoSave = True

                    for semIdx in xrange(self.semanticSequences[-1][DICT_NUM_SEMANTICS]) :
                        if semIdx not in self.semanticSequences[-1][DICT_COMMAND_TYPE].keys() :
                            self.semanticSequences[-1][DICT_COMMAND_TYPE][semIdx] = DICT_COMMAND_TYPE_KEY
                            print "COMMAND TYPE DICT_COMMAND_TYPE_KEY added for", self.semanticSequences[-1][DICT_SEQUENCE_NAME]
                            tmpDoSave = True
                        if semIdx not in self.semanticSequences[-1][DICT_COMMAND_BINDING].keys() :
                            self.semanticSequences[-1][DICT_COMMAND_BINDING][semIdx] = np.string0(semIdx)
                            print "COMMAND BINDING ", self.semanticSequences[-1][DICT_COMMAND_BINDING][semIdx], " added for", self.semanticSequences[-1][DICT_SEQUENCE_NAME]
                            tmpDoSave = True
                    if tmpDoSave :
                        np.save(self.semanticSequences[-1][DICT_SEQUENCE_LOCATION], self.semanticSequences[-1])

                self.setListOfLoadedSemSequences()
            else :
                QtGui.QMessageBox.warning(self, "Invalid Actor Sequence", ("<p align='center'>This actor sequence cannot be used for synthesis:<br><br>\"<b>"+message+ 
                                                                           "</b>\"<br><br>Please return to the <i>Define Actor Sequences</i> tab.</p>"))
            
    def checkSemanticSequence(self, semanticSequence) :
        if DICT_DISTANCE_MATRIX_LOCATION not in semanticSequence.keys() :
            return 1, "Distance Matrix not computed"
        if np.load(semanticSequence[DICT_DISTANCE_MATRIX_LOCATION]).shape[0] != len(semanticSequence[DICT_FRAMES_LOCATIONS].keys()) :
            return 11, "Mismatch between distance matrix and number of frames"

        if DICT_TRANSITION_COSTS_LOCATION not in semanticSequence.keys() :
            return 2, "Transition costs not computed"
        if np.load(semanticSequence[DICT_TRANSITION_COSTS_LOCATION]).shape[0] != len(semanticSequence[DICT_FRAMES_LOCATIONS].keys()) :
            return 21, "Mismatch between transition matrix and number of frames"

        ## only care about the stuff below if masks have been defined
        if DICT_MASK_LOCATION in semanticSequence.keys() :
            frameKeys = np.sort(semanticSequence[DICT_FRAMES_LOCATIONS].keys())

            if DICT_PATCHES_LOCATION not in semanticSequence.keys() :
                return 31, "Segmentation incomplete (patches not computed)"
            patchKeys = np.load(semanticSequence[DICT_PATCHES_LOCATION]).item().keys()

            for i, key in enumerate(frameKeys) :
                if key not in semanticSequence[DICT_BBOXES].keys() and i < len(frameKeys)-1 :
                    return 3, "BBox not defined for frame "+np.string_(key)

                if key not in patchKeys and i < len(frameKeys)-1 :
                    return 32, "Segmentation incomplete (patch for frame "+np.string_(key)+" not available)"

        if DICT_FRAME_SEMANTICS not in semanticSequence.keys() :
            return 4, "Actions not defined"
        if semanticSequence[DICT_FRAME_SEMANTICS].shape[0] != len(semanticSequence[DICT_FRAMES_LOCATIONS].keys()) :
            return 41, "Mismatch between action vector and number of frames"

        return 0, ""
            
    #### CLEANUP ####
    
    def cleanup(self) :
        self.doPlaySequence = False
        self.playSequenceButton.setIcon(self.playIcon)
        self.playTimer.stop()

        self.frameInfo.setText(self.oldInfoText)
        
        if self.autoSaveBox.isChecked() :
            self.saveSynthesisedSequence()
        
        try :
            del self.preloadedPatches
    #         del self.preloadedTransitionCosts
    #         del self.synthesisedSequence
        except :
            print

        if self.infoDialog.isVisible() :
            self.infoDialog.done(0)
            
    #### GUI STUFF ####
    
    def createGUI(self) :
        
        ## WIDGETS ##
        
        self.frameLabel = ImageLabel("Frame")
        self.frameLabel.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.MinimumExpanding)
        self.frameLabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignHCenter)
        self.frameLabel.setStyleSheet("QLabel { margin: 0px; border: 1px solid gray; border-radius: 0px; }")
        self.frameLabel.installEventFilter(self)
        
        self.frameInfo = QtGui.QLabel("Info text")
        self.frameInfo.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignHCenter)
        
        self.instancesShowIndicator = InstancesShowWidget()
        
        self.frameIdxSlider = SemanticsSlider(QtCore.Qt.Horizontal)
        self.frameIdxSlider.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Minimum)
        self.frameIdxSlider.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.frameIdxSlider.setTickPosition(QtGui.QSlider.TicksBothSides)
        self.frameIdxSlider.setMinimum(0)
        self.frameIdxSlider.setMaximum(0)
        self.frameIdxSlider.setTickInterval(50)
        self.frameIdxSlider.setSingleStep(1)
        self.frameIdxSlider.setPageStep(100)
        self.frameIdxSlider.installEventFilter(self)
    
        self.frameIdxSpinBox = QtGui.QSpinBox()
        self.frameIdxSpinBox.setRange(0, 0)
        self.frameIdxSpinBox.setSingleStep(1)
        self.frameIdxSpinBox.installEventFilter(self)
        
        self.renderFpsSpinBox = QtGui.QSpinBox()
        self.renderFpsSpinBox.setRange(1, 60)
        self.renderFpsSpinBox.setSingleStep(1)
        self.renderFpsSpinBox.setValue(30)
        self.renderFpsSpinBox.setToolTip("FPS to use when playing back the synthesised sequence")
        
#         self.sequenceInstancesListTable = QtGui.QTableWidget(1, 1)
#         self.sequenceInstancesListTable.horizontalHeader().setResizeMode(0, QtGui.QHeaderView.Stretch)
#         self.sequenceInstancesListTable.setHorizontalHeaderItem(0, QtGui.QTableWidgetItem("Instances"))
#         self.sequenceInstancesListTable.horizontalHeader().setResizeMode(QtGui.QHeaderView.Fixed)
#         self.sequenceInstancesListTable.setSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.MinimumExpanding)
#         self.sequenceInstancesListTable.setFixedWidth(180)
#         self.sequenceInstancesListTable.setEditTriggers(QtGui.QAbstractItemView.NoEditTriggers)
#         self.sequenceInstancesListTable.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
#         self.sequenceInstancesListTable.setItem(0, 0, QtGui.QTableWidgetItem("No instances"))
#         self.sequenceInstancesListTable.setEnabled(False)
#         self.sequenceInstancesListTable.installEventFilter(self)

        self.sequenceInstancesListModel = QtGui.QStandardItemModel(1, 1)
        self.sequenceInstancesListModel.setHorizontalHeaderLabels(["Instances"])
        self.sequenceInstancesListModel.setItem(0, 0, QtGui.QStandardItem("None"))
        
        self.sequenceInstancesListTable = QtGui.QTableView()
        self.sequenceInstancesListTable.setEditTriggers(QtGui.QAbstractItemView.NoEditTriggers)
        self.sequenceInstancesListTable.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
        self.sequenceInstancesListTable.horizontalHeader().setStretchLastSection(True)
        self.sequenceInstancesListTable.horizontalHeader().hide()
        self.sequenceInstancesListTable.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Minimum)
        self.sequenceInstancesListTable.verticalHeader().setDefaultSectionSize(SLIDER_NOT_SELECTED_HEIGHT)
        self.sequenceInstancesListTable.setMinimumHeight(10)
        self.sequenceInstancesListTable.setFixedWidth(120)
        self.sequenceInstancesListTable.setEnabled(False)
        self.sequenceInstancesListTable.verticalHeader().hide()
        self.sequenceInstancesListTable.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.sequenceInstancesListTable.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.sequenceInstancesListTable.setStyleSheet("QTableView { border: none; } QTableView::item { selection-background-color:red;}")
        self.sequenceInstancesListTable.setShowGrid(False)
        
        self.sequenceInstancesDelegateList = [InstancesListDelegate()]
        self.sequenceInstancesListTable.setItemDelegateForRow(0, self.sequenceInstancesDelegateList[-1])
        self.sequenceInstancesListTable.setModel(self.sequenceInstancesListModel)


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
        self.loadedSequencesListTable.verticalHeader().setDefaultSectionSize(LIST_SECTION_SIZE)
        self.loadedSequencesListTable.setMinimumHeight(10)
        self.loadedSequencesListTable.setEnabled(False)

        self.loadedSequencesDelegateList = [SequencesListDelegate()]
        self.loadedSequencesListTable.setItemDelegateForRow(0, self.loadedSequencesDelegateList[-1])
        self.loadedSequencesListTable.setModel(self.loadedSequencesListModel)
        
        self.drawSpritesBox = QtGui.QCheckBox("Render Sprites")
        self.drawSpritesBox.setChecked(True)
        self.drawBBoxBox = QtGui.QCheckBox("Render Bounding Box")
        self.drawCenterBox = QtGui.QCheckBox("Render BBox Center")
        
        self.playSequenceButton = QtGui.QToolButton()
        self.playSequenceButton.setToolTip("Play Generated Sequence")
        self.playSequenceButton.setCheckable(False)
        self.playSequenceButton.setShortcut(QtGui.QKeySequence("Alt+P"))
        self.playSequenceButton.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        self.playSequenceButton.setIcon(self.playIcon)
        self.playSequenceButton.setEnabled(False)
        
        self.toggleDelaySpinBox = QtGui.QSpinBox()
        self.toggleDelaySpinBox.setRange(2, 300)
        self.toggleDelaySpinBox.setValue(6)
        self.toggleDelaySpinBox.setSingleStep(2)
        self.toggleDelaySpinBox.setToolTip("Number of frames to wait before toggling to desired semantic class using smoothstep")
        
        self.extendLengthSpinBox = QtGui.QSpinBox()
        self.extendLengthSpinBox.setRange(10, 1000)
        self.extendLengthSpinBox.setSingleStep(10)
        self.extendLengthSpinBox.setValue(self.EXTEND_LENGTH-1)
        self.extendLengthSpinBox.setToolTip("Number of frames to synthesise in one go")
        
        self.autoSaveBox = QtGui.QCheckBox("Autosave")
#         self.autoSaveBox.setChecked(True)

        
        self.deleteSequenceButton = QtGui.QPushButton("Delete Sequence")
        self.deleteSequenceButton.setEnabled(False)
        
        self.loadNumberSequenceButton = QtGui.QPushButton("Synthesise by numbers")
        self.loadNumberSequenceButton.setEnabled(False)
        self.showNumbersSequenceBox = QtGui.QCheckBox("Show")
        self.showNumbersSequenceBox.setChecked(True)
        self.loopSequenceBox = QtGui.QCheckBox("Loop")
        self.loopSequenceBox.setChecked(False)
        
        self.resolveCompatibilityBox = QtGui.QCheckBox("Solve Compatibility")
#         self.resolveCompatibilityBox.setChecked(True)
        self.resolveCompatibilityBox.setToolTip("Use or ignore vertical <i>compatibility</i> cost")
        
        self.randomizeSequenceBox = QtGui.QCheckBox("Randomize Synth")
        self.randomizeSequenceBox.setToolTip("Repeat synthesis after randomly changing some unary costs")
#         self.randomizeSequenceBox.setChecked(True)

        self.toggleSpeedDeltaSpinBox = QtGui.QDoubleSpinBox()
        self.toggleSpeedDeltaSpinBox.setRange(0.0, 0.5)
        self.toggleSpeedDeltaSpinBox.setSingleStep(0.01)
        self.toggleSpeedDeltaSpinBox.setValue(0.1)
        self.toggleSpeedDeltaSpinBox.setToolTip("Cost added to the transition cost. The higher, the more costly it is to follow the timeline")

        self.semanticsImportanceSpinBox = QtGui.QDoubleSpinBox()
        self.semanticsImportanceSpinBox.setRange(0.0, 4000.0)
        self.semanticsImportanceSpinBox.setSingleStep(1.0)
        self.semanticsImportanceSpinBox.setValue(50.0)
        self.semanticsImportanceSpinBox.setToolTip("The higher the number the more important it is to show the semantics the user asked for")
        
        self.propagationSigmaSpinBox = QtGui.QDoubleSpinBox()
        self.propagationSigmaSpinBox.setRange(0.01, 100.0)
        self.propagationSigmaSpinBox.setSingleStep(0.01)
        self.propagationSigmaSpinBox.setValue(0.02)
        self.propagationSigmaSpinBox.setButtonSymbols(QtGui.QAbstractSpinBox.NoButtons)
        self.propagationSigmaSpinBox.setToolTip("The lower, the easier it is for frames to be given the same action label (the distance is less important)")
        
        self.costsAlphaSpinBox = QtGui.QDoubleSpinBox()
        self.costsAlphaSpinBox.setRange(0.0, 1.0)
        self.costsAlphaSpinBox.setSingleStep(0.01)
        self.costsAlphaSpinBox.setValue(0.65)
        self.costsAlphaSpinBox.setToolTip("The higher, the more important the unaries over the pairwise")
        
        self.compatibilityAlphaSpinBox = QtGui.QDoubleSpinBox()
        self.compatibilityAlphaSpinBox.setRange(0.0, 1.0)
        self.compatibilityAlphaSpinBox.setSingleStep(0.01)
        self.compatibilityAlphaSpinBox.setValue(0.65)
        self.compatibilityAlphaSpinBox.setToolTip("The higher, the more important the vertical over the horizontal pairwise")
        
        
        self.sequenceXOffsetSpinBox = QtGui.QSpinBox()
        self.sequenceXOffsetSpinBox.setRange(-1280, 1280)
        self.sequenceXOffsetSpinBox.setSingleStep(1)
        self.sequenceXOffsetSpinBox.setValue(0)
        
        self.sequenceYOffsetSpinBox = QtGui.QSpinBox()
        self.sequenceYOffsetSpinBox.setRange(-720, 720)
        self.sequenceYOffsetSpinBox.setSingleStep(1)
        self.sequenceYOffsetSpinBox.setValue(0)
        
        self.sequenceXScaleSpinBox = QtGui.QDoubleSpinBox()
        self.sequenceXScaleSpinBox.setRange(0.0, 5.0)
        self.sequenceXScaleSpinBox.setSingleStep(0.01)
        self.sequenceXScaleSpinBox.setValue(1.0)
        
        self.sequenceYScaleSpinBox = QtGui.QDoubleSpinBox()
        self.sequenceYScaleSpinBox.setRange(0.0, 5.0)
        self.sequenceYScaleSpinBox.setSingleStep(0.01)
        self.sequenceYScaleSpinBox.setValue(1.0)
        
        self.startingSemanticsComboBox = QtGui.QComboBox()
        
        self.addNewSemanticSequenceButton = QtGui.QPushButton("Add To Output")
        self.cancelNewSemanticSequenceButton = QtGui.QPushButton("Cancel")
        
        
        self.compatibilitySigmaSpinBox = QtGui.QDoubleSpinBox()
        self.compatibilitySigmaSpinBox.setRange(0.01, 100.0)
        self.compatibilitySigmaSpinBox.setSingleStep(0.01)
        self.compatibilitySigmaSpinBox.setValue(0.1)
        self.compatibilitySigmaSpinBox.setButtonSymbols(QtGui.QAbstractSpinBox.NoButtons)
        self.compatibilitySigmaSpinBox.setToolTip("The higher, the wider the distribution used to compare frame feats (e.g. semantic vectors for label propagation-based cost)")
        
        self.compatibilitySigmaDividerSpinBox = QtGui.QSpinBox()
        self.compatibilitySigmaDividerSpinBox.setRange(1, 1000)
        self.compatibilitySigmaDividerSpinBox.setSingleStep(1)
        self.compatibilitySigmaDividerSpinBox.setValue(1)
        self.compatibilitySigmaDividerSpinBox.setPrefix("/")
        self.compatibilitySigmaDividerSpinBox.setButtonSymbols(QtGui.QAbstractSpinBox.NoButtons)
        self.compatibilitySigmaDividerSpinBox.setToolTip("Divide sigma by this number")
        
        self.compatibilityTypeComboBox = QtGui.QComboBox()
        self.compatibilityTypeComboBox.addItems(["Label Propagation", "L2 Gaussian", "Shortest Path"])
        self.compatibilityTypeComboBox.setCurrentIndex(0)
        
        self.pathCompatibilityMinJumpLengthSpinBox = QtGui.QSpinBox()
        self.pathCompatibilityMinJumpLengthSpinBox.setRange(1, 50)
        self.pathCompatibilityMinJumpLengthSpinBox.setSingleStep(1)
        self.pathCompatibilityMinJumpLengthSpinBox.setValue(1)
        self.pathCompatibilityMinJumpLengthSpinBox.setToolTip("When computing shortest path, throw away all jumps shorter than this number")
        
        self.infoDialog = InfoDialog(self, "Compatibility Information")
        self.infoDialog.imageLabel.installEventFilter(self)
        self.infoDialog.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        self.infoDialog.installEventFilter(self)
        
        self.optimizationDownsampleRateSpinBox = QtGui.QSpinBox()
        self.optimizationDownsampleRateSpinBox.setRange(1, 4)
        self.optimizationDownsampleRateSpinBox.setSingleStep(1)
        self.optimizationDownsampleRateSpinBox.setValue(1)
        self.optimizationDownsampleRateSpinBox.setToolTip("Performs the optimization using each frame or every second, third or fourth frame and getting increasingly faster")
        
        ## SIGNALS ##
        
        self.frameIdxSlider.valueChanged[int].connect(self.frameIdxSpinBox.setValue)
        self.frameIdxSpinBox.valueChanged[int].connect(self.frameIdxSlider.setValue)
        self.frameIdxSpinBox.valueChanged[int].connect(self.showFrame)
        
        self.renderFpsSpinBox.valueChanged[int].connect(self.setRenderFps)
        
        self.loadedSequencesListTable.clicked.connect(self.changeSelectedSemSequence)
#         self.sequenceInstancesListTable.clicked.connect(self.changeSelectedInstances)
        selectionModel = self.sequenceInstancesListTable.selectionModel()
        selectionModel.selectionChanged.connect(self.changeSelectedInstances)
        
        self.playSequenceButton.clicked.connect(self.playSequenceButtonPressed)
        self.deleteSequenceButton.clicked.connect(self.deleteGeneratedSequence)
        self.loadNumberSequenceButton.clicked.connect(self.loadNumberSequence)
        
        
        self.sequenceXOffsetSpinBox.valueChanged.connect(self.updateNewSemanticSequenceTransformation)
        self.sequenceYOffsetSpinBox.valueChanged.connect(self.updateNewSemanticSequenceTransformation)
        self.sequenceXScaleSpinBox.valueChanged.connect(self.updateNewSemanticSequenceTransformation)
        self.sequenceYScaleSpinBox.valueChanged.connect(self.updateNewSemanticSequenceTransformation)
        
        self.startingSemanticsComboBox.currentIndexChanged[int].connect(self.currentStartingSemanticsChanged)
        
        self.toggleDelaySpinBox.valueChanged[int].connect(self.setToggleDelay)
        self.extendLengthSpinBox.valueChanged[int].connect(self.setExtendLength)
        
        self.addNewSemanticSequenceButton.clicked.connect(self.addNewSemanticSequence)
        self.cancelNewSemanticSequenceButton.clicked.connect(self.cancelNewSemanticSequence)
        
        self.propagationSigmaSpinBox.editingFinished.connect(self.updateCompatibilityLabelsAndMatrices)
        self.compatibilitySigmaSpinBox.editingFinished.connect(self.updateCompatibilityMatrices)
        self.compatibilitySigmaDividerSpinBox.editingFinished.connect(self.updateCompatibilityMatrices)
        self.compatibilityTypeComboBox.currentIndexChanged.connect(self.updateCompatibilityMatrices)
        self.pathCompatibilityMinJumpLengthSpinBox.editingFinished.connect(self.updateCompatibilityMatrices)
        
        self.optimizationDownsampleRateSpinBox.editingFinished.connect(self.updatePreloadedTransitionMatrices)
        
        self.instancesShowIndicator.instanceDoShowChanged[int].connect(self.toggleInstancesDoShow)
        
        ## LAYOUTS ##
        
        mainLayout = QtGui.QHBoxLayout()
        
        renderingControls = QtGui.QGroupBox("Rendering Controls")
        renderingControls.setStyleSheet("QGroupBox { margin: 5px; border: 2px groove gray; border-radius: 3px; } QGroupBox::title {left: 15px; top: -7px; font: bold;}")
        renderingControlsLayout = QtGui.QVBoxLayout()
        renderingControlsLayout.addWidget(self.drawSpritesBox)
        renderingControlsLayout.addWidget(self.drawBBoxBox)
        renderingControlsLayout.addWidget(self.drawCenterBox)
        renderingControlsLayout.addWidget(self.playSequenceButton)
        renderingControlsLayout.addWidget(self.renderFpsSpinBox)
        renderingControls.setLayout(renderingControlsLayout)
        
        
        sequenceControls = QtGui.QGroupBox("Sequence Controls")
        sequenceControls.setStyleSheet("QGroupBox { margin: 5px; border: 2px groove gray; border-radius: 3px; } QGroupBox::title {left: 15px; top: -7px; font: bold;}")
        sequenceControlsLayout = QtGui.QGridLayout(); idx = 0
        sequenceControlsLayout.addWidget(self.loadNumberSequenceButton, idx, 0, 1, 2, QtCore.Qt.AlignCenter)
        sequenceControlsLayout.addWidget(self.showNumbersSequenceBox, idx, 2, 1, 1, QtCore.Qt.AlignCenter)
        sequenceControlsLayout.addWidget(self.loopSequenceBox, idx, 3, 1, 1, QtCore.Qt.AlignCenter); idx += 1
        sequenceControlsLayout.addWidget(self.resolveCompatibilityBox, idx, 0, 1, 2, QtCore.Qt.AlignLeft);
        sequenceControlsLayout.addWidget(self.randomizeSequenceBox, idx, 2, 1, 2, QtCore.Qt.AlignRight); idx += 1
        sequenceControlsLayout.addWidget(self.deleteSequenceButton, idx, 0, 1, 2, QtCore.Qt.AlignLeft);
        sequenceControlsLayout.addWidget(self.autoSaveBox, idx, 2, 1, 2, QtCore.Qt.AlignRight); idx += 1
        sequenceControlsLayout.addWidget(QtGui.QLabel("Transition Cost Delta"), idx, 0, 1, 2, QtCore.Qt.AlignLeft);
        sequenceControlsLayout.addWidget(self.toggleSpeedDeltaSpinBox, idx, 2, 1, 2, QtCore.Qt.AlignRight); idx += 1
#         sequenceControlsLayout.addWidget(QtGui.QLabel("Action Importance"), idx, 0, 1, 2, QtCore.Qt.AlignLeft);
#         sequenceControlsLayout.addWidget(self.semanticsImportanceSpinBox, idx, 2, 1, 2, QtCore.Qt.AlignRight); idx += 1
        sequenceControlsLayout.addWidget(QtGui.QLabel("Propagation Sigma [/10]"), idx, 0, 1, 2, QtCore.Qt.AlignLeft);
        sequenceControlsLayout.addWidget(self.propagationSigmaSpinBox, idx, 2, 1, 2, QtCore.Qt.AlignRight); idx += 1
        sequenceControlsLayout.addWidget(QtGui.QLabel("a"), idx, 0, 1, 1, QtCore.Qt.AlignLeft);
        sequenceControlsLayout.addWidget(self.costsAlphaSpinBox, idx, 1, 1, 1, QtCore.Qt.AlignLeft);
        sequenceControlsLayout.addWidget(QtGui.QLabel("b"), idx, 2, 1, 1, QtCore.Qt.AlignRight);
        sequenceControlsLayout.addWidget(self.compatibilityAlphaSpinBox, idx, 3, 1, 1, QtCore.Qt.AlignRight); idx += 1
        sequenceControlsLayout.addWidget(QtGui.QLabel("Optimization speed"), idx, 0, 1, 2, QtCore.Qt.AlignLeft);
        sequenceControlsLayout.addWidget(self.optimizationDownsampleRateSpinBox, idx, 2, 1, 2, QtCore.Qt.AlignRight); idx += 1
        sequenceControlsLayout.addWidget(QtGui.QLabel("Toogle Delay"), idx, 0, 1, 2, QtCore.Qt.AlignLeft);
        sequenceControlsLayout.addWidget(self.toggleDelaySpinBox, idx, 2, 1, 2, QtCore.Qt.AlignRight); idx += 1
        sequenceControlsLayout.addWidget(QtGui.QLabel("Extend Length"), idx, 0, 1, 2, QtCore.Qt.AlignLeft);
        sequenceControlsLayout.addWidget(self.extendLengthSpinBox, idx, 2, 1, 2, QtCore.Qt.AlignRight); idx += 1
        sequenceControlsLayout.addWidget(QtGui.QLabel("Compatiblity Sigma"), idx, 0, 1, 2, QtCore.Qt.AlignLeft);
        sequenceControlsLayout.addWidget(self.compatibilitySigmaSpinBox, idx, 2, 1, 1, QtCore.Qt.AlignRight);
        sequenceControlsLayout.addWidget(self.compatibilitySigmaDividerSpinBox, idx, 3, 1, 1, QtCore.Qt.AlignRight); idx += 1
        sequenceControlsLayout.addWidget(QtGui.QLabel("Compatiblity Type"), idx, 0, 1, 2, QtCore.Qt.AlignLeft);
        sequenceControlsLayout.addWidget(self.compatibilityTypeComboBox, idx, 2, 1, 2, QtCore.Qt.AlignRight); idx += 1
        sequenceControlsLayout.addWidget(QtGui.QLabel("Dijkstra Min Jump Length"), idx, 0, 1, 2, QtCore.Qt.AlignLeft);
        sequenceControlsLayout.addWidget(self.pathCompatibilityMinJumpLengthSpinBox, idx, 2, 1, 2, QtCore.Qt.AlignRight); idx += 1
        sequenceControls.setLayout(sequenceControlsLayout)
        
        
#         self.addNewSemanticSequenceControls = QtGui.QGroupBox("New Sequence Instance Controls")
#         self.addNewSemanticSequenceControls.setStyleSheet("QGroupBox { margin: 5px; border: 2px groove gray; border-radius: 3px; } QGroupBox::title {left: 15px; top: -7px; font: bold;}")
        self.addNewSemanticSequenceControls = QtGui.QDialog(self)
        self.addNewSemanticSequenceControls.setWindowTitle("Add Instance to Output")
        addNewSemanticSequenceControlsLayout = QtGui.QGridLayout(); idx = 0
        addNewSemanticSequenceControlsLayout.addWidget(QtGui.QLabel("Offset (x, y)"), idx, 0, 1, 1, QtCore.Qt.AlignLeft)
        addNewSemanticSequenceControlsLayout.addWidget(self.sequenceXOffsetSpinBox, idx, 1, 1, 1, QtCore.Qt.AlignLeft)
        addNewSemanticSequenceControlsLayout.addWidget(self.sequenceYOffsetSpinBox, idx, 2, 1, 1, QtCore.Qt.AlignLeft); idx += 1
        addNewSemanticSequenceControlsLayout.addWidget(QtGui.QLabel("Scale (x, y)"), idx, 0, 1, 1, QtCore.Qt.AlignLeft)
        addNewSemanticSequenceControlsLayout.addWidget(self.sequenceXScaleSpinBox, idx, 1, 1, 1, QtCore.Qt.AlignLeft)
        addNewSemanticSequenceControlsLayout.addWidget(self.sequenceYScaleSpinBox, idx, 2, 1, 1, QtCore.Qt.AlignLeft); idx += 1
        addNewSemanticSequenceControlsLayout.addWidget(QtGui.QLabel("Starting action"), idx, 0, 1, 1, QtCore.Qt.AlignLeft)
        addNewSemanticSequenceControlsLayout.addWidget(self.startingSemanticsComboBox, idx, 1, 1, 2, QtCore.Qt.AlignLeft); idx += 1
        horizontalLine =  QtGui.QFrame()
        horizontalLine.setFrameStyle(QtGui.QFrame.HLine)
        horizontalLine.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        addNewSemanticSequenceControlsLayout.addWidget(horizontalLine,idx, 0 , 1, 3); idx += 1
        addNewSemanticSequenceControlsLayout.addWidget(self.cancelNewSemanticSequenceButton, idx, 0, 1, 1, QtCore.Qt.AlignCenter)
        addNewSemanticSequenceControlsLayout.addWidget(self.addNewSemanticSequenceButton, idx, 1, 1, 2, QtCore.Qt.AlignCenter); idx += 1
        self.addNewSemanticSequenceControls.setLayout(addNewSemanticSequenceControlsLayout)
#         self.addNewSemanticSequenceControls.setVisible(False)
        
        
        controlsLayout = QtGui.QVBoxLayout()
        controlsLayout.addWidget(self.loadedSequencesListTable)
        controlsLayout.addWidget(self.addNewSemanticSequenceControls)
        controlsLayout.addWidget(renderingControls)
        controlsLayout.addWidget(sequenceControls)
        
        sliderLayout = QtGui.QHBoxLayout()
        sliderLayout.addWidget(self.instancesShowIndicator)
        sliderLayout.addWidget(self.sequenceInstancesListTable)
        sliderLayout.addWidget(self.frameIdxSlider)
        sliderLayout.addWidget(self.frameIdxSpinBox)
        sliderLayout.setSpacing(1)
        
        self.frameLabelWidget = QtGui.QGroupBox("")
        frameHLayout = QtGui.QHBoxLayout()
        frameHLayout.addStretch()
        frameHLayout.addWidget(self.frameLabel)
        frameHLayout.addStretch()        
        
        frameVLayout = QtGui.QVBoxLayout()
        frameVLayout.addStretch()
        frameVLayout.addLayout(frameHLayout)
        frameVLayout.addWidget(self.frameInfo)
        frameVLayout.addStretch()
        frameVLayout.addLayout(sliderLayout)
        
        mainLayout.addLayout(controlsLayout)
        mainLayout.addLayout(frameVLayout)
        self.setLayout(mainLayout)

