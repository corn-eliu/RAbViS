
## Imports and defines
import numpy as np
import scipy as sp
from scipy import special
import cv2
import sys
from scipy import ndimage

dataFolder = "data/"

def histEq(frame) :
    frame = np.array(frame, dtype=uint8)
    hstg = ndimage.measurements.histogram(frame, 0, 255, 256)
    csumhstg = np.cumsum(hstg)
    normcsum = cv2.normalize(csumhstg, None, 0, 255, cv2.NORM_MINMAX)
    eqFrame = np.zeros_like(frame)
    eqFrame = np.reshape(normcsum[frame], frame.shape)
    return eqFrame

## returns a diagonal kernel based on given binomial coefficients
def diagkernel(c) :
    k = np.eye(len(c))
    k = k * c/np.sum(c)
    return k

##  compute l2 distance between given frames
def l2dist(f1, f2) : 
    img = f1 - f2
    result = np.linalg.norm(img)
#     img = img ** 2
#     result = np.sqrt(np.sum(img))
    return result

## compute euclidean distance assuming f is an array where each row is a flattened image (1xN array, N=W*H*Channels)
## euclidean distance defined as the length of the the displacement vector:
## len(q-p) = sqrt(len(q)^2+len(p)^2 - 2*dot(p, q)) where p and q are two images in vector format and 1xN size
def distEuc(f) :
    ## gives sum over squared intensity values for each image
    ff = np.sum(f*f, axis=1)
    ## first term is sum between each possible combination of frames
    ## second term is the the dot product between each frame as in the formula above
    d = np.sqrt(np.reshape(ff, [len(ff),1])+ff.T - 2*np.dot(f, f.T))
    return d

## Turn distances to probabilities
def dist2prob(dM, sigmaMult, normalize, verbose=False) :
    sigma = sigmaMult*np.mean(dM[np.nonzero(dM)])
    if verbose :
        print 'sigma', sigma
    pM = np.exp((-dM)/sigma)
## normalize probabilities row-wise
    if normalize :
        normTerm = np.sum(pM, axis=1)
        normTerm = cv2.repeat(normTerm, 1, dM.shape[1])
        pM = pM / normTerm
    return pM

## Turn distances with extra range constraint to probabilities
def rangedist2prob(dM, sigmaMult, rangeDist, normalize, verbose=False) :
    sigma = sigmaMult*np.mean(dM[np.nonzero(dM)])
    if verbose :
        print 'sigma', sigma
    pM = np.exp(-(dM/sigma+ rangeDist))
## normalize probabilities row-wise
    if normalize :
        normTerm = np.sum(pM, axis=1)
        normTerm = cv2.repeat(normTerm, 1, dM.shape[1])
        pM = pM / normTerm
    return pM

## get a random frame based on given probabilities
def randFrame(probs) :
    indices = np.argsort(probs)[::-1] # get descending sort indices
    sortedProbs = probs[indices] # sort probs in descending order
#     print 'sortedProbs', sortedProbs
#     print np.int(probs.shape[0]/10)
    sortedProbs = sortedProbs[0:5]#np.int(probs.shape[0]/10)] # get highest 10%
#     sortedProbs = sortedProbs[0]f
#     print 'sortedProbs', sortedProbs
    sortedProbs = sortedProbs/np.sum(sortedProbs) # normalize
#     print 'sortedProbs', sortedProbs
    prob = np.random.rand(1)
#     print 'prob', prob
    csum = np.cumsum(sortedProbs)
    j = 0
    while csum[j] < prob :
        j = j+1
#     print 'final j', j
    return indices[j]

def doAlphaBlend(pairs, weights, movie) :
    blended = np.zeros(np.hstack((movie.shape[0:-1], len(pairs))), dtype=np.uint8)
    for pair, w, idx in zip(pairs, weights, xrange(0, len(pairs))) :
        if pair[0] != pair[1] :
            blended[:, :, :, idx] = movie[:, :, :, pair[0]]*w + movie[:, :, :, pair[1]]*(1.0-w)
    return blended

## compute distance between each pair of images
def computeDistanceMatrix(movie, savedDistMat) :
    try :
        ## load distanceMatrix from file
        distanceMatrix = np.load(savedDistMat)
        print "loaded distance matrix from ", savedDistMat
    except IOError :
        print "computing distance matrix and saving to ", savedDistMat
        distanceMatrix = np.zeros([movie.shape[3], movie.shape[3]])
        distanceMatrix = distEuc(np.reshape(movie/255.0, [np.prod(movie.shape[0:-1]), movie.shape[-1]]).T)
        # for i in range(0, movie.shape[3]) :
        #     for j in range(i+1, movie.shape[3]) :
        #         distanceMatrix[j, i] = distanceMatrix[i, j] = l2dist(movie[:, :, :, i], movie[:, :, :, j])
        # #         print distanceMatrix[j, i],
        #     print (movie.shape[3]-i),
            
        ## save file
        np.save(savedDistMat, distanceMatrix)
    
    return distanceMatrix

## Preserve dynamics: convolve wih binomial kernel
def filterDistanceMatrix(distanceMatrix, numFilterFrames, isRepetitive) :
    # numFilterFrames = 4 ## actual total size of filter is numFilterFrames*2 +1
    # isRepetitive = False ## see if this can be chosen automatically

    if isRepetitive :
        kernel = np.eye(numFilterFrames*2+1)
    else :
        coeff = special.binom(numFilterFrames*2, range(0, numFilterFrames*2 +1)); print coeff
        kernel = diagkernel(coeff)
    # distanceMatrixFilt = cv2.filter2D(distanceMatrix, -1, kernel)
    distanceMatrixFilt = ndimage.filters.convolve(distanceMatrix, kernel, mode='constant')
    distanceMatrixFilt = distanceMatrixFilt[numFilterFrames:-numFilterFrames,numFilterFrames:-numFilterFrames]
    
    return distanceMatrixFilt

## Avoid dead ends: estimate future costs
def estimateFutureCost(alpha, p, distanceMatrixFilt) :
    # alpha = 0.999
    # p = 2.0
    
    distMatFilt = distanceMatrixFilt[1:distanceMatrixFilt.shape[1], 0:-1]
    distMat = distMatFilt ** p
    
    last = np.copy(distMat)
    current = np.zeros(distMat.shape)
    
    ## while distance between last and current is larger than threshold
    iterations = 0 
    while np.linalg.norm(last - current) > 0.1 : 
        for i in range(distMat.shape[0]-1, -1, -1) :
            m = np.min(distMat, axis=1)
            distMat[i, :] = (distMatFilt[i, :] ** p) + alpha*m
            
        last = np.copy(current)
        current = np.copy(distMat)
        
        sys.stdout.write('\r' + "Iteration " + np.string_(iterations) + "; distance " + np.string_(np.linalg.norm(last - current)))
        sys.stdout.flush()
        
        iterations += 1
    
    print
    print 'finished in', iterations, 'iterations'
    
    return distMat

def getProbabilities(distMat, sigmaMult, rangeDist, normalizeRows, verbose=False) :
    ## compute probabilities from distanceMatrix and the cumulative probabilities
    if rangeDist == None :
        probabilities = dist2prob(distMat, sigmaMult, normalizeRows, verbose)
    else :
        probabilities = rangedist2prob(distMat, sigmaMult, rangeDist, normalizeRows, verbose)
    # since the probabilities are normalized on each row, the right most column will be all ones
    cumProb = np.cumsum(probabilities, axis=1)
    if verbose :
        print probabilities.shape, cumProb.shape
    
    return probabilities, cumProb

## Find a sequence of frames based on final probabilities
def getFinalFrames(cumProb, totalFrames, correction, startFrame, loopTexture, verbose) :
    # totalFrames = 1000
    finalFrames = []
    
    ## prob matrix is shrunk so row indices don't match frame numbers anymore unless corrected
    # correction = numFilterFrames+1
    
    
    currentFrame = startFrame#np.ceil(np.random.rand()*(cumProb.shape[0]-1))
    # currentFrame = 400
    if verbose :
        print 'starting at frame', currentFrame+correction
    finalFrames.append(currentFrame)
    for i in range(1, totalFrames) :
    #     currentFrame = randFrame(probabilities[currentFrame, :])
        finalFrames.append(getNewFrame(finalFrames[-1], cumProb))
        if verbose :
            print 'frame', i, 'of', totalFrames, 'taken from frame', finalFrames[-1]+correction #, prob
    
    if loopTexture and len(finalFrames) > 1 :
        ## add more random frames if necessary
        while finalFrames[-1] > finalFrames[0] :
            finalFrames.append(getNewFrame(finalFrames[-1], cumProb))
            if verbose :
                print 'additional frame', len(finalFrames)-totalFrames, 'taken from frame', finalFrames[-1]+correction#, prob
        ## add the remaining needed frames
        if finalFrames[-1] != finalFrames[0] :
            finalFrames = np.concatenate((finalFrames, list(np.arange(finalFrames[-1]+1, finalFrames[0]))))
            
            if verbose :
                print 'total additional frames', len(finalFrames[totalFrames:len(finalFrames)])
                print finalFrames[totalFrames:len(finalFrames)]
            
    
    finalFrames = np.array(finalFrames)
    finalFrames = finalFrames+correction
    return finalFrames

def getNewFrame(currentFrame, cumProb, minJumpDist = 10) :
    newFrame = np.copy(currentFrame)
    while np.abs(newFrame-currentFrame) < minJumpDist and newFrame-currentFrame != 1 :
        prob = np.random.rand(1)
        newFrame = np.round(np.sum(cumProb[currentFrame, :] < prob))
#         print "tralal", newFrame, currentFrame, np.abs(newFrame-currentFrame) < 10, newFrame-currentFrame != 1
    return newFrame

## render the frames in finalFrames
def renderFinalFrames(movie, finalFrames, numInterpolationFrames) :
    finalMovie = []
    finalMovie.append(np.array(np.copy(movie[:, :, :, finalFrames[0]]), dtype=np.uint8))
    finalJumps = []
    jump = 0
    cycleLengh = 1
    f = 1
    
    while f < len(finalFrames) :
        finalMovie.append(np.array(np.copy(movie[:, :, :, finalFrames[f]]), dtype=np.uint8))
        
        ## if it's a jump then do some sort of frame interpolation
        if finalFrames[f] == finalFrames[f-1]+1 :
            cycleLengh +=1
        else :
            cycleLengh = 1
            jump += 1
#             print "jump", jump, "from frame", finalFrames[f-1], "to frame", finalFrames[f]
            finalJumps.append(f)
            
            if numInterpolationFrames < 1 :
                f += 1
                continue
            
            ## find pairs of frames to interpolate between
            toInterpolate = np.zeros([numInterpolationFrames*2, 2])
            toInterpolate[:, 0] = np.arange(finalFrames[f-1]-numInterpolationFrames+1,finalFrames[f-1]+numInterpolationFrames+1)
            toInterpolate[:, 1] = np.arange(finalFrames[f]-numInterpolationFrames,finalFrames[f]+numInterpolationFrames)
            
            ## correct for cases where we're at the beginning of the movie or at the end
            if f < numInterpolationFrames :
                toIgnore = numInterpolationFrames-f
                toInterpolate[0:toIgnore, :] = 0
            elif f > len(finalFrames)-numInterpolationFrames :
                toIgnore = numInterpolationFrames+f-len(finalFrames)
                toInterpolate[-toIgnore:numInterpolationFrames*2, :] = 0
                
            ## do alpha blending 
            blendWeights = np.arange(1.0-1.0/(numInterpolationFrames*2.0), 0.0, -1.0/((numInterpolationFrames+1)*2.0))
            blended = doAlphaBlend(np.array(toInterpolate, dtype=np.int), blendWeights, movie)
            frameIndices = xrange(f-numInterpolationFrames, f+numInterpolationFrames)
            # put blended frames into finalMovie
            for idx, b in zip(frameIndices, xrange(0, len(blended))) :
                if idx >= 0 and idx < len(finalMovie) :
                    finalMovie[idx] = np.array(np.copy(blended[:, :, :, b]), dtype=np.uint8)
                elif idx == len(finalMovie) and idx < len(finalFrames):
                    finalMovie.append(np.array(np.copy(blended[:, :, :, b]), dtype=np.uint8))
                    f += 1 ## jumping frame now as otherwise I would copy it back from finalFrames instead of having the interpolated version
        
        f += 1
    
    return finalMovie, finalJumps

## save video
def saveMovieToAvi(finalMovie, outputData, videoName) :
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'DIB ')
    # videoName = "_ab.avi" if doInterpolation else "_no_ab.avi"
    out = cv2.VideoWriter(outputData+videoName,fourcc, 30.0, (movie.shape[1], movie.shape[0]))
    
    for f in xrange(0, finalMovie.shape[-1]) :
        frame = cv2.cvtColor(np.array(finalMovie[:, :, :, f]), cv2.COLOR_RGB2BGR)
        out.write(frame)
    
    # Release everything if job is finished\
    out.release()
