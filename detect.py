import numpy as np
import cv2

# This file defines a wrapper for the cv2.SimpleBlobDetector method which enables the definition of functions performing
# detection using a defined set of parameters. If you want to use this code on you own dataset, you will probably have
# to define your own detection function based on the detect wrapper, and experiment with the parameters. Defining a new
# function for each dataset to be worked with is probably not a good way for things to work, so feel free to come up
# with a better way of doing things. Some detection functions that have been used in the work done with this code are
# included, and can be used as examples.

def detect(image,
           invert,
           minThreshold=0,
           maxThreshold=255,
           thresholdStep=1,
           minDistBetweenBlobs=0,
           filterByArea=False,
           minArea=0,
           maxArea=None,
           filterByCircularity=False,
           minCircularity=0.0,
           maxCircularity=None,
           filterByConvexity=False,
           minConvexity=0.0,
           maxConvexity=None,
           filterByInertia=False,
           minInertiaRatio=0.0,
           maxInertiaRatio=None
           ):
    """Wrapper for the cv2.SimpleBlobDetector method"""

    if invert:
        image = 255 - image

    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = minThreshold
    params.maxThreshold = maxThreshold
    params.thresholdStep = thresholdStep
    params.minDistBetweenBlobs = minDistBetweenBlobs
    params.filterByArea = filterByArea
    params.minArea = minArea
    params.maxArea = maxArea
    params.filterByCircularity = filterByCircularity
    params.minCircularity = minCircularity
    params.maxCircularity = maxCircularity
    params.filterByConvexity = filterByConvexity
    params.minConvexity = minConvexity
    params.maxConvexity = maxConvexity
    params.filterByInertia = filterByInertia
    params.minInertiaRatio = minInertiaRatio
    params.maxInertiaRatio = maxInertiaRatio

    # Set up the detector with the given parameters.
    detector = cv2.SimpleBlobDetector_create(params)
    # Do the detection
    keypoints = detector.detect(image)

    blobs = np.zeros((len(keypoints), 3))

    # This is to make the output work with other code
    for i, keypoint in enumerate(keypoints):
        blobs[i][0] = keypoint.pt[1]
        blobs[i][1] = keypoint.pt[0]
        blobs[i][2] = keypoint.size / 2

    return blobs

def droplets(image):

    blobs = detect(image,
                   invert=False,
                   minThreshold=50,
                   filterByArea=True,
                   minArea=200,
                   filterByCircularity=True,
                   minCircularity=0.85,
                   filterByInertia=True,
                   minInertiaRatio=0.8
                   )

    return blobs

def wiresWithoutDroplets(image):

    blobs = detect(image,
                   invert=True,
                   maxThreshold=130,
                   filterByArea=True,
                   minArea=20,
                   filterByCircularity=True,
                   minCircularity=0.7,
                   filterByConvexity=True,
                   minConvexity=0.9,
                   )

    return blobs

def tiled(image):

    blobs = detect(image,
                   invert=True,
                   maxThreshold=200,
                   filterByArea=True,
                   minArea=40,
                   filterByCircularity=True,
                   minCircularity=0.8,
                   filterByConvexity=True,
                   minConvexity=0.9,
                   )

    return blobs

def tiled_2(image):

    blobs = detect(image,
                   invert=True,
                   maxThreshold=200,
                   filterByArea=True,
                   minArea=200,
                   filterByCircularity=True,
                   minCircularity=0.85,
                   filterByConvexity=True,
                   minConvexity=0.9,
                   )

    return blobs

def random(image):

    blobs = detect(image,
                   invert=True,
                   maxThreshold=200,
                   filterByArea=True,
                   minArea=40,
                   maxArea=450,
                   filterByCircularity=True,
                   minCircularity=0.7
                   )

    return blobs