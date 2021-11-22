#! Caio Augusto Alves Nolasco - RA:195181
#! Projeto 5 - MC920 - Introdução ao Processamento de Imagem Digital

import numpy as np
import cv2
from matplotlib import pyplot as plt

imagePathA = input("Enter image A path: ")
imagePathB = input("Enter image B path: ")
outputPath = input("Enter image output path: ") #Read paths of the two images to be stitched and the path for the output.
featAlgorithm = input("Choose feature detection algorithm ('sift', 'surf', 'orb' or 'brief'): ") #Ask the user for the desired feature detection algorithm.
ratio = float(input("Enter ratio of selected matches: ")) #Read from user the desired ratio of matches to be selected.

def feature_descriptors (image, name):
    imgGray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) #Convert image to grayscale from RGB.  

    if(featAlgorithm == "sift"):
        featureDetect = cv2.xfeatures2d.SIFT_create()
    elif(featAlgorithm == "surf"):
        featureDetect = cv2.xfeatures2d.SURF_create()
    elif(featAlgorithm == "orb"):
        featureDetect = cv2.ORB_create()
    #Create feature detector according to given paramenter.

    elif(featAlgorithm == "brief"):
        starFeat = cv2.xfeatures2d.StarDetector_create()
        brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
        #BRIEF algorithm only detects descriptors. For feature detection, STAR algorithm is used.

        kp = starFeat.detect(image, None)
        kp, des = brief.compute(image, kp)

    if(featAlgorithm != "brief"):
        #If algorithm is not BRIEF, keypoints and descriptors can be detected at the same time.
        kp, des = featureDetect.detectAndCompute(imgGray,None)

    drawnKeypoints = cv2.drawKeypoints(image, kp, None)
    outputPathKeypoints = outputPath + featAlgorithm + name +"keypointsDrawn.jpg"
    cv2.imwrite(outputPathKeypoints, drawnKeypoints)
    #Save image with keypoint drawn.

    return kp, des


def feature_matcher (imgA, imgB, kp1, kp2, des1, des2):
    if featAlgorithm == "sift" or featAlgorithm == "surf":
        match = cv2.BFMatcher(cv2.NORM_L2)
        #If algorithm is SIFT or SURF, use euclidian distance to compare distance between descriptors.
    elif featAlgorithm == "orb" or featAlgorithm == "brief":
        #If algorithm is SIFT or SURF, use hamming distance to compare distance between descriptors.
        match = cv2.BFMatcher(cv2.NORM_HAMMING)

    matches = match.knnMatch(des1,des2,k=2)
    #Find matches between descriptors using K nearest neighbor matcher.

    matchedRatios = []

    for m,n in matches:
        if m.distance < ratio*n.distance:
            matchedRatios.append(m)
            #Filter matches according to ratio given.

    drawnMatches = cv2.drawMatches(imgA,kp1,imgB,kp2,matchedRatios,None,flags=2)
    outputPathMatches = outputPath + featAlgorithm +"matches.jpg"
    cv2.imwrite(outputPathMatches, drawnMatches)
    #Save image with matches between descriptors found.

    return matchedRatios

def homography_matrix (matches, kp1, kp2):
    if len(matches) >= 4:
        src = np.float32([ kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
        dst = np.float32([ kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
        H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        #Compute homography matrix with the selected matches.
    else:
        raise AssertionError("Not enough matches found")

    print(H)
    return H

def warp_and_stitch (imgA, imgB, H):
    panaromic = cv2.warpPerspective(imgA,H,(imgB.shape[1] + imgA.shape[1], imgB.shape[0]))
    #Align images using homography matrix and warpPerspectives().

    panaromic[0:imgB.shape[0], 0:imgB.shape[1]] = imgB
    #Stitch both images.

    outputPathPanoramic = outputPath + featAlgorithm + "finalPanoramic.jpg"
    cv2.imwrite(outputPathPanoramic, panaromic)
    #Save final panoramic image.

if __name__ == '__main__':
    imgA = cv2.imread(imagePathA)
    imgB = cv2.imread(imagePathB)

    keypointsA, descriptorsA = feature_descriptors(imgA, "A")
    keypointsB, descriptorsB = feature_descriptors(imgB, "B")

    matches = feature_matcher(imgA, imgB, keypointsA, keypointsB, descriptorsA, descriptorsB)

    H = homography_matrix(matches, keypointsA, keypointsB)

    warp_and_stitch(imgA, imgB, H)

