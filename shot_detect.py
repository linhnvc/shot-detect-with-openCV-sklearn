import imp
import numpy as np
import cv2 as cv
import sys
import os.path
import numpy as np  
from sklearn.svm import SVC  
from sklearn.preprocessing import normalize
import pickle


svclassifier = pickle.load(open('./myModel94.sav', 'rb'))


def devide(x,y):
    if y == 0 and x != 0:
        return float(x) / 10
    elif y == 1 and x < 20:
        return float(x) / 10
    elif y == 0 and x == 0:
        return 0
    else:
        return round(x/y,4)

def compareKeyPoint(frame_1, frame_2):
    try: 
        img1 = cv.cvtColor(frame_1, cv.COLOR_BGR2GRAY)
        img2 = cv.cvtColor(frame_2, cv.COLOR_BGR2GRAY)
        img1 = cv.resize(img1,(640,360), interpolation = cv.INTER_CUBIC)
        img2 = cv.resize(img2,(640,360), interpolation = cv.INTER_CUBIC)
        
        
        # Initiate SIFT detector
        sift = cv.xfeatures2d.SIFT_create()
        # find the keypoints and descriptors with SIFT
        keyPointsFrame1, descriptorsFrame1 = sift.detectAndCompute(img1,None)
        keyPointsFrame2, descriptorsFrame2 = sift.detectAndCompute(img2,None)
        
        
        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)   # or pass empty dictionary
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(descriptorsFrame1, descriptorsFrame2, k=2)
        count = 0
        for i,(m,n) in enumerate(matches):
                if m.distance < 0.75 * n.distance:
                    count = count + 1
    except:
        return 0, 0 , 0 
    else:
        return count, len(keyPointsFrame1), len(keyPointsFrame2)
    
def shotDetect(videoPath):
    cap = cv.VideoCapture(videoPath)
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    prevRet, prevFrame = cap.read()
    frameShape = prevFrame.shape
    shotNumber = 0
    prevNumber = [0, 0, 0]
    while True:
        numContinuesZero = 0
        isShotStart = True
        lastOneDetect = 0
        keyFrame = 0
        maxNumKeyPoint = 0

        out = cv.VideoWriter(sys.argv[2] + 'shot' + str(shotNumber + 1) + '.mp4',fourcc, 20.0, (frameShape[1],frameShape[0]))
        shotLog = open(sys.argv[2] + 'shot' + str(shotNumber +1 ) + '.txt', 'w+')
        while(cap.isOpened()):
            currRet, currFrame = cap.read()
            if currRet == False:
                sys.exit(0)

            totalMatches, numKp1, numKp2 = compareKeyPoint(prevFrame, currFrame)
            subMatches = abs(prevNumber[0] - totalMatches)
            subNumKeyPoint = abs(prevNumber[2] - numKp2)
            gradientMatches = devide(subMatches, totalMatches)
            gradientNumKeypoint = devide(subNumKeyPoint, numKp2)
            
            x_predict = np.asarray([totalMatches, gradientMatches, numKp2, gradientNumKeypoint])
            isNextShot = svclassifier.predict(x_predict.reshape(1,-1))
            
            shotLog.write(str(totalMatches)  + '\t' + str(subMatches) + '\t' + str(gradientMatches) + '\t' + str(numKp2) +'\t' + str(subNumKeyPoint) + '\t' + str(gradientNumKeypoint) + '\t' + str(isNextShot[0]) + '\n')
            
            print(str(totalMatches)  + '\t' + str(subMatches) + '\t' + str(gradientMatches) + '\t' + str(numKp2) +'\t' + str(subNumKeyPoint) + '\t' + str(gradientNumKeypoint) + '\t' + str(isNextShot[0]))
            prevNumber = [totalMatches, numKp1, numKp2]

            if isShotStart == True and totalMatches < 5:
                out.write(currFrame)
            elif totalMatches == 0 and subMatches == 0 and numKp2 == 0 and subNumKeyPoint == 0:
                numContinuesZero += 1
                if isShotStart == True:
                    out.write(currFrame)
                elif numContinuesZero > 10 and isShotStart == False:
                    out.release()
                    shotLog.close()
                    prevFrame = currFrame
                    # cv.imwrite(sys.argv[2] + 'shot' + str(shotNumber) + '.png',keyFrame)
                    shotNumber += 1
                    break
                else:
                    out.write(currFrame)
            elif int(isNextShot[0]) == 1:
                if lastOneDetect < 6:
                    out.write(currFrame)
                else:
                    out.release()
                    shotLog.close()
                    prevFrame = currFrame
                    # cv.imwrite(sys.argv[2] + 'shot' + str(shotNumber) + '.png',keyFrame)
                    shotNumber += 1
                    break
                lastOneDetect = 0
            else:
                numContinuesZero = 0
                isShotStart = False
                out.write(currFrame)
                lastOneDetect += 1
            prevFrame = currFrame

            if numKp2 > maxNumKeyPoint:
                keyFrame = currFrame
                maxNumKeyPoint = numKp2 

        else:
            sys.exit(0)
    
shotDetect(sys.argv[1])
