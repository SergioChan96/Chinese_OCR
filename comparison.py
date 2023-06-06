import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

def initiate():
  global sift
  sift = cv2.SIFT_create()
  FLANN_INDEX_KDTREE = 1
  index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
  search_params = dict(checks=50)
  global flann
  flann = cv2.FlannBasedMatcher(index_params, search_params)

def identify(image):
  gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
  thresh, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
  kp, des = sift.detectAndCompute(binary, None)
  bestMatch = 0
  bestFile = ''
  for filename in os.scandir('TestData'):
    img = cv2.imread(filename.path, cv2.IMREAD_GRAYSCALE)
    thresh, binaryDict = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kpDict, desDict = sift.detectAndCompute(binaryDict, None)
    matches = flann.knnMatch(des, desDict, k=2)
    good_matches = []
    for m, n in matches:
      if m.distance < 0.7 * n.distance:
        good_matches.append(m)
        
    similarity = len(good_matches) / len(matches)

    if(similarity > bestMatch):
      bestMatch = similarity
      bestFile = filename
    # Compute the similarity score
    
  print('Similarity score:', bestMatch)
  return bestFile
  


if __name__ == '__main__':
  image = cv2.imread('pour2.png')
  initiate()
  filename = identify(image)
  print('Best match:', filename)
   
# build database of characters descriptors to compare with
# database could be just a txt file 


