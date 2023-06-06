import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from PIL import Image, ImageFont, ImageDraw

def getCharacterImage(character):
  font_name = "resources/SourceHanSerif-VF.ttf.ttc"
  font_size = 50 # px
  # Create Font using PIL
  font = ImageFont.truetype(font_name, font_size)
  img = Image.Image()._new(font.getmask(character))
  return np.asarray(img)

def initiate():
  global sift
  sift = cv2.SIFT_create()
  FLANN_INDEX_KDTREE = 1
  index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
  search_params = dict(checks=50)
  global flann
  flann = cv2.FlannBasedMatcher(index_params, search_params)
  with open('resources/dictionary.txt', encoding="utf8") as f:
    global json_data
    string = f.read() 
    string = string.replace('\n', '')
    string = string.replace('}{', '},{')
    string = "[" + string + "]"
    json_data = json.loads(string)

def identify(image):
  gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
  thresh, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
  kp, des = sift.detectAndCompute(binary, None)
  bestMatch = 0
  bestChar = ''
  counter = 0
  for entry in json_data:
    counter += 1
    if counter == 2000:
      break
    img = getCharacterImage(entry['character'])
    kpDict, desDict = sift.detectAndCompute(img, None)
    if desDict is None or len(desDict) < 2:
      print('no feature matching for', entry['character'])
      continue
    matches = flann.knnMatch(des, desDict, k=2)
    good_matches = []
    for m, n in matches:
      if m.distance < 0.7 * n.distance:
        good_matches.append(m)
        
    similarity = len(good_matches) / len(matches)

    if(similarity > bestMatch):
      bestMatch = similarity
      bestChar = entry['character']
    
  print('Similarity score:', bestMatch)
  return bestChar
  


if __name__ == '__main__':
  image = cv2.imread('testData/Make_Complete.png')
  initiate()
  filename = identify(image)
  print('Best match:', filename)


