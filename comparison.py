import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import time
import multiprocessing
from PIL import Image, ImageFont
from skimage.transform import resize

def getCharacterImage(character):
  img = Image.Image()._new(font.getmask(character))
  img = np.asarray(img)
  return (img / 255).astype(np.uint8)

def loadFont(font_size):
  global font
  font_name = "resources/SourceHanSerif-VF.ttf.ttc"
  font = ImageFont.truetype(font_name, font_size)

def initiate():
  # global sift
  # sift = cv2.SIFT_create()
  # FLANN_INDEX_KDTREE = 1
  # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
  # search_params = dict(checks=50)
  # global flann
  # flann = cv2.FlannBasedMatcher(index_params, search_params)
  with open('resources/dictionary.txt', encoding="utf8") as f:
    global json_data
    string = f.read() 
    string = string.replace('\n', '')
    string = string.replace('}{', '},{')
    string = "[" + string + "]"
    json_data = json.loads(string)

def identify(image):
  bestChar = ''
  counter = 0
  best_xcorr = 0
  for entry in json_data:
    counter += 1
    if counter == 10000:
      break
    template = getCharacterImage(entry['character'])
    resized = cv2.resize(image, (template.shape[1], template.shape[0]))
    xcorr = cross_correlation(template, resized)
    if xcorr > best_xcorr:
      bestChar = entry['character']
      best_xcorr = xcorr
  print('Similarity score:', best_xcorr)
  return bestChar

# def identify(image):
#   gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#   thresh, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#   kp, des = sift.detectAndCompute(binary, None)
#   bestMatch = 0
#   bestChar = ''
#   for entry in os.scandir('database/'):
#     array = np.load(entry.path, allow_pickle=True)
#     if array is None or len(array) < 2:
#       print('no feature matching for', entry.name.split('.')[0])
#       continue
#     matches = flann.knnMatch(des, array, k=2)
#     good_matches = []
#     for m, n in matches:
#       if m.distance < 0.7 * n.distance:
#         good_matches.append(m)
        
#     similarity = len(good_matches) / len(matches)

#     if(similarity > bestMatch):
#       bestMatch = similarity
#       bestChar = entry.name.split('.')[0]
    
#   print('Similarity score:', bestMatch)
#   return bestChar

def preprocessing(image):
  gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
  thresh, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
  binary = np.where(binary == 255, 0, 1)
  return binary.astype(np.uint8)
  
def cross_correlation(image1, image2):
  sum = 0
  std_div1 = np.std(image1)
  std_div2 = np.std(image2)
  mean1 = np.mean(image1)
  mean2 = np.mean(image2)
  return np.sum((image1 - mean1) * (image2 - mean2)) / (std_div1 * std_div2)

  
if __name__ == '__main__':
<<<<<<< Updated upstream
  image = cv2.imread(os.path.join(os.getcwd(), 'TestData/Make_Complete.png'))
=======
>>>>>>> Stashed changes
  initiate()
  for images in os.scandir('testData/'):
    t0 = time.time()
    image = cv2.imread(images.path)
    image = preprocessing(image)
    loadFont(image.shape[0])
    filename = identify(image)
    print('Best match for', images.name, ':', filename)
    t1 = time.time()
    print('Time taken in seconds:', t1 - t0)


