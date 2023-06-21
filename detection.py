import numpy as np
import math
import cv2
global THRESHHOLD_BLACK
THRESHHOLD_BLACK = 10
global THRESHHOLD_WHITE
THRESHHOLD_WHITE = 245
def find_text_regions(image):
  # Apply the threshold
  _, binary = apply_threshold(gray)

  # Apply opening to reduce noise
  binary = apply_opening(binary)

  # Find the contours in the binary image
  # Search for big white space and cut surrounding of
  contours = find_contours(binary)

  # Filter the contours based on their size
  text_regions = [cnt for cnt in contours if is_large_enough(cnt)]

  return text_regions

def preprocess(img):
  height, width = img.shape
  new_img = img.copy()
  # apply filter
  for i in range(1, height-1):
      for j in range(1, width-1):
          new_img[i,j] = (img[i-1,j-1] + img[i-1,j] + img[i-1,j+1] +
                          img[i,j-1]   + img[i,j]   + img[i,j+1] +
                          img[i+1,j-1] + img[i+1,j] + img[i+1,j+1]) // 9

  return new_img

def process(self, imageIn, imageOut, attrOut, mask, previewMode):
    # The image will be affected so it's generated a new instance
    imageIn = imageIn.copy()

    maxWhiteSpace = self.getAttribute("maxWhiteSpace")
    maxFontLineWidth = self.getAttribute("maxFontLineWidth")
    minTextWidth = self.getAttribute("minTextWidth")
    grayScaleThreshold = self.getAttribute("grayScaleThreshold")

    Marvin.thresholding(imageIn, imageIn, grayScaleThreshold)

    segments = [[] for _ in range(imageIn.getHeight())]

    # map of already processed pixels
    processed = MarvinJSUtils.createMatrix2D(imageIn.getWidth(), imageIn.getHeight(), False)

    patternStartX = -1
    patternLength = 0
    whitePixels = 0
    blackPixels = 0

    for y in range(imageIn.getHeight()):
        for x in range(imageIn.getWidth()):
            if not processed[x][y]:
                color = imageIn.getIntColor(x, y)

                if color == 0xFFFFFFFF and patternStartX != -1:
                    whitePixels += 1
                    blackPixels = 0

                if color == 0xFF000000:
                    blackPixels += 1

                    if patternStartX == -1:
                        patternStartX = x

                    whitePixels = 0

                # check white and black pattern maximum lenghts
                if whitePixels > maxWhiteSpace or blackPixels > maxFontLineWidth or x == imageIn.getWidth() - 1:
                    if patternLength >= minTextWidth:
                        segments[y].append([patternStartX, y, patternStartX + patternLength, y])

                    whitePixels = 0
                    blackPixels = 0
                    patternLength = 0
                    patternStartX = -1

                if patternStartX != -1:
                    patternLength += 1

                processed[x][y] = True

    # Group line patterns intersecting in x coordinate and too near in y coordinate.
    for y in range(imageIn.getHeight() - 2):
        listY = segments[y]
        for w in range(y + 1, y + 3):
            listW = segments[w]
            for i in range(len(listY)):
                sA = listY[i]
                for j in range(len(listW)):
                    sB = listW[j]
                    # horizontal intersection
                    if (sA[0] <= sB[0] and sA[2] >= sB[2]) or (sA[0] >= sB[0] and sA[0] <= sB[2]) or (sA[2] >= sB[0] and sA[2] <= sB[2]):
                        sA[0] = min(sA[0], sB[0])
                        sA[2] = max(sA[2], sB[2])
                        sA[3] = sB[3]

                        listY.pop(i)
                        i -= 1

                        listW.pop(j)
                        listW.append(sA)

                        break

    # Convert the result to a List<> of MarvinSegment objects.
    marvinSegments = []
    for y in range(imageIn.getHeight()):
        for seg in segments[y]:
            marvinSegments.append(MarvinSegment(seg[0], seg[1], seg[2], seg[3]))

    attrOut.set("matches", marvinSegments)


if __name__ == '__main__':
  image = cv2.imread('newspaper.jpg', cv2.IMREAD_GRAYSCALE)
  image = preprocess(image)
  text_regions = find_text_regions(image)
