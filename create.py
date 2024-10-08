import cv2 as cv
import HandTrackingModule as hd
import pandas as pd
from PIL import Image  
from labels import labels
import numpy as np  
import time

cap = cv.VideoCapture(0)
detector = hd.handDetector()
data = pd.read_csv("Training_set.csv", delimiter = ",")
output = []

print(len(data))
for i in range(len(data)):
  filename, label = data.loc[i, "filename"], data.loc[i, "label"]
  img = cv.imread(f"train/{filename}") 
  img = cv.resize(img, (480, 480))
  img2 = img.copy()
  img = detector.findHands(img)
  try:
    positions = detector.findPosition(img)
    # print(len(positions), label)
    if len(positions) == 42:
      # print("SALOM")
      positions.append(labels[label])
      # print(positions)
      output.append(positions)
      # cv.imshow("Image", img)
      # time.sleep(1)
  except Exception as ex:
    print(f'An Exception Occurred: {ex}')
  
  k = cv.waitKey(1)
  if k == 27:
    break

  img2 = cv.flip(img2, 1)
  img2 = detector.findHands(img2)
  try:
    positions = detector.findPosition(img2)
    # print(len(positions), label)
    if len(positions) == 42:
      positions.append(labels[label])
      # print(positions)
      output.append(positions)
      # cv.imshow("Image", img2)
      # time.sleep(1)
  except Exception as ex:
    print(f'An Exception Occurred: {ex}')
  
  

  if i % 100 == 0:
    print(f"{i} - epoch")
  k = cv.waitKey(1)
  if k == 27:
    break


res_data = {}
for i in range(0, len(output[0]) - 1, 2):
  res_data[f"hand{(i + 2) // 2}_x"] = [output[j][i] for j in range(len(output))]
  res_data[f"hand{(i + 2) // 2}_y"] = [output[j][i + 1] for j in range(len(output))]

res_data["labels"] = [output[j][-1] for j in range(len(output))]

training_data = pd.DataFrame(res_data)
training_data.to_csv("training_data.csv", index = False)