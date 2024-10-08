import cv2 as cv
import HandTrackingModule as hd
import pandas as pd
from PIL import Image  
from labels import labels
import torch
import numpy as np  
import time
from torch import nn, optim, cuda
from create_model import signNet

model = torch.load("model.pth")
device = "cuda" if cuda.is_available() else "cpu"
model.to(device)
# xy = np.loadtxt("training_data.csv", delimiter = ",", dtype = np.float32, skiprows = 1)
# x_data = xy[:, 0: -1]
# y_data = xy[:, -1].astype("int64")
# sugeno = Sugeno(l = 2)
# sugeno.train(x_data, y_data)

cap = cv.VideoCapture(0)
# cv.namedWindow("image", cv.WINDOW_NORMAL);
# cv.resizeWindow("image", 270, 200)
detector = hd.handDetector()
while True:
  sucess, img = cap.read()
  img = detector.findHands(img)
  try:
    positions = detector.findPosition(img)
    if len(positions) == 42:
      data = torch.tensor([positions]).to(device)
      output = model(data)
      pred = output.data.max(1, keepdim=True)[1][0][0].item()   # torch.max() funkisyasi har ikkala qiymat va indekslarni qaytaradi
      # pred = sugeno.predict_one(positions)
      if pred < 26:
        cv.putText(img, chr(pred + 65), (10, 100), cv.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 3)
  except Exception as ex:
    print(f'An Exception Occurred: {ex}')
  cv.imshow('image', img)
  k = cv.waitKey(1)
  if k == 27:
    break