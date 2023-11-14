import concurrent.futures
import math
import threading
from pathlib import Path
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import transforms
from tqdm import tqdm

model = torch.load(Path('TheresWaldoModelV4'))
model.eval()
kernel_size = (64, 64)
stride = 8

trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def runModelbatch(windowBatch, batchNumber, model, resultArray):
    for j, y in enumerate(windowBatch):
        resultArray[batchNumber][j] = (float(torch.sigmoid(model(y.view(1, 3, 64, 64)))[0][0]))


imgPath = "waldo_dataset/original-images/12_test_2.jpg"
image: Image = Image.open(imgPath)

input_dims = image.size
print(input_dims)

width_difference = (kernel_size[0] - (input_dims[0] % kernel_size[0]))
height_difference = (kernel_size[1] - (input_dims[1] % kernel_size[1]))

width_padding: Tuple[int, int] = (math.ceil(width_difference / 2), math.floor(width_difference / 2))
height_padding: Tuple[int, int] = (math.ceil(height_difference / 2), math.floor(height_difference / 2))

print("Padding Image Axis: ", width_padding, ", ", height_padding)

image = np.pad(image, (height_padding, width_padding, (0, 0)))
output_image = np.copy(image)

input = trans(image)
print(image.shape)

iterate = input.unfold(1, kernel_size[1], stride).unfold(2, kernel_size[0], stride)

iterate = iterate.permute(1, 2, 0, 3, 4)
imageShape = iterate.shape
print(imageShape)

predMatrix = numpy.empty((iterate.shape[0], iterate.shape[1]))

with tqdm(total=len(iterate)) as pbar:
    with concurrent.futures.ThreadPoolExecutor() as threadpool:
        futures = {threadpool.submit(runModelbatch, x, i, model, predMatrix) for i, x in enumerate(iterate)}
        for future in concurrent.futures.as_completed(futures):
            pbar.update(1)

# Deconvolve Model Predictions into array with size of padded image
output_image = np.pad(cv2.resize(predMatrix, (0, 0), fx=stride * 1, fy=stride * 1), (height_padding, width_padding))

# show side-by-side of image->prediction mask
f, axarr = plt.subplots(1, 2)
axarr[0].imshow(image)
axarr[1].imshow(output_image)

plt.show()
