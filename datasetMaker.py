
import os
import glob
import shutil
import torch
from torchvision.transforms import transforms
import PIL.Image
import matplotlib.pyplot as plt
import numpy as np
import natsort as ns
from PIL import Image
import glob
import math
import matplotlib.pyplot as plt
from functools import reduce
import slidingWindow as sd


# Function to Find factors (Modified from https://stackoverflow.com/questions/6800193/what-is-the-most-efficient-way-of-finding-all-the-factors-of-a-number-in-python)
def factors(n):
    factors = sorted(list(set(reduce(list.__add__,
                                     ([i, n // i] for i in range(1, int(n ** 0.5) + 1) if n % i == 0)))))
    return (factors[len(factors) // 2-1], factors[math.ceil((len(factors) // 2))]) if len(factors) != 1 else (
    factors[0], factors[0])  # find median factors


kernel_size = (64, 64, 3)
stride = 12

trans = transforms.Compose([
    transforms.ToTensor(),
])

toImage = transforms.ToPILImage()

# Image Data
bounding_boxes = [((705, 510), (742, 562)), ((81, 521), (103, 556)), ((1386, 464), (1422, 501)),
                  ((1483, 278), (1506, 306)), ((1581, 593), (1607, 632)), ((1714, 407), (1753, 454)),
                  ((812, 951), (850, 999)), ((1163, 292), (1191, 322)), ((229, 739), (263, 772)),
                  ((716, 157), (733, 181)), ((458, 1530), (481, 1554)), ((843, 517), (873, 564)),
                  ((197, 1878), (235, 1920)), ((294, 428), (313, 455)), ((750, 248), (771, 280))]
plotDims = factors(len(bounding_boxes))
positivePath = "data/waldo"

f, axarr = plt.subplots(plotDims[0], plotDims[1])
for i, imgPath in enumerate(ns.natsorted(glob.glob("dataset_orig_images/*"))):
    print(imgPath)
    box = bounding_boxes[i]
    image = Image.open(imgPath)
    relevantSection = image.crop(((box[1][0]) - kernel_size[0], (box[1][1]) - kernel_size[1],
                                  (box[0][0]) + kernel_size[0], (box[0][1]) + kernel_size[1]))
    axarr[i % 3][i % 5].imshow(relevantSection)

    iterate = trans(relevantSection).unfold(1, kernel_size[1], stride).unfold(2, kernel_size[0], stride).permute(1, 2, 0, 3, 4)

    for j, tensor in enumerate(iterate):
        for k, window in enumerate(tensor):
            img = toImage(window)
            img.save(positivePath + f"/{i}_{j}.{k}.jpg")

plt.show()
