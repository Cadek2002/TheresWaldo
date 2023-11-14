import numpy
import torch
from torchvision.transforms import transforms
from PIL import Image
from pathlib import Path
import glob
import numpy as np
import math
import matplotlib.pyplot as plt

input = plt.imread("waldo_dataset/original-images/12.jpg")
output = plt.imread("waldo_dataset/original-images/13.jpg")

f, axarr = plt.subplots(2,1)
axarr[0].imshow(input)
axarr[1].imshow(output)

plt.show()