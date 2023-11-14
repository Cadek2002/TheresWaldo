import torch
from torchvision.transforms import transforms
from PIL import Image
from pathlib import Path
import glob

model = torch.load(Path('model'))
model.eval()

trans = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

for img in glob.glob("data/validation/waldo/*"):
    image = Image.open(img)
    input = trans(image)
    input = input.view(1, 3, 64,64)
    output = torch.sigmoid(model(input))
    prediction = int(torch.max(output.data, 1)[0].numpy())
    print(img, ": ", output)
