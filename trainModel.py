import os
import glob
import shutil
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss
import torchvision
from torchvision import datasets, models, transforms
from tqdm import tqdm
from IPython.core.pylabtools import figsize
def inference(test_data):
  idx = torch.randint(1, len(test_data), (1,))
  sample = torch.unsqueeze(test_data[idx][0], dim=0).to(device)

  if torch.sigmoid(model(sample)) < 0.5:
    print("Prediction : waldo")
  else:
    print("Prediction : notwaldo")


  plt.imshow(test_data[idx][0].permute(1, 2, 0))

def build_dataset():
  data_dir = "data"

  #create training dir
  training_dir = os.path.join(data_dir,"training")
  if not os.path.isdir(training_dir):
    os.mkdir(training_dir)

  #create notwaldo in training
  notwaldo_training_dir = os.path.join(training_dir,"notwaldo")
  print(notwaldo_training_dir)
  if not os.path.isdir(notwaldo_training_dir):
    os.mkdir(notwaldo_training_dir)

  #create waldo in training
  waldo_training_dir = os.path.join(training_dir,"waldo")
  print(waldo_training_dir)
  if not os.path.isdir(waldo_training_dir):
    os.mkdir(waldo_training_dir)

  #create validation dir
  validation_dir = os.path.join(data_dir,"validation")
  if not os.path.isdir(validation_dir):
    os.mkdir(validation_dir)

  #create notwaldo in validation
  notwaldo_validation_dir = os.path.join(validation_dir,"notwaldo")
  if not os.path.isdir(notwaldo_validation_dir):
    os.mkdir(notwaldo_validation_dir)

  #create waldo in validation
  waldo_validation_dir = os.path.join(validation_dir,"waldo")
  if not os.path.isdir(waldo_validation_dir):
    os.mkdir(waldo_validation_dir)

  split_size = 0.80
  waldo_imgs_size = len(glob.glob("data/waldo/*"))
  notwaldo_imgs_size = len(glob.glob("data/notwaldo/*"))

  for i,img in enumerate(glob.glob("data/waldo/*")):
    if i < (waldo_imgs_size * split_size):
      shutil.move(img,waldo_training_dir)
    else:
      shutil.move(img,waldo_validation_dir)

  for i,img in enumerate(glob.glob("data/notwaldo/*")):
    if i < (notwaldo_imgs_size * split_size):
      shutil.move(img,notwaldo_training_dir)
    else:
      shutil.move(img,notwaldo_validation_dir)

  samples_notwaldo = [os.path.join(notwaldo_training_dir,np.random.choice(os.listdir(notwaldo_training_dir),1)[0]) for _ in range(8)]
  samples_waldo = [os.path.join(waldo_training_dir,np.random.choice(os.listdir(waldo_training_dir),1)[0]) for _ in range(8)]

  nrows = 4
  ncols = 4

  fig, ax = plt.subplots(nrows,ncols,figsize = (10,10))
  ax = ax.flatten()

  for i in range(nrows*ncols):
    if i < 8:
      pic = plt.imread(samples_notwaldo[i%8])
      ax[i].imshow(pic)
      ax[i].set_axis_off()
    else:
      pic = plt.imread(samples_waldo[i%8])
      ax[i].imshow(pic)
      ax[i].set_axis_off()

  plt.show()

traindir = "data/training"
testdir = "data/validation"

#Preprocess
train_transforms = transforms.Compose([transforms.ToTensor(), torchvision.transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])])
test_transforms = transforms.Compose([transforms.ToTensor(), torchvision.transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])])

#datasets
train_data = datasets.ImageFolder(traindir,transform=train_transforms)
test_data = datasets.ImageFolder(testdir,transform=test_transforms)

#dataloader
trainloader = torch.utils.data.DataLoader(train_data, shuffle = True, batch_size=16)
testloader = torch.utils.data.DataLoader(test_data, shuffle = True, batch_size=16)


def make_train_step(model, optimizer, loss_fn):
  def train_step(x,y):
    #make prediction
    yhat = model(x)
    #enter train mode
    model.train()
    #compute loss
    loss = loss_fn(yhat,y)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    #optimizer.cleargrads()

    return loss
  return train_step

device = "cuda" if torch.cuda.is_available() else "cpu"
print("training on %s" % device)
model = models.resnet18(pretrained=True)

#freeze all params
for params in model.parameters():
  params.requires_grad_ = False

#add a new final layer
nr_filters = model.fc.in_features  #number of input features of last layer
model.fc = nn.Linear(nr_filters, 1)

model = model.to(device)

#loss
loss_fn = BCEWithLogitsLoss() #binary cross entropy with sigmoid, so no need to use sigmoid in the model

#optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

#train step
train_step = make_train_step(model, optimizer, loss_fn)

losses = []
val_losses = []

epoch_train_losses = []
epoch_test_losses = []

n_epochs = 100
early_stopping_tolerance = 3
early_stopping_threshold = 0.003

for epoch in range(n_epochs):
  epoch_loss = 0
  for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):  # iterate over batches
    x_batch, y_batch = data
    x_batch = x_batch.to(device)  # move to training device
    y_batch = y_batch.unsqueeze(1).float()  # convert target to same nn output shape
    y_batch = y_batch.to(device)  # move to training device

    loss = train_step(x_batch, y_batch)
    epoch_loss += loss / len(trainloader)
    losses.append(loss)

  epoch_train_losses.append(epoch_loss)
  print('\nEpoch : {}, train loss : {}'.format(epoch + 1, epoch_loss))

  # validation doesnt requires gradient
  with torch.no_grad():
    cum_loss = 0
    for x_batch, y_batch in testloader:
      x_batch = x_batch.to(device)
      y_batch = y_batch.unsqueeze(1).float()  # convert target to same nn output shape
      y_batch = y_batch.to(device)

      # model to eval mode
      model.eval()

      yhat = model(x_batch)
      val_loss = loss_fn(yhat, y_batch)
      cum_loss += loss / len(testloader)
      val_losses.append(val_loss.item())

    epoch_test_losses.append(cum_loss)
    print('Epoch : {}, val loss : {}'.format(epoch + 1, cum_loss))

    best_loss = min(epoch_test_losses)

    # save best model
    if cum_loss <= best_loss:
      best_model_wts = model.state_dict()

    # early stopping
    early_stopping_counter = 0
    if cum_loss > best_loss:
      early_stopping_counter += 1

    if (early_stopping_counter == early_stopping_tolerance) or (best_loss <= early_stopping_threshold):
      print("\nTerminating: early stopping")
      break  # terminate training

# load best model
model.load_state_dict(best_model_wts)
torch.save(model, "TheresWaldoModelV4")