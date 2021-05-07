"""
Note on installation to open the saved model
- CUDA 10.0
- Pytorch 1.2.0

How to install:
  https://varhowto.com/install-pytorch-cuda-10-0/#:~:text=Note%3A%20PyTorch%20only%20supports%20CUDA,cu100%2Ftorch_stable.html).

Pytorch Installation:
  https://pytorch.org/
"""
import os
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as tt
from torch.utils.data import random_split
from torchvision.utils import make_grid
import matplotlib
import matplotlib.pyplot as plt
from Minh.DIP.onecyclelr import OneCycleLR
%matplotlib inline

matplotlib.rcParams['figure.facecolor'] = '#ffffff'
  
#data_dir = "/home/cdsw/Minh/DIP/data/imagenette2-160"
#data_dir = "/home/cdsw/Minh/DIP/data/imagenette2-160-hori-flip"
#data_dir = "/home/cdsw/Minh/DIP/data/imagenette2-160-vert-flip"
#data_dir = "/home/cdsw/Minh/DIP/data/imagenette2-160-amf"
#data_dir = "/home/cdsw/Minh/DIP/data/imagenette2-160-butterworth-lpf"
#data_dir = "/home/cdsw/Minh/DIP/data/imagenette2-160-butterworth-hpf"
data_dir = "/home/cdsw/Minh/DIP/data/imagenette2-160-canny-ed"
#data_dir = "/home/cdsw/Minh/DIP/data/imagenette2-160-gaussian-lpf"
#data_dir = "/home/cdsw/Minh/DIP/data/imagenette2-160-gaussian-hpf"
#data_dir = "/home/cdsw/Minh/DIP/data/imagenette2-160-hist-equal"
#data_dir = "/home/cdsw/Minh/DIP/data/imagenette2-160-ideal-hpf"
#data_dir = "/home/cdsw/Minh/DIP/data/imagenette2-160-ideal-lpf"
#data_dir = "/home/cdsw/Minh/DIP/data/imagenette2-160-invert"
#data_dir = "/home/cdsw/Minh/DIP/data/imagenette2-160-rand-rotate"
#data_dir = "/home/cdsw/Minh/DIP/data/imagenette2-160-hori-shear"
#data_dir = "/home/cdsw/Minh/DIP/data/imagenette2-160-vert-shear"
#data_dir = "/home/cdsw/Minh/DIP/data/imagenette2-160-translate"

save_dir = "/home/cdsw/Minh/DIP/saved_models"

def check_dir():
  print(os.listdir(data_dir + '/train'))
  classes = os.listdir(data_dir + '/train')
  classes.remove('.DS_Store')
  print(classes)

#########################################
#DATA TRANSFORMATION + NORMALIZATION
# stats = means for 3 channels images. 
stats = ((0.485, 0.456, 0.406) , (0.229, 0.224, 0.225)) 
  
# resizes image to 64x64 before processing and tensor conversion. Note: no image augmentation, just normalization
train_tfms = tt.Compose([
  tt.Resize((64,64)),
  tt.ToTensor(),
  tt.Normalize(*stats, inplace=True)])
valid_tfms = tt.Compose([tt.Resize((64,64)), tt.ToTensor(), tt.Normalize(*stats)])

#########################################
#CREATE PYTORCH DATASET & DATALOADER
train_ds = ImageFolder(data_dir+'/train', train_tfms)
valid_ds = ImageFolder(data_dir+'/val', valid_tfms)
batch_size = 64

# data loaders
train_dl = DataLoader(train_ds, batch_size, shuffle = True, num_workers = 3, pin_memory = True)
valid_dl = DataLoader(valid_ds, batch_size*2, num_workers = 3, pin_memory = True)

#########################################
#HELPER IN VISUALIZATION
def denormalize(images, means, stds):
  means = torch.tensor(means).reshape(1, 3, 1, 1)
  stds = torch.tensor(stds).reshape(1, 3, 1, 1)
  return images * stds + means

def show_batch(dl):
  for images, labels in dl:
      fig, ax = plt.subplots(figsize=(12, 12))
      ax.set_xticks([]); ax.set_yticks([])
      denorm_images = denormalize(images, *stats)
      ax.imshow(make_grid(denorm_images[:64], nrow=8).permute(1, 2, 0).clamp(0,1))
      break

show_batch(train_dl)
show_batch(valid_dl)

#########################################
#SET UP MACHINE
def get_default_device():
  """Pick GPU if available, else CPU"""
  if torch.cuda.is_available():
      return torch.device('cuda')
  else:
      return torch.device('cpu')
    
def to_device(data, device):
  """Move tensor(s) to chosen device"""
  if isinstance(data, (list,tuple)):
      return [to_device(x, device) for x in data]
  return data.to(device, non_blocking=True)

class DeviceDataLoader():
  """Wrap a dataloader to move data to a device"""
  def __init__(self, dl, device):
    self.dl = dl
    self.device = device
        
  def __iter__(self):
    """Yield a batch of data after moving it to device"""
    for b in self.dl: 
      yield to_device(b, self.device)

  def __len__(self):
    """Number of batches"""
    return len(self.dl)

device = get_default_device()
print(device)

train_dl = DeviceDataLoader(train_dl, device)
valid_dl = DeviceDataLoader(valid_dl, device)

#########################################
#BUILD RESNET MODEL
class SimpleResidualBlock(nn.Module):
  def __init__(self):
      super().__init__()
      self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
      self.relu1 = nn.ReLU()
      self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
      self.relu2 = nn.ReLU()
        
  def forward(self, x):
      out = self.conv1(x)
      out = self.relu1(out)
      out = self.conv2(out)
      return self.relu2(out) + x # ReLU can be applied before or after adding the input
    
# create a simple residual module
simple_resnet = to_device(SimpleResidualBlock(), device)

# check the shape of the image
for images, labels in train_dl:
  out = simple_resnet(images)
  print(out.shape)
  break
    
del simple_resnet, images, labels
torch.cuda.empty_cache()

#########################################
#TRAINING PROCEDURE
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))
  
# build ResNet Training Architecture
class ImageClassificationBase(nn.Module):
  def training_step(self, batch):
    images, labels = batch 
    out = self(images)                  # Generate predictions
    loss = F.cross_entropy(out, labels) # Calculate loss
    return loss
    
  def validation_step(self, batch):
    images, labels = batch 
    out = self(images)                    # Generate predictions
    loss = F.cross_entropy(out, labels)   # Calculate loss
    acc = accuracy(out, labels)           # Calculate accuracy
    return {'val_loss': loss.detach(), 'val_acc': acc}
        
  def validation_epoch_end(self, outputs):
    batch_losses = [x['val_loss'] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
    batch_accs = [x['val_acc'] for x in outputs]
    epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
    return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
  def epoch_end(self, epoch, result):
    print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))
# convolution block
def conv_block(in_channels, out_channels, pool=False):
  layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
  if pool: layers.append(nn.MaxPool2d(2))
  return nn.Sequential(*layers)

# RESNET9
class ResNet9(ImageClassificationBase):
  def __init__(self, in_channels, num_classes):
    super().__init__()
        
    self.conv1 = conv_block(in_channels, 128)
    self.conv2 = conv_block(128, 256, pool=True)
    self.res1 = nn.Sequential(conv_block(256, 256), conv_block(256, 256))
        
    self.conv3 = conv_block(256, 512, pool=True)
    self.conv4 = conv_block(512, 1024, pool=True)
    self.res2 = nn.Sequential(conv_block(1024, 1024), conv_block(1024, 1024))
        
    self.classifier = nn.Sequential(nn.MaxPool2d(8), 
                                        nn.Flatten(), 
                                        nn.Dropout(0.2),
                                        nn.Linear(1024, num_classes))
        
  def forward(self, xb):
    out = self.conv1(xb)
    out = self.conv2(out)
    out = self.res1(out) + out
    out = self.conv3(out)
    out = self.conv4(out)
    out = self.res2(out) + out
    out = self.classifier(out)
    return out
  
model = to_device(ResNet9(3, 10), device)
## 3 channel inputs with 10 classes overall.
print(model)

#########################################
# TRAIN MODEL
@torch.no_grad()
def evaluate(model, val_loader):
  model.eval()
  outputs = [model.validation_step(batch) for batch in val_loader]
  return model.validation_epoch_end(outputs)

def get_lr(optimizer):
  for param_group in optimizer.param_groups:
    return param_group['lr']

##################
def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader, 
                  weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):
  
  torch.cuda.empty_cache()
  history = []
    
  # Set up cutom optimizer with weight decay
  optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay) # model, lr, momentum
  
  # Set up one-cycle learning rate scheduler
  sched = OneCycleLR(optimizer, num_steps=len(train_loader), lr_range=(0.1,1.0))
  
  for epoch in range(epochs):
    # Training Phase 
    model.train()
    train_losses = []
    lrs = []
    for batch in train_loader:
      loss = model.training_step(batch)
      train_losses.append(loss)
      loss.backward()
            
      # Gradient clipping
      if grad_clip: 
        nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            
      optimizer.step()
      optimizer.zero_grad()
            
      # Record & update learning rate
      lrs.append(get_lr(optimizer))
      sched.step()
        
    # Validation phase
    result = evaluate(model, val_loader)
    result['train_loss'] = torch.stack(train_losses).mean().item()
    result['lrs'] = lrs
    model.epoch_end(epoch, result)
    history.append(result)
    
  return history

# pre evaluation with random weigths
history = [evaluate(model, valid_dl)]
print(history)

epochs = 30
grad_clip = 0.1
#weight_decay = 1e-5
#max_lr = 0.001
weight_decay = 0.01 #0.01 + 0.01 max=> 69, 0.01 + 0.1: 74%
max_lr = 0.1 # 0.01 69%
opt_func = torch.optim.Adam

%%time
history += fit_one_cycle(epochs, max_lr, model, train_dl, valid_dl, 
                             grad_clip=grad_clip, 
                             weight_decay=weight_decay, 
                             opt_func=opt_func)

#########################################
# Accuracy and Loss Plot
def plot_accuracies(history):
  accuracies = [x['val_acc'] for x in history]
  plt.plot(accuracies, '-x')
  plt.xlabel('epoch')
  plt.ylabel('accuracy')
  plt.title('Accuracy vs. No. of epochs')

plot_accuracies(history)

def plot_losses(history):
  train_losses = [x.get('train_loss') for x in history]
  val_losses = [x['val_loss'] for x in history]
  plt.plot(train_losses, '-bx')
  plt.plot(val_losses, '-rx')
  plt.xlabel('epoch')
  plt.ylabel('loss')
  plt.legend(['Training', 'Validation'])
  plt.title('Loss vs. No. of epochs')
  
plot_losses(history)

def plot_lrs(history):
  lrs = np.concatenate([x.get('lrs', []) for x in history])
  plt.plot(lrs)
  plt.xlabel('Batch no.')
  plt.ylabel('Learning rate')
  plt.title('Learning Rate vs. Batch no.')

plot_lrs(history)

#torch.save(model.state_dict(), save_dir+'_original.pth')
#torch.save(model.state_dict(), save_dir+'_horizontal_flip.pth')
#torch.save(model.state_dict(), save_dir+'_vertical_flip.pth')
#torch.save(model.state_dict(), save_dir+'_rand_rotate.pth')
#torch.save(model.state_dict(), save_dir+'_hori_shear.pth')
#torch.save(model.state_dict(), save_dir+'_vert_shear.pth')
#torch.save(model.state_dict(), save_dir+'_translate.pth')
#torch.save(model.state_dict(), save_dir+'_invert.pth')
#torch.save(model.state_dict(), save_dir+'_ideal_hpf.pth')
#torch.save(model.state_dict(), save_dir+'_ideal_lpf.pth')
#torch.save(model.state_dict(), save_dir+'_gaussian_hpf.pth')
#torch.save(model.state_dict(), save_dir+'_gaussian_lpf.pth')
#torch.save(model.state_dict(), save_dir+'_butterworth_hpf.pth')
#torch.save(model.state_dict(), save_dir+'_butterworth_lpf.pth')
#torch.save(model.state_dict(), save_dir+'_amf.pth')
torch.save(model.state_dict(), save_dir+'_canny_ed.pth')


#########################################
# TEST WITH INDIVIDUAL IMAGES
def predict_image(img, model):
  # Convert to a batch of 1
  xb = to_device(img.unsqueeze(0), device)
  # Get predictions from model
  yb = model(xb)
  # Pick index with highest probability
  _, preds  = torch.max(yb, dim=1)
  # Retrieve the class label
  return train_ds.classes[preds[0].item()]

# test 1
img, label = valid_ds[0]
plt.imshow(img.permute(1, 2, 0).clamp(0, 1))
print('Label:', train_ds.classes[label], ', Predicted:', predict_image(img, model))

# test 2 
img, label = valid_ds[355]
plt.imshow(img.permute(1, 2, 0))
print('Label:', valid_ds.classes[label], ', Predicted:', predict_image(img, model))

# test 3
img, label = valid_ds[200]
plt.imshow(img.permute(1, 2, 0))
print('Label:', train_ds.classes[label], ', Predicted:', predict_image(img, model))

#########################################
#MAIN
def main():
  print("Running")
  check_dir()

if __name__ == "__main__":
  main()