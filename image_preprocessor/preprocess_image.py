from scipy import ndarray
import skimage.io
import skimage as sk
from skimage import transform
from skimage import util
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from Minh.DIP.augmentation.affine.flip import flip_horizontal, flip_vertical
from Minh.DIP.augmentation.affine.rotate import rotate
from Minh.DIP.augmentation.affine.shear import vertical_shear, horizontal_shear
from Minh.DIP.augmentation.affine.translate import translate
from Minh.DIP.augmentation.frequency.frequency_filter import freq_filter
from Minh.DIP.augmentation.intensity.amf import amf
from Minh.DIP.augmentation.intensity.hist_equalization import hist_equalization
from Minh.DIP.augmentation.intensity.invert import invert
from Minh.DIP.augmentation.edge_detection.canny_edge import canny


#################################
#image_path = "/home/cdsw/Minh/DIP/data/original/train/n03417042"
#save_path = "/home/cdsw/Minh/DIP/data/imagenette2-160-amf/train/n03417042"
image_path = "/home/cdsw/Minh/DIP/data/original/val/n03425413"
save_path = "/home/cdsw/Minh/DIP/data/imagenette2-160-amf/val/n03425413"

def preprocess_amf():
  print("Running")
  # loop on all files of the folder and build a list of files paths
  image_name_list = [f for f in os.listdir(image_path) if os.path.splitext(f)[-1] == '.JPEG']
  image_list = [os.path.join(image_path, f) for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))]

  for i in range(len(image_list)):
    image_to_transform = sk.io.imread(image_list[i])

    # concatenate name
    image_name = image_name_list[i].split('.',1)
    new_image_name = image_name[0]
    new_image_name = new_image_name + "_amf.JPEG"
    print(new_image_name)

    transformed_image = amf(image_to_transform)
    
    if(transformed_image is None):
      print("None")
      continue # did will not do anything and pass to the next image
      
    sk.io.imsave(save_path+'/'+new_image_name , transformed_image) 

#################################
#image_path = "/home/cdsw/Minh/DIP/data/original/train/n03888257"
#save_path = "/home/cdsw/Minh/DIP/data/imagenette2-160-butterworth-hpf/train/n03888257"
#image_path = "/home/cdsw/Minh/DIP/data/original/val/n03888257"
#save_path = "/home/cdsw/Minh/DIP/data/imagenette2-160-butterworth-hpf/val/n03888257"

def preprocess_butterworth_hpf():
  print("Running")
  # loop on all files of the folder and build a list of files paths
  image_name_list = [f for f in os.listdir(image_path) if os.path.splitext(f)[-1] == '.JPEG']
  image_list = [os.path.join(image_path, f) for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))]

  for i in range(len(image_list)):
    image_to_transform = sk.io.imread(image_list[i])

    # concatenate name
    image_name = image_name_list[i].split('.',1)
    new_image_name = image_name[0]
    new_image_name = new_image_name + "_butterworth_hpf.JPEG"
    print(new_image_name)
    transformed_image = freq_filter(image_to_transform, 8, 2, "butterworth_hpf")

    if(transformed_image is None):
      print("None")
      continue # did will not do anything
    
    sk.io.imsave(save_path+'/'+new_image_name , transformed_image) 

#################################
#image_path = "/home/cdsw/Minh/DIP/data/original/train/n03888257"
#save_path = "/home/cdsw/Minh/DIP/data/imagenette2-160-butterworth-lpf/train/n03888257"
#image_path = "/home/cdsw/Minh/DIP/data/original/val/n03888257"
#save_path = "/home/cdsw/Minh/DIP/data/imagenette2-160-butterworth-lpf/val/n03888257"

def preprocess_butterworth_lpf():
  print("Running")
  # loop on all files of the folder and build a list of files paths
  image_name_list = [f for f in os.listdir(image_path) if os.path.splitext(f)[-1] == '.JPEG']
  image_list = [os.path.join(image_path, f) for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))]

  for i in range(len(image_list)):
    image_to_transform = sk.io.imread(image_list[i])

    # concatenate name
    image_name = image_name_list[i].split('.',1)
    new_image_name = image_name[0]
    new_image_name = new_image_name + "_butterworth_lpf.JPEG"
    print(new_image_name)
    transformed_image = freq_filter(image_to_transform, 8, 2, "butterworth_lpf")

    if(transformed_image is None):
      print("None")
      continue # did will not do anything
    
    sk.io.imsave(save_path+'/'+new_image_name , transformed_image) 
    
#################################
#image_path = "/home/cdsw/Minh/DIP/data/original/train/n03888257"
#save_path = "/home/cdsw/Minh/DIP/data/imagenette2-160-gaussian-lpf/train/n03888257"
#image_path = "/home/cdsw/Minh/DIP/data/original/val/n03888257"
#save_path = "/home/cdsw/Minh/DIP/data/imagenette2-160-gaussian-lpf/val/n03888257"

def preprocess_gaussian_lpf():
  print("Running")
  # loop on all files of the folder and build a list of files paths
  image_name_list = [f for f in os.listdir(image_path) if os.path.splitext(f)[-1] == '.JPEG']
  image_list = [os.path.join(image_path, f) for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))]

  for i in range(len(image_list)):
    image_to_transform = sk.io.imread(image_list[i])

    # concatenate name
    image_name = image_name_list[i].split('.',1)
    new_image_name = image_name[0]
    new_image_name = new_image_name + "_gaussian_lpf.JPEG"
    print(new_image_name)
    
    transformed_image = freq_filter(image_to_transform, 8, 2, "gaussian_lpf")

    if(transformed_image is None):
      print("None")
      continue # did will not do anything
    
    sk.io.imsave(save_path+'/'+new_image_name , transformed_image) 
    
#################################
#image_path = "/home/cdsw/Minh/DIP/data/original/train/n03888257"
#save_path = "/home/cdsw/Minh/DIP/data/imagenette2-160-gaussian-hpf/train/n03888257"
#image_path = "/home/cdsw/Minh/DIP/data/original/val/n03888257"
#save_path = "/home/cdsw/Minh/DIP/data/imagenette2-160-gaussian-hpf/val/n03888257"

def preprocess_gaussian_hpf():
  print("Running")
  # loop on all files of the folder and build a list of files paths
  image_name_list = [f for f in os.listdir(image_path) if os.path.splitext(f)[-1] == '.JPEG']
  image_list = [os.path.join(image_path, f) for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))]

  for i in range(len(image_list)):
    image_to_transform = sk.io.imread(image_list[i])

    # concatenate name
    image_name = image_name_list[i].split('.',1)
    new_image_name = image_name[0]
    new_image_name = new_image_name + "_gaussian_hpf.JPEG"
    print(new_image_name)
    
    transformed_image = freq_filter(image_to_transform, 8, 2, "gaussian_hpf")

    if(transformed_image is None):
      print("None")
      continue # did will not do anything

    sk.io.imsave(save_path+'/'+new_image_name , transformed_image) 
    
#################################
#image_path = "/home/cdsw/Minh/DIP/data/imagenette2-160/train/n03888257"
#save_path = "/home/cdsw/Minh/DIP/data/imagenette2-160-canny-ed/train/n03888257"
#image_path = "/home/cdsw/Minh/DIP/data/imagenette2-160/val/n03888257"
#save_path = "/home/cdsw/Minh/DIP/data/imagenette2-160-canny-ed/val/n03888257"

def preprocess_canny_ed():
  print("Running")
  # loop on all files of the folder and build a list of files paths
  image_name_list = [f for f in os.listdir(image_path) if os.path.splitext(f)[-1] == '.JPEG']
  image_list = [os.path.join(image_path, f) for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))]

  for i in range(len(image_list)):
    image_to_transform = sk.io.imread(image_list[i])

    # concatenate name
    image_name = image_name_list[i].split('.',1)
    new_image_name = image_name[0]
    new_image_name = new_image_name + "_canny_ed.JPEG"
    print(new_image_name)

    transformed_image = canny(image_to_transform)
    sk.io.imsave(save_path+'/'+new_image_name , transformed_image) 
    
#################################
#image_path = "/home/cdsw/Minh/DIP/data/original/train/n03888257"
#save_path = "/home/cdsw/Minh/DIP/data/imagenette2-160-invert/train/n03888257"
#image_path = "/home/cdsw/Minh/DIP/data/original/val/n03888257"
#save_path = "/home/cdsw/Minh/DIP/data/imagenette2-160-invert/val/n03888257"

def preprocess_invert():
  print("Running")
  # loop on all files of the folder and build a list of files paths
  image_name_list = [f for f in os.listdir(image_path) if os.path.splitext(f)[-1] == '.JPEG']
  image_list = [os.path.join(image_path, f) for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))]

  for i in range(len(image_list)):
    image_to_transform = sk.io.imread(image_list[i])

    # concatenate name
    image_name = image_name_list[i].split('.',1)
    new_image_name = image_name[0]
    new_image_name = new_image_name + "_invert.JPEG"
    print(new_image_name)

    transformed_image = invert(image_to_transform)
    sk.io.imsave(save_path+'/'+new_image_name , transformed_image) 
    
#################################
#image_path = "/home/cdsw/Minh/DIP/data/imagenette2-160/train/n03888257"
#save_path = "/home/cdsw/Minh/DIP/data/imagenette2-160-hist-equal/train/n03888257"
#image_path = "/home/cdsw/Minh/DIP/data/imagenette2-160/val/n03888257"
#save_path = "/home/cdsw/Minh/DIP/data/imagenette2-160-hist-equal/val/n03888257"

def preprocess_hist_equal():
  print("Running")
  # loop on all files of the folder and build a list of files paths
  image_name_list = [f for f in os.listdir(image_path) if os.path.splitext(f)[-1] == '.JPEG']
  image_list = [os.path.join(image_path, f) for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))]

  for i in range(len(image_list)):
    image_to_transform = sk.io.imread(image_list[i])

    # concatenate name
    image_name = image_name_list[i].split('.',1)
    new_image_name = image_name[0]
    new_image_name = new_image_name + "_hist_equal.JPEG"
    print(new_image_name)

    transformed_image = hist_equalization(image_to_transform)
    sk.io.imsave(save_path+'/'+new_image_name , transformed_image)     

#################################
#image_path = "/home/cdsw/Minh/DIP/data/imagenette2-160/train/n03888257"
#save_path = "/home/cdsw/Minh/DIP/data/imagenette2-160-hori-shear/train/n03888257"
#image_path = "/home/cdsw/Minh/DIP/data/imagenette2-160/val/n03888257"
#save_path = "/home/cdsw/Minh/DIP/data/imagenette2-160-hori-shear/val/n03888257"

def preprocess_shear_hori():
  print("Running")
  # loop on all files of the folder and build a list of files paths
  image_name_list = [f for f in os.listdir(image_path) if os.path.splitext(f)[-1] == '.JPEG']
  image_list = [os.path.join(image_path, f) for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))]

  for i in range(len(image_list)):
    image_to_transform = sk.io.imread(image_list[i])

    # concatenate name
    image_name = image_name_list[i].split('.',1)
    new_image_name = image_name[0]
    new_image_name = new_image_name + "_hori_shear.JPEG"
    print(new_image_name)

    transformed_image = horizontal_shear(image_to_transform)
    sk.io.imsave(save_path+'/'+new_image_name , transformed_image) 

#################################
#image_path = "/home/cdsw/Minh/DIP/data/imagenette2-160/train/n03888257"
#save_path = "/home/cdsw/Minh/DIP/data/imagenette2-160-vert-shear/train/n03888257"
#image_path = "/home/cdsw/Minh/DIP/data/imagenette2-160/val/n03888257"
#save_path = "/home/cdsw/Minh/DIP/data/imagenette2-160-vert-shear/val/n03888257"

def preprocess_shear_verti():
  print("Running")
  # loop on all files of the folder and build a list of files paths
  image_name_list = [f for f in os.listdir(image_path) if os.path.splitext(f)[-1] == '.JPEG']
  image_list = [os.path.join(image_path, f) for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))]

  for i in range(len(image_list)):
    image_to_transform = sk.io.imread(image_list[i])

    # concatenate name
    image_name = image_name_list[i].split('.',1)
    new_image_name = image_name[0]
    new_image_name = new_image_name + "_vert_shear.JPEG"
    print(new_image_name)

    transformed_image = vertical_shear(image_to_transform)
    sk.io.imsave(save_path+'/'+new_image_name , transformed_image) 
    
#################################
#image_path = "/home/cdsw/Minh/DIP/data/imagenette2-160/train/n03888257"
#save_path = "/home/cdsw/Minh/DIP/data/imagenette2-160-translater/train/n03888257"
#image_path = "/home/cdsw/Minh/DIP/data/imagenette2-160/val/n03888257"
#save_path = "/home/cdsw/Minh/DIP/data/imagenette2-160-translate/val/n03888257"

def preprocess_translate():
  print("Running")
  # loop on all files of the folder and build a list of files paths
  image_name_list = [f for f in os.listdir(image_path) if os.path.splitext(f)[-1] == '.JPEG']
  image_list = [os.path.join(image_path, f) for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))]

  for i in range(len(image_list)):
    image_to_transform = sk.io.imread(image_list[i])

    # concatenate name
    image_name = image_name_list[i].split('.',1)
    new_image_name = image_name[0]
    new_image_name = new_image_name + "_translate.JPEG"
    print(new_image_name)

    transformed_image = translate(image_to_transform)
    sk.io.imsave(save_path+'/'+new_image_name , transformed_image) 

#################################
#image_path = "/home/cdsw/Minh/DIP/data/original/train/n03888257"
#save_path = "/home/cdsw/Minh/DIP/data/imagenette2-160-ideal-hpf/train/n03888257"
#image_path = "/home/cdsw/Minh/DIP/data/original/val/n03888257"
#save_path = "/home/cdsw/Minh/DIP/data/imagenette2-160-ideal-hpf/val/n03888257"

def preprocess_ideal_hpf():
  print("Running")
  # loop on all files of the folder and build a list of files paths
  image_name_list = [f for f in os.listdir(image_path) if os.path.splitext(f)[-1] == '.JPEG']
  image_list = [os.path.join(image_path, f) for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))]

  for i in range(len(image_list)):
    image_to_transform = sk.io.imread(image_list[i])

    # concatenate name
    image_name = image_name_list[i].split('.',1)
    new_image_name = image_name[0]
    new_image_name = new_image_name + "_ideal_hpf.JPEG"
    print(new_image_name)

    transformed_image = freq_filter(image_to_transform, 8, 2, "ideal_hpf")
    if(transformed_image is None):
      print("None")
      continue # did will not do anything and pass to the next image
  
    sk.io.imsave(save_path+'/'+new_image_name , transformed_image) 
    
#################################
#image_path = "/home/cdsw/Minh/DIP/data/original/train/n03888257"
#save_path = "/home/cdsw/Minh/DIP/data/imagenette2-160-ideal-lpf/train/n03888257"
#image_path = "/home/cdsw/Minh/DIP/data/original/val/n03888257"
#save_path = "/home/cdsw/Minh/DIP/data/imagenette2-160-ideal-lpf/val/n03888257"

def preprocess_ideal_lpf():
  print("Running")
  # loop on all files of the folder and build a list of files paths
  image_name_list = [f for f in os.listdir(image_path) if os.path.splitext(f)[-1] == '.JPEG']
  image_list = [os.path.join(image_path, f) for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))]

  for i in range(len(image_list)):
    image_to_transform = sk.io.imread(image_list[i])

    # concatenate name
    image_name = image_name_list[i].split('.',1)
    new_image_name = image_name[0]
    new_image_name = new_image_name + "_ideal_lpf.JPEG"
    print(new_image_name)
    transformed_image = freq_filter(image_to_transform, 8, 2, "ideal_lpf")

    if(transformed_image is None):
      print("None")
      continue # did will not do anything
    
    sk.io.imsave(save_path+'/'+new_image_name , transformed_image) 
    
#################################
#image_path = "/home/cdsw/Minh/DIP/data/imagenette2-160/train/n03888257"
#save_path = "/home/cdsw/Minh/DIP/data/imagenette2-160-rand-rotate/train/n03888257"
#image_path = "/home/cdsw/Minh/DIP/data/imagenette2-160/val/n03888257"
#save_path = "/home/cdsw/Minh/DIP/data/imagenette2-160-rand-rotate/val/n03888257"

def preprocess_rotate():
  print("Running")
  # loop on all files of the folder and build a list of files paths
  image_name_list = [f for f in os.listdir(image_path) if os.path.splitext(f)[-1] == '.JPEG']
  image_list = [os.path.join(image_path, f) for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))]

  for i in range(len(image_list)):
    image_to_transform = sk.io.imread(image_list[i])

    # concatenate name
    image_name = image_name_list[i].split('.',1)
    new_image_name = image_name[0]
    new_image_name = new_image_name + "_rand_rotate.JPEG"
    print(new_image_name)

    transformed_image = rotate(image_to_transform, max_theta = 360, mode = "random")
    sk.io.imsave(save_path+'/'+new_image_name , transformed_image) 

#################################
#image_path = "/home/cdsw/Minh/DIP/data/imagenette2-160/train/n03888257"
#save_path = "/home/cdsw/Minh/DIP/data/imagenette2-160-vert-flip/train/n03888257"
#image_path = "/home/cdsw/Minh/DIP/data/imagenette2-160/val/n03888257"
#save_path = "/home/cdsw/Minh/DIP/data/imagenette2-160-vert-flip/val/n03888257"

def preprocess_flip_vert():
  print("Running")
  # loop on all files of the folder and build a list of files paths
  image_name_list = [f for f in os.listdir(image_path) if os.path.splitext(f)[-1] == '.JPEG']
  image_list = [os.path.join(image_path, f) for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))]

  for i in range(len(image_list)):
    image_to_transform = sk.io.imread(image_list[i])

    # concatenate name
    image_name = image_name_list[i].split('.',1)
    new_image_name = image_name[0]
    new_image_name = new_image_name + "_vert_flip.JPEG"
    print(new_image_name)

    transformed_image = flip_vertical(image_to_transform)
    sk.io.imsave(save_path+'/'+new_image_name , transformed_image) 

#################################
#image_path = "/home/cdsw/Minh/DIP/data/imagenette2-160/train/n03888257"
#save_path = "/home/cdsw/Minh/DIP/data/imagenette2-160-hori-flip/train/n03888257"
#image_path = "/home/cdsw/Minh/DIP/data/imagenette2-160/val/n03888257"
#save_path = "/home/cdsw/Minh/DIP/data/imagenette2-160-hori-flip/val/n03888257"

def preprocess_flip_hori():
  print("Running")
  # loop on all files of the folder and build a list of files paths
  image_name_list = [f for f in os.listdir(image_path) if os.path.splitext(f)[-1] == '.JPEG']
  image_list = [os.path.join(image_path, f) for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))]

  for i in range(len(image_list)):
    image_to_transform = sk.io.imread(image_list[i])
#    plt.imshow(image_to_transform)
#    plt.show()

    # concatenate name
    image_name = image_name_list[i].split('.',1)
    new_image_name = image_name[0]
    new_image_name = new_image_name + "_hori_flip.JPEG"
    print(new_image_name)

    transformed_image = flip_horizontal(image_to_transform)
#    plt.imshow(transformed_image)
#    plt.show()
    sk.io.imsave(save_path+'/'+new_image_name , transformed_image)  

#################################
if __name__ == "__main__":
#  preprocess_flip_hori()
#  preprocess_flip_vert()
#  preprocess_rotate()
#  preprocess_shear_hori()
#  preprocess_shear_verti()
#  preprocess_translate()
#  preprocess_ideal_hpf()
#  preprocess_ideal_lpf()
  preprocess_amf()
#  preprocess_hist_equal()
#  preprocess_invert()
#  preprocess_canny_ed()
#  preprocess_gaussian_lpf()
#  preprocess_gaussian_hpf()