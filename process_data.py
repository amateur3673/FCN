import tensorflow as tf
import numpy as np
import cv2
import os

from cfgs import VOC_CLASSES,VOC_COLORMAP,crop_size,train_dir,test_dir

def build_colormap2label():
   '''
   Convert into an array represent the index of each color in the image
   '''
   colormap2label=np.zeros((256**3,))
   for idx,colormap in enumerate(VOC_COLORMAP):
      colormap2label[(colormap[0]*256+colormap[1])*256+colormap[2]]=idx
   return colormap2label

def convert_voc_annotations(img,colormap2label):
   '''
   Convert an segmentation image to label corresponding to each pixel
   Parameters:
   img: 3D numpy array, represent the image
   colormap2label: the colormap2label array that we use in above function
   '''
   img=img.astype(np.int32)
   idx=(img[:,:,0]*256+img[:,:,1])*256+img[:,:,2]
   return colormap2label[idx]

def random_crop(img,annotation_img,crop_size):
   '''
   Random crop a fixed size from the original images
   Parameters:
   img: the image, 3D numpy array
   annotation_img: annotation image, 3D numpy array
   crop_size: tuple, represents the width and height of the cropped region
   '''
   start_height=np.random.randint(low=0,high=img.shape[0]-crop_size[0]+1)
   start_width=np.random.randint(low=0,high=img.shape[1]-crop_size[1]+1)
   crop_img=img[start_height:start_height+crop_size[0],start_width:start_width+crop_size[1]]
   crop_annotation=annotation_img[start_height:start_height+crop_size[0],start_width:start_width+crop_size[1]]
   return crop_img,crop_annotation

def bilinear_interpolation(channels,kernel_size):
    '''
    Initialize bilinear interpolation for upsampling tranpose convolution
    Parameters:
    channels: number of categories
    kernel_size: kernel size of of the transpose convolution
    Return a numpy array represent the kernel
    '''
    factor=(kernel_size+1)//2
    if(kernel_size%2==1):
       center=factor-1
    else:
       center=factor-0.5
    og=(np.arange(kernel_size).reshape(-1,1),np.arange(kernel_size).reshape(1,-1))
    filt=(1-np.abs(og[0]-center)/factor)*(1-np.abs(og[1]-center)/factor)
    weights=np.zeros((kernel_size,kernel_size,channels,channels))
    for i in range(channels):
       weights[:,:,i,i]=filt
    return weights

def get_images(path,crop_value):
    '''
    Get image from directory.
    Parameters:
    path: path to the image
    crop_value: the size we crop (reject all images less than crop value)
    '''
    images=[]
    gt_images=[]
    colormap=build_colormap2label()
    img_name=os.listdir(path+'/SegmentationClass')
    for name in img_name:
       img=cv2.imread(path+'/SegmentationClass/'+name)[:,:,::-1]
       if(img.shape[0]>crop_value[0] and img.shape[1]>crop_value[1]):
           label=convert_voc_annotations(img,colormap)
           if(len(np.unique(label))>1):
               gt_images.append(img)
               img=cv2.imread(path+'/JPEGImages/'+name[:-3]+'jpg')[:,:,::-1]
               images.append(img)
    print(len(images))
    return images,gt_images

class DataGen(tf.keras.utils.Sequence):
   '''
   Class for datagenerator
   '''
   def __init__(self,train_img,gt_img,crop_value=(320,480),batch_size=8,num_classes=21):
      '''
      Initialize the DataGen. Parameters:
      parameters:
      train_img: list of the training image
      gt_img: list of the colormap
      crop_value: value we use cropping
      batch_size: batch size of the model
      num_classes: number of classes in the training set
      '''
      self.list_images=train_img
      self.groundtruth_img=gt_img
      self.crop_value=crop_value
      self.batch_size=batch_size
      self.colormap=build_colormap2label()
      self.num_classes=num_classes
      self.on_epoch_end()
   def on_epoch_end(self):
      self.indices=np.arange(len(self.list_images))
      np.random.shuffle(self.indices)
   def process_batch(self,batch_index):
      '''
      Process a batch for the training set.
      Parameters:
      batch_index: 1D array, represents the index of the batch
      '''
      images=np.zeros((len(batch_index),self.crop_value[0],self.crop_value[1],3))
      labels=np.zeros((len(batch_index),self.crop_value[0],self.crop_value[1]))

      for i in range(len(batch_index)):
         img,gt=random_crop(self.list_images[batch_index[i]],self.groundtruth_img[batch_index[i]],self.crop_value)
         gt=convert_voc_annotations(gt,self.colormap)
         img=img.astype(np.float32)
         images[i]=img
         labels[i]=gt
      images=tf.keras.applications.resnet.preprocess_input(images)
      labels=tf.one_hot(labels,depth=self.num_classes,axis=3)
      return images,labels

   def __getitem__(self, index):
      if (index+1)*self.batch_size<len(self.list_images):
         batch_index=self.indices[index*self.batch_size:(index+1)*self.batch_size]
      else:
         batch_index=self.indices[index*self.batch_size:]
      return self.process_batch(batch_index)
   
   def __len__(self):
      '''
      Number of steps per epoch
      '''
      return math.ceil(len(self.list_images)/self.batch_size)

