import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import math
import cv2
#Color map of each object in pascal voc datatset
VOC_COLORMAP = [[0, 0, 0], [128, 128, 0], [128, 128, 128],
[64, 0, 0],[64, 0, 128], [192, 128, 128]]

#Corresponding classes
VOC_CLASSES = ['background', 'bird'
, 'car', 'cat', 'dog', 'person']

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
def ratio(img,colormap):
   label=convert_voc_annotations(img,colormap).astype(np.uint8)
   pos=np.where(label==0)
   return len(pos[0])/(img.shape[0]*img.shape[1])

def random_crop(img,annotation_img,crop_size):
   '''
   Random crop a fixed size from the original images
   Parameters:
   img: the image, 3D numpy array
   annotation_img: annotation image, 3D numpy array
   crop_size: tuple, represents the width and height of the cropped region
   '''
   start_height=np.random.randint(low=0,high=img.shape[0]-crop_size[0])
   start_width=np.random.randint(low=0,high=img.shape[1]-crop_size[1])
   crop_img=img[start_height:start_height+crop_size[0],start_width:start_width+crop_size[1]]
   crop_annotation=annotation_img[start_height:start_height+crop_size[0],start_width:start_width+crop_size[1]]
   return crop_img,crop_annotation

def bilinear_interpolation(channels,kernel_size):
    '''
    Initialize bilinear interpolation for upsampling
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

def network_model(type='fcn-8',num_classes=21):
   '''
   Build the resnet model for FCN, here we use ResNet-50. Parameters:
   type: fcn-32 or fcn-16 or fcn-8.
   num_classes: number of classes in the training set
   '''
    
   resnet=tf.keras.applications.ResNet50(include_top=False,weights='imagenet')
   resnet.trainable=False
   layer_32=resnet.get_layer('conv5_block3_out').output
   layer_16=resnet.get_layer('conv4_block6_out').output
   layer_8=resnet.get_layer('conv3_block4_out').output

   y_32=tf.keras.layers.Conv2D(filters=num_classes,kernel_size=1,strides=1,padding='same',activation='relu')(layer_32)
   if(type=='fcn-32'):
      weights_32=bilinear_interpolation(num_classes,64)
      kernel=tf.keras.initializers.Constant(value=weights_32)
      outputs=tf.keras.layers.Conv2DTranspose(filters=num_classes,kernel_size=64,strides=32,kernel_initializer=kernel,activation='softmax')(y_32)
   
   elif(type=='fcn-16'):
      weights_32=bilinear_interpolation(num_classes,4)
      kernel=tf.keras.initializers.Constant(value=weights_32)
      y_32=tf.keras.layers.Conv2DTranspose(filters=num_classes,kernel_size=4,strides=2,padding='same',activation='relu',kernel_initializer=kernel)(y_32)
      y_16=tf.keras.layers.Conv2D(filters=num_classes,kernel_size=1,strides=1,padding='same',activation='relu')(layer_16)
      y_16=tf.keras.layers.Add()([y_32,y_16])
      weights_16=bilinear_interpolation(num_classes,32)
      kernel=tf.keras.initializers.Constant(value=weights_16)
      outputs=tf.keras.layers.Conv2DTranspose(filters=num_classes,kernel_size=32,strides=16,padding='same',activation='softmax',kernel_initializer=kernel)(y_16)
   
   else:
      weights=bilinear_interpolation(num_classes,4)
      kernel=tf.keras.initializers.Constant(value=weights)
      y_32=tf.keras.layers.Conv2DTranspose(filters=num_classes,kernel_size=4,strides=2,padding='same',activation='relu',kernel_initializer=kernel)(y_32)
      y_16=tf.keras.layers.Conv2D(filters=num_classes,kernel_size=1,strides=1,padding='same',activation='relu')(layer_16)
      y_16=tf.keras.layers.Add()([y_32,y_16])
      y_16=tf.keras.layers.Conv2DTranspose(filters=num_classes,kernel_size=4,strides=2,padding='same',activation='relu',kernel_initializer=kernel)(y_16)
      y_8=tf.keras.layers.Conv2D(filters=num_classes,kernel_size=1,strides=1,padding='same',activation='relu')(layer_8)
      y_8=tf.keras.layers.Add()([y_16,y_8])
      weights_8=bilinear_interpolation(num_classes,16)
      kernel=tf.keras.initializers.Constant(value=weights_8)
      outputs=tf.keras.layers.Conv2DTranspose(filters=num_classes,kernel_size=16,strides=8,padding='same',activation='softmax',kernel_initializer=kernel)(y_8)
   
   return tf.keras.Model(inputs=resnet.inputs,outputs=outputs)

def get_images(path,crop_value=(320,480)):
   
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

def pixel_loss(y_true,y_pred):
   '''
   Loss function of the model.
   Parameters:
   y_true: ground truth label
   y_pred: predict label
   '''
   loss=tf.keras.losses.categorical_crossentropy(y_true,y_pred,from_logits=True)
   return tf.math.reduce_mean(loss)

def prediction(model,images):
    '''
    Make the prediction
    Parameters:
    images: 3D or 4D tensor, 4D represents the batch of the image
    model: keras model
    '''
    if(len(images.shape)==3):images=images.reshape(1,images.shape[0],images.shape[1],images.shape[2])
    images=images.astype(np.float32)
    predictions=model(images).numpy()
    return np.argmax(predictions,axis=3)

def draw_pred_images(image,pred):
    pred_img=np.zeros_like(image).astype(np.uint8)
    for i in range(pred_img.shape[0]):
       for j in range(pred_img.shape[1]):
         for c in range(3):
            pred_img[i,j,c]=VOC_COLORMAP[pred[i,j]][c]
    return pred_img

def train_model(model_type='fcn-8',num_classes=21,path='VOCtrainval_11-May-2012/VOCdevkit/VOC2012',crop_value=(320,480),batch_size=8,epochs=10):
   '''
   Perform training model
   '''
   model=network_model(type=model_type,num_classes=num_classes)

   optimizers=tf.keras.optimizers.Adam(learning_rate=0.001)
   model.compile(optimizer=optimizers,loss=pixel_loss)

   datagen=DataGen(path=path,crop_value=crop_value,batch_size=batch_size,num_classes=num_classes)
   
   hist=model.fit_generator(datagen,steps_per_epoch=len(datagen.list_images)//batch_size,epochs=epochs)

   return hist,model

if __name__=='__main__':
   (train_img,train_gt),(test_img,test_gt)=split_train_test(path='VOCtrainval_11-May-2012/VOCdevkit/VOC2012',rat=1.0)
   print(len(train_img))
   print(len(train_gt))
   print(len(test_img))
   print(len(test_gt))
