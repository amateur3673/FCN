import tensorflow as tf
import numpy as np
from cfgs import model, model_dir, VOC_CLASSES, VOC_COLORMAP 
import cv2


def predictions(model,images):
    '''
    Make the predictions about images
    Parameters:
    model: keras model
    images: can be 3D or 4D numpy array, represent the RGB image, expect in range [0,255]
    '''
    if(len(images.shape)==3):
        images=images.reshape(1,images.shape[0],images.shape[1],images.shape[2])
    
    images=images.astype(np.float32)
    pred=model(images).numpy()
    return np.argmax(pred,axis=3)

def create_mask(pred):
    '''
    Create the mask for image prediction
    Parameters:
    pred: array of the same size as the image
    '''
    mask=np.zeros((pred.shape[0],pred.shape[1],3)).astype(np.uint8)
    pred=pred.astype(np.uint8)
    # Reject the redundant pixels
    for class_id in np.unique(pred):
        pos=np.where(pred==class_id)
        if(len(pos[0])<30):
            pred[pos[0],pos[1]]=0
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            for c in range(3):
                mask[i,j,c]=VOC_COLORMAP[pred[i,j]][c]
    return pred,mask

if __name__=='__main__':
    model=tf.keras.models.load_model(model_dir+model+'.h5')
    while(True):
        print('Enter the image:')
        name=input()
        img=cv2.imread(name)[:,:,::-1]
        shape0=img.shape[0]//32
        shape1=img.shape[1]//32
        shape0*=32
        shape1*=32
        img=cv2.resize(img,(shape1,shape0))
        pred=predictions(model,np.reshape(img.astype(np.float32),(1,img.shape[0],img.shape[1],img.shape[2])))
        pred=pred.reshape(pred.shape[1],pred.shape[2])
        pred,mask=create_mask(pred)
        print('Segment the object: ',end='')
        for class_id in np.unique(pred):
            print(VOC_CLASSES[class_id],end=' ')
        cv2.imshow('segment_image',mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
