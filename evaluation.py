'''
Evaluate the model.
Support 4 evaluation matrix determined in the cfgs file
'''
import tensorflow as tf
import os
import numpy as np
import cv2
from process_data import random_crop
import cfgs
from process_data import build_colormap2label,convert_voc_annotations
from cfgs import VOC_CLASSES,VOC_COLORMAP
from inference import predictions

def get_images(val_dir):
    '''
    Unlike get images for the training set, we only round the image resolution
    to the neareast devisible by 32.
    Parameters:
    val_dir: validation directory
    '''
    list_images=[]
    list_gt=[]
    image_names=os.listdir(val_dir+'/SegmentationClass')
    colormap=build_colormap2label()
    for name in image_names:
        label=cv2.imread(val_dir+'/SegmentationClass/'+name)[:,:,::-1]
        img=cv2.imread(val_dir+'/JPEGImages/'+name[:-3]+'jpg')[:,:,::-1]
        shape0=img.shape[0]//32
        shape1=img.shape[1]//32
        shape0*=32;shape1*=32
        img,label=random_crop(img,label,crop_size=(shape0,shape1))
        list_images.append(img)
        label=convert_voc_annotations(label,colormap).astype(np.uint8)
        list_gt.append(label)
    return list_images,list_gt

def compare(pred,gt,num_classes):
    '''
    Compare the pred and the ground truth for an image
    pred: 2D numpy array, prediction at each pixel location
    gt: 2D numpy array, same size as pred
    eval_mat: numpy array, with the shape of (num_class,num_class)
    '''
    eval_mat=np.zeros((num_classes,num_classes)).astype(np.int32)
    # eval_mat[i,j] means that number of class i be predicted to class j
    for i in range(num_classes):
        gt_mask=np.where(gt==i,1,0) #ground truth mask
        for j in range(num_classes):
            pred_mask=np.where(pred==j,1,0) #pred mask
            eval_mat[i,j]=np.sum(gt_mask*pred_mask)
    return eval_mat

def evaluate(list_images,list_gt,model,num_classes,mode):
    '''
    Evaluate the model
    '''
    eval_mat=np.zeros((num_classes,num_classes)).astype(np.int64)
    for i in range(len(list_images)):
        pred=predictions(model,list_images[i])
        pred=np.reshape(pred,(pred.shape[1],pred.shape[2])).astype(np.uint8)
        eval_mat=eval_mat+compare(pred,list_gt[i],num_classes)
    
    if(mode=='pacc'):
        sum_diag=0.0
        for i in range(num_classes):
            sum_diag+=eval_mat[i,i]
        total=float(eval_mat.sum())
        return sum_diag/total,eval_mat
    elif(mode=='macc'):
        result=0.0
        for i in range(num_classes):
            result+=float(eval_mat[i,i])/float(np.sum(eval_mat[i,:]))
        return result/num_classes,eval_mat
    elif(mode=='mIU'):
        result=0.0
        for i in range(num_classes):
            intersect=float(eval_mat[i,i])
            union=float(np.sum(eval_mat[i,:])+np.sum(eval_mat[:,i])-eval_mat[i,i])
            result+=intersect/union
        return result/num_classes,eval_mat
    elif(mode=='wIU'):
        result=0.0
        for i in range(num_classes):
            intersect=float(eval_mat[i,i])
            union=float(np.sum(eval_mat[i,:])+np.sum(eval_mat[:,i])-intersect)
            result+=np.sum(eval_mat[i,:])*intersect/union
        weight=np.sum(eval_mat)
        return result/weight,eval_mat

if __name__=='__main__':
    list_images,list_gt=get_images(val_dir=cfgs.test_dir)
    print('There are {} images in the validation set'.format(len(list_images)))
    model=tf.keras.models.load_model(cfgs.model_dir+cfgs.model+'.h5')
    res,mat=evaluate(list_images,list_gt,model,num_classes=len(cfgs.VOC_CLASSES),mode=cfgs.eval)
    print('Evaluation matrix '+str(cfgs.eval)+' :'+str(res))
