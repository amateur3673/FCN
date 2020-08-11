'''
Config file.
You need to set some parameters of this file to get, also train the model
'''

# PASCAL VOC Classes (select some class of pascal voc label)
VOC_CLASSES = ['background', 'bird','car', 'cat', 'dog', 'person']

# PASCAL VOC Colormap (corresponding colormap in RGB of these classes in the segmentation ground truth image)
VOC_COLORMAP = [[0, 0, 0], [128, 128, 0], [128, 128, 128],
[64, 0, 0],[64, 0, 128], [192, 128, 128]]

# type of model, here we support FCN-8s, FCN-16s and FCN-32s

model='fcn-8'

# Crop size, the amount of image we crop for training
# Expected to be a tuple represents the image height and width, should be divisible by 32 

crop_size=(320,480)

# training set directory

train_dir='VOCtrainval_11-May-2012/VOCdevkit/VOC2012'

# test set directory

test_dir='VOCdevkit/VOC2007'

# model file directory

model_dir='model/'

# Batch size

batch_size=8

# Number of epochs

epochs=70

# learning rate

lr=1e-4

# evaluation, must be one of these following choice:

# 'pacc': pixel accuracy
# 'macc': mean accuracy
# 'mIU': mean IU
# 'wIU': weighted IU

eval='mIU'