import tensorflow as tf
import network
import cfgs
from process_data import DataGen,get_images

print('Creating the model ...')
if(cfgs.model!='fcn-32' and cfgs.model!='fcn-16' and cfgs.model!='fcn-8'):
    raise AssertionError('Only support for FCN-32s, FCN-16s, FCN-8s model')

if(cfgs.model=='fcn-32'):
    model=network.fcn_32_net(num_classes=len(cfgs.VOC_CLASSES))

elif(cfgs.model=='fcn-16'):
    model=network.fcn_16_net(num_classes=len(cfgs.VOC_CLASSES))

elif(cfgs.model=='fcn-8'):
    model=network.fcn_8_net(num_classes=len(cfgs.VOC_CLASSES))

opt=tf.keras.optimizers.SGD(learning_rate=cfgs.lr,momentum=0.9,nesterov=True)

model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])

print('Get the image from directory ...')

train_img,train_gt=get_images(path=cfgs.train_dir,crop_value=cfgs.crop_size)
test_img,test_gt=get_images(path=cfgs.test_dir,crop_value=cfgs.crop_size)

train_gen=DataGen(train_img,train_gt,crop_value=cfgs.crop_size,batch_size=cfgs.batch_size,num_classes=len(cfgs.VOC_CLASSES))
test_gen=DataGen(test_img,test_gt,crop_value=cfgs.crop_size,batch_size=cfgs.batch_size,num_classes=len(cfgs.VOC_CLASSES))

print('Train some last layers ...')
model.fit_generator(train_gen,steps_per_epoch=len(train_img)//cfgs.batch_size,epochs=10,validation_data=test_gen)

print('Unfreeze all layers for finetuning')
for layer in model.layers:
    layer.trainable=True
model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])

model.fit_generator(train_gen,steps_per_epoch=len(train_img)//cfgs.batch_size,epochs=cfgs.epochs,validation_data=test_gen)

model.save(cfgs.model_dir+cfgs.model+'.h5')
