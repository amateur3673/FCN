import tensorflow as tf
from process_data import bilinear_interpolation

# Create FCN model with backbone ResNet

def fcn_32_net(num_classes):
    '''
    Build the fcn 32 network on pretrained model ResNet-50
    num_classes: number of classes
    '''
    #download the pretrained model
    resnet=tf.keras.applications.ResNet50(include_top=False,weights='imagenet')
    resnet.trainable=False
    #Initialize the weights of tranpose convolution layer
    weights=bilinear_interpolation(num_classes,64)
    kernel=tf.keras.initializers.Constant(value=weights)
    #Create the base layer for the model
    conv=tf.keras.layers.Conv2D(filters=num_classes,kernel_size=1,strides=1,padding='same',activation='relu')
    tconv=tf.keras.layers.Conv2DTranspose(filters=num_classes,kernel_size=64,strides=32,padding='same',activation='softmax',kernel_initializer=kernel)
    #puts every together

    inputs=tf.keras.Input(shape=(None,None,3))
    x=process(inputs)
    x=resnet(x,training=False)
    x=conv(x)
    outputs=tconv(x)

    return tf.keras.Model(inputs=inputs,outputs=outputs)

def fcn_16_net(num_classes):
    '''
    Build fcn-16s network on pretrained ResNet-50
    '''

    resnet=tf.keras.applications.ResNet50(include_top=False,weights='imagenet')
    resnet.trainable=False
    #Initialize weights
    weights_2=bilinear_interpolation(num_classes,4)
    kernel_2=tf.keras.initializers.Constant(value=weights_2)

    weights_16=bilinear_interpolation(num_classes,32)
    kernel_16=tf.keras.initializers.Constant(value=weights_16)

    #Build the model
    y_32=tf.keras.layers.Conv2D(filters=num_classes,kernel_size=1,strides=1,padding='same',activation='relu')(resnet.output)
    y_16=tf.keras.layers.Conv2D(filters=num_classes,kernel_size=1,strides=1,padding='same',activation='relu')(resnet.get_layer('conv4_block6_out').output)

    y_32_up=tf.keras.layers.Conv2DTranspose(filters=num_classes,kernel_size=4,strides=2,padding='same',activation='relu',kernel_initializer=kernel_2)(y_32)
    y_16_skip=tf.keras.layers.Add()([y_32_up,y_16])
    outputs=tf.keras.layers.Conv2DTranspose(filters=num_classes,kernel_size=32,strides=16,padding='same',activation='softmax',kernel_initializer=kernel_16)(y_16_skip)

    return tf.keras.Model(inputs=resnet.input,outputs=outputs)
    
def fcn_8_net(num_classes):
    '''
    Construct FCN-8s
    '''
    resnet=tf.keras.applications.ResNet50(include_top=False)
    resnet.trainable=False
    #Initialize the weight
    weights_2=bilinear_interpolation(num_classes,4)
    kernel_2=tf.keras.initializers.Constant(value=weights_2)
    
    weights_8=bilinear_interpolation(num_classes,16)
    kernel_8=tf.keras.initializers.Constant(value=weights_8)

    y_32=tf.keras.layers.Conv2D(filters=num_classes,kernel_size=1,activation='relu')(resnet.output)

    y_16=tf.keras.layers.Conv2D(filters=num_classes,kernel_size=1,activation='relu')(resnet.get_layer('conv4_block6_out').output)

    y_8=tf.keras.layers.Conv2D(filters=num_classes,kernel_size=1,activation='relu')(resnet.get_layer('conv3_block4_out').output)

    y_32_up=tf.keras.layers.Conv2DTranspose(filters=num_classes,kernel_size=4,strides=2,padding='same',activation='relu',kernel_initializer=kernel_2)(y_32)
    y_16_skip=tf.keras.layers.Add()([y_32_up,y_16])
    y_16_up=tf.keras.layers.Conv2DTranspose(filters=num_classes,kernel_size=4,strides=2,padding='same',activation='relu',kernel_initializer=kernel_2)(y_16_skip)
    y_8_skip=tf.keras.layers.Add()([y_16_up,y_8])
    outputs=tf.keras.layers.Conv2DTranspose(filters=num_classes,kernel_size=16,strides=8,padding='same',activation='softmax',kernel_initializer=kernel_8)(y_8_skip)

    return tf.keras.Model(inputs=resnet.inputs,outputs=outputs)
