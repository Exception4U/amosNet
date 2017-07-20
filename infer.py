from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from scipy import misc
import argparse
import csv


def load_weight(sess, weight_path):
    pre_trained_weights = np.load(open(weight_path, "rb"), encoding="latin1").\
        item()
    keys = sorted(pre_trained_weights.keys())
    for k in keys:
        with tf.variable_scope(k, reuse=True):
            temp = tf.get_variable('weights')
            sess.run(temp.assign(pre_trained_weights[k]['weights']))
        with tf.variable_scope(k, reuse=True):
            temp = tf.get_variable('biases')
            sess.run(temp.assign(pre_trained_weights[k]['biases']))


def conv(input, filter_size, in_channels, out_channels, name, strides, padding, groups):
    with tf.variable_scope(name) as scope:
        filt = tf.get_variable('weights', shape=[filter_size, filter_size, int(in_channels/groups), out_channels])
        bias = tf.get_variable('biases',  shape=[out_channels])
    if groups == 1:
        return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(input, filt, strides=strides, padding=padding), bias))
    else:
        # Split input and weights and convolve them separately
        input_groups = tf.split(split_dim = 3, num_split=groups, value=input)
        filt_groups = tf.split(split_dim = 3, num_split=groups, value=filt)
        output_groups = [ tf.nn.conv2d( i, k, strides = strides, padding = padding) for i,k in zip(input_groups, filt_groups)]

        conv = tf.concat(concat_dim = 3, values = output_groups)
        return tf.nn.relu(tf.nn.bias_add(conv, bias))

def fc(input, in_channels, out_channels, name, relu):
    input = tf.reshape(input , [-1, in_channels])
    with tf.variable_scope(name) as scope:
        filt = tf.get_variable('weights', shape=[in_channels , out_channels])
        bias = tf.get_variable('biases',  shape=[out_channels])
    if relu:
        return tf.nn.relu(tf.nn.bias_add(tf.matmul(input, filt), bias))
    else:
        return tf.nn.bias_add(tf.matmul(input, filt), bias)


def pool(input, padding, name):
    return tf.nn.max_pool(input, ksize=[1,3,3,1], strides=[1,2,2,1], padding=padding, name= name)


#placeholders
x = tf.placeholder(tf.float32, shape = [None, None, None,3])

#AmosNet Conv-Layers
net_layers={}
net_layers['conv1'] = conv(x, 11, 3, 96, name= 'conv1', strides=[1,4,4,1] ,padding='VALID', groups=1)
net_layers['pool1'] = pool(net_layers['conv1'], padding='VALID', name='pool1')
net_layers['lrn1']  = tf.nn.lrn(net_layers['pool1'], depth_radius=2, alpha=2e-5, beta=0.75,name='norm1')

net_layers['conv2'] = conv(net_layers['lrn1'], 5, 96, 256, name= 'conv2', strides=[1,1,1,1] ,padding='SAME', groups=2)
net_layers['pool2'] = pool(net_layers['conv2'], padding='VALID', name='pool2')
net_layers['lrn2']  = tf.nn.lrn(net_layers['pool2'], depth_radius=2, alpha=2e-5, beta=0.75,name='norm2')

net_layers['conv3'] = conv(net_layers['lrn2'], 3, 256, 384, name='conv3', strides=[1,1,1,1] ,padding='SAME', groups=1)

net_layers['conv4'] = conv(net_layers['conv3'], 3, 384, 384, name='conv4', strides=[1,1,1,1] ,padding='SAME', groups=2)

net_layers['conv5'] = conv(net_layers['conv4'], 3, 384, 256, name='conv5', strides=[1,1,1,1] ,padding='SAME', groups=2)

net_layers['conv6'] = conv(net_layers['conv5'], 3, 256, 256, name='conv6', strides=[1,1,1,1] ,padding='SAME', groups=2)
net_layers['pool6'] = pool(net_layers['conv6'], padding='VALID', name='pool6')

net_layers['fc7'] = fc(net_layers['pool6'],  6*6*256, 4096, name='fc7_new', relu = 1)
net_layers['fc8'] = fc(net_layers['fc7'], 4096, 2543, name='fc8_new', relu = 0)

net_layers['prob'] = tf.nn.softmax(net_layers['fc8'])
net_layers['pred'] = tf.argmax(tf.nn.softmax(net_layers['fc8']), dimension = 1)

def normalize_input(img):
    img = img.astype(dtype=np.float32)
    img = img[:, :, [2, 1, 0]] # swap channel from RGB to BGR
    img = img - [104,117,124]
    return img

def load_preprocess_image(img_path):
    img = misc.imread(img_path)
    img = normalize_input(img)
    img = misc.imresize(np.asarray(img), (256, 256))
    return img

def main(weight_path, img_paths, layer):
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4

    with tf.Session(config=config) as sess:
        sess.run(tf.initialize_all_variables())
        load_weight(sess, weight_path=weight_path)

        with open('results.csv',"w") as f:
            writer = csv.writer(f, delimiter=',',  quotechar='"', quoting=csv.QUOTE_ALL)
            for img_path in img_paths:
                img = load_preprocess_image(img_path)

                img = tf.slice(img, begin=[14, 14, 0], size=[227, 227, -1])
                img = tf.expand_dims(img, dim=0)
                [img] = sess.run([img])

                [result] = sess.run(net_layers[layer], feed_dict={x:img})

                print(img_path, result)
                result = result.flatten()

                writer.writerow(result)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", help="layer to output eg:conv1, pool2, fc7", type=str)
    parser.add_argument( "--weight_path", help="path to stored CNN-weights",  type=str)
    parser.add_argument( "--img_path", nargs='+', help="path of the image",  type=str)

    config = parser.parse_args()
    print(config)
    main(config.weight_path, config.img_path, config.layer)
