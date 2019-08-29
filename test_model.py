import os

import numpy as np
import tensorflow as tf

import aae
import matplotlib.pyplot as plt
from matplotlib import gridspec, colors
from datetime import datetime
import scipy.io as sio
from skimage.measure import compare_ssim, compare_psnr
import math

noise_level=35.


def psnr(target, reconstract):
    # target:目标图像  ref:参考图像  scale:尺寸大小
    target = (target-target.min())/(target.max()-target.min())
    reconstract = (reconstract-reconstract.min()) / \
        (reconstract.max()-reconstract.min())
    target_data = np.array(target)
    ref_data = np.array(reconstract)
    diff = ref_data - target_data
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff ** 2.))
    return 20*np.log10(1.0/rmse)


sess = tf.Session()
# 先加载图和参数变量
saver = tf.train.import_meta_graph('./model/model.meta')
saver.restore(sess, tf.train.latest_checkpoint('./model'))


# 访问placeholders变量，并且创建feed-dict来作为placeholders的新值
graph = tf.get_default_graph()
x_hat = graph.get_tensor_by_name("input_img:0")
keep_prob = graph.get_tensor_by_name("keep_prob:0")
# 接下来，访问你想要执行的op
y = graph.get_tensor_by_name("MLP_decoder/decode_output:0")


load_fn = 'abu-airport-1.mat'  # 100*100*205
load_data = sio.loadmat(load_fn)
train_data = load_data['data']
train_data = np.array(train_data)
train_data_shape = train_data.shape
train_data = train_data.reshape([train_data_shape[0]*train_data_shape[1], 205])
train_data = ((train_data-train_data.min()) /
              (train_data.max()-train_data.min()))
"""
load_fn = 'abu-airport-1_noise_25.mat'  # 100*100*205
load_data = sio.loadmat(load_fn)
train_data_noise = load_data['Im']
train_data_noise = np.array(train_data_noise)
train_data_noise = train_data_noise.reshape(
    [train_data_shape[0]*train_data_shape[1], 205])
train_data_noise = ((train_data_noise-train_data_noise.min()) /
                    (train_data_noise.max()-train_data_noise.min()))
"""
train_data_noise = train_data + noise_level/255 * \
    np.random.RandomState(42).randn(train_data.shape[0], train_data.shape[1])
train_data_noise = ((train_data_noise-train_data_noise.min()) /
                    (train_data_noise.max()-train_data_noise.min()))

#load_fn = 'abu-airport-2.mat'  # 100*100*205
#load_fn = 'abu-airport-2.mat'  # 100*100*205
load_fn = 'abu-urban-5.mat'  # 100*100*205
load_data = sio.loadmat(load_fn)
test_data = load_data['data']
test_data = np.array(test_data)
test_data_shape = test_data.shape
test_data = test_data.reshape([test_data_shape[0]*test_data_shape[1], 205])
test_data = ((test_data-test_data.min()) /
             (test_data.max()-test_data.min()))
"""
load_fn = 'abu-airport-2_noise_25.mat'  # 100*100*205
# load_fn = 'abu-airport-3_noise_25.mat'  # 100*100*205
# load_fn = 'abu-urban-5_noise_25.mat'  # 100*100*205
load_data = sio.loadmat(load_fn)
test_data_noise = load_data['Im']
test_data_noise = np.array(test_data_noise)
test_data_noise = test_data_noise.reshape(
    [test_data_shape[0]*test_data_shape[1], 205])
test_data_noise = ((test_data_noise-test_data_noise.min()) /
                   (test_data_noise.max()-test_data_noise.min()))
"""
test_data_noise = test_data + noise_level/255*np.random.RandomState(42).randn(test_data.shape[0],test_data.shape[1])
test_data_noise = 2*((test_data_noise-test_data_noise.min()) /
                   (test_data_noise.max()-test_data_noise.min()))-1
test_data = 2*test_data-1
plt.figure(1)
plt.figure(1).suptitle("abu-airport-1")
for i in range(10):
    plt.subplot(2, 5, i+1)
    image = train_data[:, i*15]
    image = image.reshape([train_data_shape[0], train_data_shape[1]])
    #image = image*(image_max-image_min)+image_min
    plt.imshow(image)
    plt.axis('off')
    plt.title("band%d" % (i*15+1))

plt.figure(2)
for i in range(10):
    plt.subplot(2, 5, i+1)
    image = train_data_noise[:, i*15]
    image = image.reshape([train_data_shape[0], train_data_shape[1]])
    #image = image*(image_max-image_min)+image_min
    plt.imshow(image)
    plt.axis('off')
    plt.title("band%d" % (i*15+1))
orig_psnr = psnr(train_data, train_data_noise)
orig = train_data.reshape([train_data_shape[0], train_data_shape[1], 205])
test = train_data_noise.reshape(
    [train_data_shape[0], train_data_shape[1], 205])
corr = compare_ssim(orig, test)
plt.figure(2).suptitle(
    "abu-airport-1_with_noise\npsnr:%f\nssim%f" % (orig_psnr, corr))

plt.figure(3)
reproduce_result = sess.run(
    y, feed_dict={x_hat: test_data_noise, keep_prob: 1})
for i in range(10):
    plt.subplot(2, 5, i+1)
    image = reproduce_result[:, i*15]
    image = image.reshape([test_data_shape[0], test_data_shape[1]])
    plt.imshow(image)
    plt.axis('off')
    plt.title("band%d" % (i*15+1))
reconstact_psnr = psnr(test_data, reproduce_result)
orig = test_data.reshape([test_data_shape[0], test_data_shape[1], 205])
reproduce_result = np.float64(reproduce_result.reshape(
    [test_data_shape[0], test_data_shape[1], 205]))

save_fn = 'test.mat'
save_array = reproduce_result
sio.savemat(save_fn, {'array': save_array})

corr = compare_ssim(orig, reproduce_result)
plt.figure(3).suptitle(
    "test_data_denoise\npsnr:%f\nssim:%f" % (reconstact_psnr, corr))


plt.figure(4)
for i in range(10):
    plt.subplot(2, 5, i+1)
    image = test_data[:, 15*i]
    image = image.reshape([test_data_shape[0], test_data_shape[1]])
    #image = image*(image_max-image_min)+image_min
    plt.imshow(image)
    plt.axis('off')
    plt.title("band%d" % (i*15+1))
plt.figure(4).suptitle("test_target")
plt.show()
