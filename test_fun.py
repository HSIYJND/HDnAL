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

a = np.array([[0,0,0]])
b = np.array([[1,4,7]])
print(a)
print(b)
x = tf.placeholder(tf.float32,[None,3])
y = tf.placeholder(tf.float32,[None,3])
z = aae.OPD(x,y)
sess = tf.Session()

print(sess.run(z,feed_dict={x:a,y:-b}))