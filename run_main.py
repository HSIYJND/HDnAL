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


def psnr(target, reconstract, scale):
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


def main():
    """ parameters """
    # network architecture
    noise_level=35

    dim_img = 205  # number of pixels for a MNIST image
    dim_z = 50                      # to visualize learned manifold

    # train
    n_epochs = 20000
    batch_size = 500
    learn_rate = 1e-4

    """ prepare data """
    # 读取数据集
    load_fn = 'abu-airport-1.mat'  # 100*100*205
    load_data = sio.loadmat(load_fn)
    load_matrix = load_data['data']
    load_matrix = np.array(load_matrix)
    load_matrix = load_matrix.reshape(
        [load_matrix.shape[0]*load_matrix.shape[1], 205])
    load_matrix = ((load_matrix-load_matrix.min()) /
                   (load_matrix.max()-load_matrix.min()))
    x_target = load_matrix
    """
    load_fn = 'abu-airport-1_noise_25.mat'  # 100*100*205
    load_data = sio.loadmat(load_fn)
    load_matrix_noise = load_data['Im']
    load_matrix_noise = np.array(load_matrix_noise)
    load_matrix_noise = load_matrix_noise.reshape(
        [load_matrix_noise.shape[0]*load_matrix_noise.shape[1], 205])
    load_matrix_noise = ((load_matrix_noise-load_matrix_noise.min()) /
                         (load_matrix_noise.max()-load_matrix_noise.min()))
    x_input = load_matrix_noise
    """
    x_input = x_target + noise_level/255*np.random.RandomState(42).randn(x_target.shape[0],x_target.shape[1])
    x_input = 2*((x_input-x_input.min()) /
               (x_input.max()-x_input.min()))-1
    x_target = 2*x_target-1
    # input placeholders
    # In denoising-autoencoder, x_hat == x + noise, otherwise x_hat == x
    x_hat = tf.placeholder(tf.float32, shape=[None, dim_img], name='input_img')
    x = tf.placeholder(tf.float32, shape=[None, dim_img], name='target_img')

    # dropout
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    # input for PMLR

    # samples drawn from prior distribution
    z_sample = tf.placeholder(
        tf.float32, shape=[None, dim_z], name='prior_sample')

    # network architecture
    y, z, R_loss, D_loss, G_loss = aae.adversarial_autoencoder(
        x_hat, x, z_sample, dim_img, dim_z, keep_prob)

    # optimization
    t_vars = tf.trainable_variables()
    di_vars = [var for var in t_vars if "discriminator" in var.name]
    de_vars = [var for var in t_vars if "MLP_decoder" in var.name]
    g_vars = [var for var in t_vars if "MLP_encoder" in var.name]
    ae_vars = g_vars + de_vars

    train_op_ae = tf.train.AdamOptimizer(
        learn_rate).minimize(R_loss, var_list=ae_vars)
    train_op_d = tf.train.AdamOptimizer(
        learn_rate/10).minimize(D_loss, var_list=di_vars)
    train_op_g = tf.train.AdamOptimizer(
        learn_rate).minimize(G_loss, var_list=g_vars)

    """ training """

    load_fn = 'abu-urban-5.mat'  # 100*100*205
    load_data = sio.loadmat(load_fn)
    valid_target_1 = load_data['data']
    valid_target_1 = np.array(valid_target_1)
    valid_target_1 = valid_target_1.reshape([100*100, 205])
    valid_target_1 = ((valid_target_1-valid_target_1.min()) /
                      (valid_target_1.max()-valid_target_1.min()))
    reproduce_image = valid_target_1
    for i in range(10):
        plt.subplot(2, 5, i+1)
        image = reproduce_image[:, 15*i]
        image = image.reshape([100, 100])
        plt.imshow(image)
    plt.title('target')
    plt.savefig('./reproduceresult/target.png')
    valid_1 = valid_target_1 + noise_level/255*np.random.RandomState(42).randn(
        valid_target_1.shape[0], valid_target_1.shape[1])
    valid_1 = (valid_1-valid_1.min()) / \
        (valid_1.max()-valid_1.min())

    load_fn = 'abu-airport-2.mat'  # 100*100*205
    load_data = sio.loadmat(load_fn)
    valid_target_2 = load_data['data']
    valid_target_2 = np.array(valid_target_2)
    valid_target_2 = valid_target_2.reshape([100*100, 205])
    valid_target_2 = ((valid_target_2-valid_target_2.min()) /
                      (valid_target_2.max()-valid_target_2.min()))

    valid_2 = valid_target_2 + noise_level/255*np.random.RandomState(42).randn(
        valid_target_2.shape[0], valid_target_2.shape[1])
    valid_2 = (valid_2-valid_2.min()) / \
        (valid_2.max()-valid_2.min())

    load_fn = 'abu-airport-3.mat'  # 100*100*205
    load_data = sio.loadmat(load_fn)
    valid_target_3 = load_data['data']
    valid_target_3 = np.array(valid_target_3)
    valid_target_3 = valid_target_3.reshape([100*100, 205])
    valid_target_3 = ((valid_target_3-valid_target_3.min()) /
                      (valid_target_3.max()-valid_target_3.min()))

    valid_3 = valid_target_3 + noise_level/255*np.random.RandomState(42).randn(
        valid_target_3.shape[0], valid_target_3.shape[1])
    valid_3 = (valid_3-valid_3.min()) / \
        (valid_3.max()-valid_3.min())

    saver = tf.train.Saver()
    # train
    with tf.Session() as sess:

        writer = tf.summary.FileWriter("logs/", sess.graph)
        sess.run(tf.global_variables_initializer(), feed_dict={keep_prob: 1})

        rand_x = np.random.RandomState(42)
        rand_y = np.random.RandomState(42)
        past = datetime.now()
        for epoch in range(n_epochs):

            # lr decay

            # Random shuffling
            rand_x.shuffle(x_input)
            rand_y.shuffle(x_target)

            # Loop over all batches
            for batch in np.arange(int(len(x_input) / batch_size)):
                # Compute the offset of the current minibatch in the data.
                start = int(batch * batch_size)
                end = int(start + batch_size)
                batch_xs_input = x_input[start:end]
                batch_xs_target = x_target[start:end]

                # draw samples from prior distribution
                samples = np.random.randn(batch_size, dim_z)
                z_id_one_hot_vector = np.ones((batch_size, 1))

                # reconstruction loss
                _, loss_likelihood = sess.run(
                    (train_op_ae, R_loss),
                    feed_dict={x_hat: batch_xs_input, x: batch_xs_target, z_sample: samples, keep_prob: 1})

                # discriminator loss
                _, d_loss = sess.run(
                    (train_op_d, D_loss),
                    feed_dict={x_hat: batch_xs_input, x: batch_xs_target, z_sample: samples, keep_prob: 1})

                # generator loss
                for _ in range(1):
                    _, g_loss = sess.run(
                        (train_op_g, G_loss),
                        feed_dict={x_hat: batch_xs_input, x: batch_xs_target, z_sample: samples, keep_prob: 1})

            #tot_loss = loss_likelihood + d_loss + g_loss

            # print cost every epoch
            now = datetime.now()
            print("\nEpoch {}/{} - {:.1f}s".format(epoch,
                                                   n_epochs, (now - past).total_seconds()))
            print("Autoencoder Loss: {}".format(np.mean(loss_likelihood)))

            print("Discriminator Loss: {}".format(
                np.mean(d_loss)))
            print("Generator Loss: {}".format(np.mean(g_loss)))
            past = now
            if epoch % 5 == 0:
                reproduce_image, latent_image = sess.run(
                    (y, z), feed_dict={x_hat: 2*valid_3-1, x: 2*valid_target_3-1, keep_prob: 1})
                # for i in range(10):
                #    plt.subplot(2, 5, i+1)
                #    image = reproduce_image[:, 15*i]
                #    image = image.reshape([100, 100])
                #    plt.imshow(image)
                reconstact_psnr_1 = psnr(
                    valid_target_1, reproduce_image, [100, 100])
                #orig = np.float32(valid_target_1.reshape([100, 100, 205]))
                #reproduce_image = reproduce_image.reshape([100, 100, 205])
                #corr = compare_ssim(orig, reproduce_image)
                # plt.savefig('./reproduceresult/the%depochdenoise1psnr_%fs.png' %
                #            (epoch, reconstact_psnr_1))
                # for i in range(10):
                #    plt.subplot(2, 5, i+1)
                #    image = latent_image[:, 3*i]
                #    image = image.reshape([100, 100])
                #    plt.imshow(image)
                #plt.savefig('./latentresult/the%depoch_reproduce.png' % epoch)

                #reproduce_image, latent_image = sess.run((y, z), feed_dict={x_hat: valid_2, x: valid_target_2, keep_prob: 1})
                # for i in range(10):
                #    plt.subplot(2, 5, i+1)
                #    image = reproduce_image[:, 15*i]
                #    image = image.reshape([100, 100])
                #    plt.imshow(image)
                #reconstact_psnr_2 = psnr(valid_target_2, reproduce_image, [100, 100])
                #orig = np.float32(valid_target_2.reshape([100, 100, 205]))
                #reproduce_image = reproduce_image.reshape([100, 100, 205])
                #corr = compare_ssim(orig, reproduce_image)
                # plt.savefig('./reproduceresult/the%depochdenoise2psnr_%fs.png' %
                #            (epoch, reconstact_psnr_2))
                #reproduce_image, latent_image = sess.run((y, z), feed_dict={x_hat: valid_3, x: valid_target_3, keep_prob: 1})
                # for i in range(10):
                #    plt.subplot(2, 5, i+1)
                #    image = reproduce_image[:, 15*i]
                #    image = image.reshape([100, 100])
                #    plt.imshow(image)
                #reconstact_psnr_3 = psnr(valid_target_3, reproduce_image, [100, 100])
                #orig = np.float32(valid_target_3.reshape([100, 100, 205]))
                #reproduce_image = reproduce_image.reshape([100, 100, 205])
                #corr = compare_ssim(orig, reproduce_image)
                # plt.savefig('./reproduceresult/the%depochdenoise3psnr_%fs.png' %
                #            (epoch, reconstact_psnr_3))
                # if reconstact_psnr_1>38 and reconstact_psnr_2>38 and reconstact_psnr_3>38:
                #    print("\nSaving models...")
                #    saver.save(sess, './38/model')
                # if reconstact_psnr_1>38 and reconstact_psnr_2>38 and reconstact_psnr_3>37.5:
                #    print("\nSaving models...")
                #    saver.save(sess, './37/model')
                # if reconstact_psnr_1>38 and reconstact_psnr_2>37.5 and reconstact_psnr_3>37.5:
                #    print("\nSaving models...")
                #    saver.save(sess, './371/model')
                with open('./log.txt', 'a') as log:
                    log.write("Epoch: {}, iteration: {}\n".format(epoch, 0))
                    log.write("Autoencoder Loss: {}\n".format(
                        np.mean(loss_likelihood)))
                    log.write("Discriminator Loss: {}\n".format(
                        np.mean(d_loss)))
                    log.write("Generator Loss: {}\n".format(np.mean(g_loss)))
                    log.write("1:{}\n\n".format(reconstact_psnr_1))
                #            log.write("1:{}:2:{}:3:{}\n\n".format(reconstact_psnr_1,reconstact_psnr_2,reconstact_psnr_3))

            if epoch % 10 == 0:
                print("\nSaving models...")
                saver.save(sess, './model/model')
        print("\nSaving models...")
        saver.save(sess, './model/model')
        writer.close()


if __name__ == '__main__':
    # main
    main()
