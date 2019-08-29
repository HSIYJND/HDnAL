import tensorflow as tf
import numpy as np

def SAD(y_true, y_pred):
    with tf.variable_scope("SID"):
        x = y_true  # 200*205
        y = y_pred  # 200*205
        y = tf.transpose(y)  # 205*200
        A = tf.matmul(x, y)  # 200*200
        A = tf.diag_part(A)  # 200*1
        B = tf.norm(x, ord='euclidean', axis=-1)
        C = tf.norm(y, ord='euclidean', axis=0)
        defen = tf.div(A, 1e-6+B*C)
        return ((1-defen))


def SID(x, y):
    with tf.variable_scope("SID"):
        """
        # x   # pixels*bands
        # y   
        """
        #x = tf.transpose(x)
        #y = tf.transpose(y)
        p_sum = tf.reduce_sum(x, axis=1)  # 200*1
        q_sum = tf.reduce_sum(y, axis=1)  # 200*1
        p_sum = tf.expand_dims(p_sum, 1)
        q_sum = tf.expand_dims(q_sum, 1)
        print(p_sum.get_shape())
        print(x.get_shape()[0])
        p = tf.div(x, 1e-6+tf.matmul(p_sum, tf.ones(
            [1, x.get_shape()[1]])))  # 200*205
        q = tf.div(y, 1e-6+tf.matmul(q_sum, tf.ones(
            [1, y.get_shape()[1]])))  # 200*205
        #p = tf.div(x, tf.matmul(p_sum, tf.ones(
        #    [1, x.get_shape()[1]])))  # 200*205
        #q = tf.div(y, tf.matmul(q_sum, tf.ones(
        #    [1, y.get_shape()[1]])))  # 200*205
        D_x_y = tf.reduce_sum(p*tf.log(1e-8+p/(q+1e-6)), axis=1)  # 200*1
        D_y_x = tf.reduce_sum(q*tf.log(1e-8+q/(p+1e-6)), axis=1)  # 200*1
        print(D_x_y.get_shape())
        return tf.add(D_x_y, D_y_x, name='resu')  # 200*1


def OPD(ri, rj):
    with tf.variable_scope("OPD"):
        """
        ri   # pixels*bands
        rj   # pixels*bands
        """
        ri = tf.transpose(ri) #205*
        rj = tf.transpose(rj) #205*
        L = ri.get_shape()[0]  # 205
        I = tf.eye(int(L))  # 205*205
        # 205*205  205                     205*                                    *205     205*         *205
        
        
        Pri_perp = I - tf.matmul(tf.matmul(ri, tf.matrix_inverse(
            tf.matmul(tf.transpose(ri), ri))), tf.transpose(ri))
        Prj_perp = I - tf.matmul(tf.matmul(rj, tf.matrix_inverse(
            tf.matmul(tf.transpose(rj), rj))), tf.transpose(rj))
        # ***                           *205        205*205  205*
        val = tf.matmul(tf.matmul(tf.transpose(ri), Prj_perp), ri) + \
            tf.matmul(tf.matmul(tf.transpose(rj), Pri_perp), rj)
        OPD = tf.sqrt(tf.abs(val)+1e-8)

        OPD = tf.diag_part(OPD)
        print(OPD.get_shape())
        return val
# MLP as encoder

# MLP as encoder


def MLP_encoder(x, n_hidden, n_output, keep_prob):
    with tf.variable_scope("MLP_encoder"):
        # initializers
        w_init = tf.contrib.layers.xavier_initializer()
        b_init = tf.constant_initializer(0.)

        # 1st hidden layer
        w0 = tf.get_variable(
            'w0', [x.get_shape()[1], n_hidden], initializer=w_init)
        b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
        h0 = tf.matmul(x, w0) + b0
        h0 = tf.nn.leaky_relu(h0, alpha=0.1)
        h0 = tf.nn.dropout(h0, keep_prob)

        # 2nd hidden layer
        w1 = tf.get_variable(
            'w1', [h0.get_shape()[1], n_hidden], initializer=w_init)
        b1 = tf.get_variable('b1', [n_hidden], initializer=b_init)
        h1 = tf.matmul(h0, w1) + b1
        h1 = tf.nn.leaky_relu(h1, alpha=0.1)
        h1 = tf.nn.dropout(h1, keep_prob)

        # output layer
        wo = tf.get_variable(
            'wo', [h1.get_shape()[1], n_output], initializer=w_init)
        bo = tf.get_variable('bo', [n_output], initializer=b_init)
        output = tf.matmul(h1, wo) + bo

    return output

# MLP as decoder


def MLP_decoder(z, n_hidden, n_output, keep_prob, reuse=False):

    with tf.variable_scope("MLP_decoder", reuse=reuse):
        # initializers
        w_init = tf.contrib.layers.xavier_initializer()
        b_init = tf.constant_initializer(0.)

        # 1st hidden layer
        w0 = tf.get_variable(
            'w0', [z.get_shape()[1], n_hidden], initializer=w_init)
        b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
        h0 = tf.matmul(z, w0) + b0
        h0 = tf.nn.leaky_relu(h0, alpha=0.1)
        h0 = tf.nn.dropout(h0, keep_prob)

        # 2nd hidden layer
        w1 = tf.get_variable(
            'w1', [h0.get_shape()[1], n_hidden], initializer=w_init)
        b1 = tf.get_variable('b1', [n_hidden], initializer=b_init)
        h1 = tf.matmul(h0, w1) + b1
        h1 = tf.nn.leaky_relu(h1, alpha=0.1)
        h1 = tf.nn.dropout(h1, keep_prob)

        # output layer
        wo = tf.get_variable(
            'wo', [h1.get_shape()[1], n_output], initializer=w_init)
        bo = tf.get_variable('bo', [n_output], initializer=b_init)
        y = tf.tanh(tf.matmul(h1, wo) + bo, name='decode_output')

    return y

# Discriminator


def discriminator(z, n_hidden, n_output, keep_prob, reuse=False):

    with tf.variable_scope("discriminator", reuse=reuse):
        # initializers
        w_init = tf.contrib.layers.xavier_initializer()
        b_init = tf.constant_initializer(0.)

        # 1st hidden layer
        w0 = tf.get_variable(
            'w0', [z.get_shape()[1], n_hidden], initializer=w_init)
        b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
        h0 = tf.matmul(z, w0) + b0
        h0 = tf.nn.leaky_relu(h0, alpha=0.1)
        h0 = tf.nn.dropout(h0, keep_prob)

        # 2nd hidden layer
        w1 = tf.get_variable(
            'w1', [h0.get_shape()[1], n_hidden], initializer=w_init)
        b1 = tf.get_variable('b1', [n_hidden], initializer=b_init)
        h1 = tf.matmul(h0, w1) + b1
        h1 = tf.nn.leaky_relu(h1, alpha=0.1)
        h1 = tf.nn.dropout(h1, keep_prob)

        # output layer
        wo = tf.get_variable(
            'wo', [h1.get_shape()[1], n_output], initializer=w_init)
        bo = tf.get_variable('bo', [n_output], initializer=b_init)
        y = tf.matmul(h1, wo) + bo

    return tf.sigmoid(y), y

# Gateway


def adversarial_autoencoder(x_hat, x, z_sample, dim_img, dim_z, keep_prob):
    lambda_1 = 0.0016
    lambda_2 = 0
    lambda_3 = 0
    # encoding

    z = MLP_encoder(x_hat, 500, dim_z, keep_prob)

    # decoding
    y = MLP_decoder(z, 500, dim_img, keep_prob)

    # loss
    marginal_likelihood = tf.reduce_mean(tf.square(
        y-x), axis=1) + lambda_1*(SAD(y, x)) #+ lambda_2*SID(y, x)#+lambda_3*OPD(y, x)

    #marginal_likelihood = tf.reduce_mean(tf.square(
    #    y-x), axis=1) + lambda_1*(SAD(y, x)) + lambda_2*SID(y, x)#+lambda_3*OPD(y, x)

    # GAN Loss
    z_real = z_sample
    z_fake = z
    D_real, D_real_logits = discriminator(z_real, 1000, 1, keep_prob)
    D_fake, D_fake_logits = discriminator(
        z_fake, 1000, 1, keep_prob, reuse=True)

    # discriminator loss
    D_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones_like(D_real_logits)))
    D_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros_like(D_fake_logits)))
    D_loss = 0.5*(D_loss_real+D_loss_fake)

    # generator loss
    G_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones_like(D_fake_logits)))

    marginal_likelihood = tf.reduce_mean(marginal_likelihood)
    D_loss = tf.reduce_mean(D_loss)
    G_loss = tf.reduce_mean(G_loss)

    return y, z, marginal_likelihood, D_loss, G_loss


def decoder(z, dim_img, n_hidden):

    y = MLP_decoder(z, n_hidden, dim_img, 1.0, reuse=True)

    return y
