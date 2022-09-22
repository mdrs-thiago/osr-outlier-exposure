"""
A weighted version of categorical_crossentropy for keras (2.0.6). This lets you apply a weight to unbalanced classes.
@url: https://gist.github.com/wassname/ce364fddfc8a025bfab4348cf5de852d
@author: wassname
"""
from tensorflow.keras import backend as K
import tensorflow as tf
import sys, pdb

def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    
    return loss

def weighted_categorical_crossentropy_ignoring_last_label(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        y_pred = K.reshape(y_pred, (-1, K.int_shape(y_pred)[-1]))
        log_softmax = tf.nn.log_softmax(y_pred)
        #log_softmax = tf.log(y_pred)
        #log_softmax = K.log(y_pred)

        y_true = K.one_hot(tf.to_int32(K.flatten(y_true)), K.int_shape(y_pred)[-1]+1)
        unpacked = tf.unstack(y_true, axis=-1)
        y_true = tf.stack(unpacked[:-1], axis=-1)

        cross_entropy = -K.sum(y_true * log_softmax * weights , axis=1)
        loss = K.mean(cross_entropy)

        return loss
    
    return loss
# 

def categorical_focal_ignoring_last_label(alpha=0.25,gamma=2):
    """
    Focal loss implementation
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    #alpha = K.variable(alpha)
    #gamma = K.variable(gamma)
    def loss(y_true, y_pred):
        y_pred = K.reshape(y_pred, (-1, K.int_shape(y_pred)[-1]))
        y_pred_softmax = tf.nn.softmax(y_pred) # I should do softmax before the loss
        #log_softmax = tf.nn.log_softmax(y_pred)
        #log_softmax = tf.log(y_pred)
        #log_softmax = K.log(y_pred)
        y_true = K.one_hot(tf.cast(K.flatten(y_true), tf.int32), K.int_shape(y_pred)[-1]+1)
        unpacked = tf.unstack(y_true, axis=-1)
        y_true = tf.stack(unpacked[:-1], axis=-1)
        focal_term = alpha * K.pow(1. - y_pred_softmax, gamma)
        cross_entropy = -K.sum(focal_term * y_true * K.log(y_pred_softmax), axis=1)
        loss = K.mean(cross_entropy)

        return loss
    
    return loss

def outlier_exposure_loss(OE_weight=0.5,alpha=0.25,gamma=2):
  
  def loss(y_true, y_pred):
    '''        
    yy_true = y_true.numpy()
    yy_pred = y_pred.numpy()

    true_idx = np.where(K.max(yy_true,axis=-1)[:,0,0] == 1)
    
    bs = yy_true.shape[0]

    all_idx = np.arange(0,bs)
    mask_idx = all_idx[np.isin(all_idx, true_idx, invert=True)]


    y_pred = K.reshape(y_pred, (-1, K.int_shape(y_pred)[-1]))

    y_pred_softmax = tf.nn.softmax(y_pred)
    y_pred_ = tf.gather(y_pred_softmax, true_idx)
    y_pred_OE = tf.gather(y_pred_softmax,mask_idx)
    
    y_true_ = tf.gather(y_true,true_idx)
    y_true_OE = tf.gather(y_true,mask_idx)

    
    y_true = K.one_hot(tf.to_int32(K.flatten(y_true)), K.int_shape(y_pred)[-1]+1)
    unpacked = tf.unstack(y_true, axis=-1)
    y_true = tf.stack(unpacked[:-1], axis=-1)


    CE = -K.sum(y_true_ * K.log(y_pred_), axis=1)
    
    CE_OE = -K.sum(y_true_OE * K.log(y_pred_OE), axis=1)

    loss = K.mean(CE) + OE_weight * K.mean(CE_OE)
    '''

    y_pred = K.reshape(y_pred, (-1, K.int_shape(y_pred)[-1]))
    y_pred_softmax = tf.nn.softmax(y_pred) # I should do softmax before the loss
    
    #log_softmax = tf.nn.log_softmax(y_pred)
    #log_softmax = tf.log(y_pred)
    #log_softmax = K.log(y_pred)
    #tf.print(y_pred)
    #print(y_pred.shape)
    #tf.print(tf.unique_with_counts(K.flatten(y_true) + 1))
    #y_true = K.one_hot(tf.cast(K.flatten(y_true), tf.int32), K.int_shape(y_pred)[-1]+1)
    #tf.print(y_true, summarize=-1)

    y_true = tf.one_hot(tf.cast(K.flatten(y_true), tf.int32), y_pred.shape[-1] + 2)
    #tf.print(y_true.shape)
    #tf.print(y_true,  summarize=-1)
    #tf.print(y_true[:5,:])
    #print('****')
    #tf.print(y_true[-5:,:])
    
    unpacked = tf.unstack(y_true, axis=-1)
    
    #print('@'*500)
    #tf.print(unpacked)
    #print(len(unpacked))

    y_true = tf.stack(unpacked[:-2], axis=-1)
    #print('@'*500)
    #print(y_true.shape[1])
    focal_term = alpha * K.pow(1. - y_pred_softmax, gamma)
    cross_entropy = -K.sum(focal_term * y_true * K.log(y_pred_softmax), axis=1)
    s_loss = K.mean(cross_entropy)
    #tf.print(y_true)
    _y_true = tf.stack(unpacked[-1:], axis=-1)
    #tf.print(_y_true[-5:])
    
    #tf.print(_y_true.shape)
    #np.set_printoptions(threshold=np.inf)

    #print(_y_true)
    #print('********')
    #print(y_true)
    #OE_y_true = tf.repeat(_y_true, repeats=[0,y_pred.shape[1]], axis=-1)
    repeat = tf.constant([1,y_pred.shape[1]])
    OE_y_true = tf.scalar_mul(1/y_pred.shape[1], tf.tile(_y_true, repeat))
    #OE_y_true = tf.tile(_y_true, repeat)
    
    #tf.print(OE_y_true)
    #tf.print(no_zeros.shape)
    
    #_y_pred = tf.boolean_mask(y_pred_softmax, OE_y_true)
    
    #tf.print(OE_y_true[:5,:])
    #print('****')
    #tf.print(OE_y_true[-5:,:])
    #tf.print(OE_y_true.shape)
    
    #OE_cross_entropy = -K.sum(OE_y_true * K.log(y_pred_softmax), axis=1)
    #y_pred_masked = tf.multiply(OE_y_true, y_pred_softmax)
    #tf.print(y_pred)
    #_log = K.log(y_pred_masked)

    #_log = tf.where(tf.math.is_finite(_log), _log, tf.zeros_like(_log))
    #tf.print(tf.unique(K.flatten(_log)))


    #OE_cross_entropy = -K.sum(y_pred_masked * _log, axis=1)
    #OE_loss = K.mean(OE_cross_entropy)
    OE_loss = K.mean(tf.keras.losses.kullback_leibler_divergence(OE_y_true, y_pred_softmax))
    #OE_loss = tf.where(tf.math.is_finite(OE_loss), OE_loss, tf.zeros_like(OE_loss))
    tf.print(OE_loss)

    tf.print(s_loss)
    #print('@'*500)
    #print(OE_y_true.shape[1])
    #print(tf.executing_eagerly())
    

    #tf.print(f'\n OE loss = ',output_stream=sys.stderr)
    #tf.print(OE_loss)
    #print(OE_loss)

    loss = s_loss + OE_weight*OE_loss

    return loss

  return loss

def weighted_categorical_focal_ignoring_last_label(weights, alpha=0.25,gamma=2):
    """
    Focal loss implementation
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    weights = K.variable(weights)
    def loss(y_true, y_pred):
        y_pred = K.reshape(y_pred, (-1, K.int_shape(y_pred)[-1]))
        y_pred_softmax = tf.nn.softmax(y_pred) # I should do softmax before the loss

        y_true = K.one_hot(tf.to_int32(K.flatten(y_true)), K.int_shape(y_pred)[-1]+1)
        unpacked = tf.unstack(y_true, axis=-1)
        y_true = tf.stack(unpacked[:-1], axis=-1)
        focal_term = alpha * K.pow(1. - y_pred_softmax, gamma)
        cross_entropy = -K.sum(focal_term * y_true * K.log(y_pred_softmax) * weights, axis=1)
        loss = K.mean(cross_entropy)

        return loss
    
    return loss
def softmax_sparse_crossentropy_ignoring_last_label(y_true, y_pred):
    y_pred = K.reshape(y_pred, (-1, K.int_shape(y_pred)[-1]))
    log_softmax = tf.nn.log_softmax(y_pred)

    y_true = K.one_hot(tf.to_int32(K.flatten(y_true)), K.int_shape(y_pred)[-1]+1)
    unpacked = tf.unstack(y_true, axis=-1)
    y_true = tf.stack(unpacked[:-1], axis=-1)

    cross_entropy = -K.sum(y_true * log_softmax, axis=1)
    cross_entropy_mean = K.mean(cross_entropy)

    return cross_entropy_mean

def sparse_accuracy_ignoring_last_label(y_true, y_pred):
    #y_pred = K.argmax(y_pred,axis=3)
        
    nb_classes = K.int_shape(y_pred)[-1]
    y_pred = K.reshape(y_pred, (-1, nb_classes))

    y_true = K.one_hot(tf.to_int32(K.flatten(y_true)),
                       nb_classes + 1)
    unpacked = tf.unstack(y_true, axis=-1)
    legal_labels = ~tf.cast(unpacked[-1], tf.bool)
    y_true = tf.stack(unpacked[:-1], axis=-1)

    return K.sum(tf.to_float(legal_labels & K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)))) / K.sum(tf.to_float(legal_labels))

import numpy as np
from tensorflow.keras.activations import softmax
from tensorflow.keras.losses import categorical_crossentropy

# # init tests
# samples=3
# maxlen=4
# vocab=5

# y_pred_n = np.random.random((samples,maxlen,vocab)).astype(K.floatx())
# y_pred = K.variable(y_pred_n)
# y_pred = softmax(y_pred)

# y_true_n = np.random.random((samples,maxlen,vocab)).astype(K.floatx())
# y_true = K.variable(y_true_n)
# y_true = softmax(y_true)

# # test 1 that it works the same as categorical_crossentropy with weights of one
# weights = np.ones(vocab)

# loss_weighted=weighted_categorical_crossentropy(weights)(y_true,y_pred).eval(session=K.get_session())
# loss=categorical_crossentropy(y_true,y_pred).eval(session=K.get_session())
# np.testing.assert_almost_equal(loss_weighted,loss)
# print('OK test1')


# # test 2 that it works differen't than categorical_crossentropy with weights of less than one
# weights = np.array([0.1,0.3,0.5,0.3,0.5])

# loss_weighted=weighted_categorical_crossentropy(weights)(y_true,y_pred).eval(session=K.get_session())
# loss=categorical_crossentropy(y_true,y_pred).eval(session=K.get_session())
# np.testing.assert_array_less(loss_weighted,loss)
# print('OK test2')

# # same keras version as I tested it on?
# import keras
# assert keras.__version__.split('.')[:2]==['2', '0'], 'this was tested on keras 2.0.6 you have %s' % keras.__version
# print('OK version')
