import tensorflow as tf
import tensorflow.keras.backend as K

def imbalanced_bce_bayesian():
    def loss(y_true, y_pred):
        y_true = K.cast(y_true, tf.float32)
        y_pred = K.cast(y_pred, tf.float32)
        prior_pos = K.mean(y_true, axis=-1, keepdims=True)
        prior_neg = K.mean(1-y_true, axis=-1, keepdims=True)
        weight = y_true / (tf.broadcast_to(prior_pos, tf.shape(y_true)) + K.epsilon()) + \
                 (1-y_true) / (tf.broadcast_to(prior_neg, tf.shape(y_true)) + K.epsilon())
        y_pred = y_pred * weight
        return tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return loss


def imbalanced_bce_resampling():
    def loss(y_true, y_pred):
        y_true = K.cast(y_true, tf.float32)
        y_pred = K.cast(y_pred, tf.float32)
        prior_pos = K.mean(y_true, axis=-1, keepdims=True)
        prior_neg = K.mean(1-y_true, axis=-1, keepdims=True)
        weight = y_true / (tf.broadcast_to(prior_pos, tf.shape(y_true)) + K.epsilon()) + \
                 (1-y_true) / (tf.broadcast_to(prior_neg, tf.shape(y_true)) + K.epsilon())
        y_true = y_true * weight
        return tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return loss


def exp_interval_loss():
    def loss(y_true, y_pred):
        y_true = K.cast(y_true, tf.float32)
        y_pred = K.cast(y_pred, tf.float32)
        y_pred = K.epsilon() + y_pred
        logp = K.log(y_pred) * y_true
        return -2*512*K.mean(logp, axis=[-1, -2])
    return loss