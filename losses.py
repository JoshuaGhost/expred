import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
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


def rnr_matrix_loss():
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1-K.epsilon())
        start_pred = y_pred[:, :, :1]
        end_pred = y_pred[:, :, 1:]
        start_true = y_true - (tf.concat([tf.zeros((tf.shape(y_true)[0], 1, 1)), y_true[:,:-1,:]], axis=1))
        start_true = tf.clip_by_value(start_true, 0, 1)
        end_true = y_true - (tf.concat([y_true[:, 1:, :], tf.zeros((tf.shape(y_true)[0], 1, 1))], axis=1))
        end_true = tf.clip_by_value(end_true, 0, 1)
        end_true = tf.tile(end_true, (1, 1, tf.shape(y_true)[1]))
        end_true = tf.transpose(end_true, perm=[0,2,1])
        #end_true = tf.linalg.set_diag(end_true, tf.zeros((tf.shape(y_true)[0], tf.shape(y_true)[1])))
        end_true = tf.linalg.band_part(end_true, 0, -1) 
        end_true = tf.tile(start_true, (1, 1, tf.shape(y_true)[1])) * end_true
        end_true = tf.math.cumsum(end_true, axis=1, reverse=True)
        end_true = tf.clip_by_value(end_true, 0, 1)
        end_true = end_true - (tf.concat([end_true[:, 1:, :], tf.zeros((tf.shape(y_true)[0], 1, tf.shape(y_true)[1]))], axis=1))
        end_true = tf.clip_by_value(end_true, 0, 1)
        loss_start = - start_true * tf.math.log(start_pred) - (1-start_true)*tf.math.log(1-start_pred)
        loss_end = - tf.math.log(end_pred) * end_true
        loss = tf.reduce_sum(loss_start, axis=[1,2]) + tf.reduce_sum(tf.reduce_sum(loss_end, axis=1), axis=-1)
        return tf.reduce_mean(loss)
    return loss