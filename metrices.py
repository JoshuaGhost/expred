from tensorflow.keras import backend as K
import tensorflow as tf
from config import *

def precision_wrapper(exp_output):
    def precision(y_true, y_pred):
        def convert_intervals_to_ration_mask(interval):
            p = K.cumsum(interval, axis=1)
            rations = K.clip(p[:, :, 0] - p[:, :, 1] + interval[:,:,1], 0, 1)
            return rations 
        def convert_logits_to_ration_mask(pred):
            max_start_arg = K.argmax(pred[:, :, 0])
            start = tf.one_hot([max_start_arg], depth=MAX_SEQ_LENGTH)
            start = tf.transpose(start, [1, 2, 0])
            max_end_arg = K.argmax(pred[:, :, 1])
            end = tf.one_hot([max_end_arg], depth=MAX_SEQ_LENGTH)
            end = tf.transpose(end, [1, 2, 0])
            interval = K.concatenate([start, end], axis=-1)
            interval = tf.dtypes.cast(interval, tf.float32)
            rations = convert_intervals_to_ration_mask(interval)
            return rations
        if exp_output == 'interval':
            y_true = convert_intervals_to_ration_mask(y_true)
            y_pred = convert_logits_to_ration_mask(y_pred)
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    return precision
 
def recall_wrapper(exp_output):
    def recall(y_true, y_pred):
        def convert_intervals_to_ration_mask(interval):
            p = K.cumsum(interval, axis=1)
            rations = K.clip(p[:, :, 0] - p[:, :, 1] + interval[:,:,1], 0, 1)
            return rations 
        def convert_logits_to_ration_mask(pred):
            max_start_arg = K.argmax(pred[:, :, 0])
            start = tf.one_hot([max_start_arg], depth=MAX_SEQ_LENGTH)
            start = tf.transpose(start, [1, 2, 0])
            max_end_arg = K.argmax(pred[:, :, 1])
            end = tf.one_hot([max_end_arg], depth=MAX_SEQ_LENGTH)
            end = tf.transpose(end, [1, 2, 0])
            interval = K.concatenate([start, end], axis=-1)
            interval = tf.dtypes.cast(interval, tf.float32)
            rations = convert_intervals_to_ration_mask(interval)
            return rations
        if exp_output == 'interval':
            y_true = convert_intervals_to_ration_mask(y_true)
            y_pred = convert_logits_to_ration_mask(y_pred)
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
    return recall

def f1_wrapper(exp_output):
    def f1(y_true, y_pred):
        def precision(y_true, y_pred):
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + K.epsilon())
            return precision
        def recall(y_true, y_pred):
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            recall = true_positives / (possible_positives + K.epsilon())
            return recall
        def convert_intervals_to_ration_mask(interval):
            p = K.cumsum(interval, axis=1)
            rations = K.clip(p[:, :, 0] - p[:, :, 1] + interval[:,:,1], 0, 1)
            return rations 
        def convert_logits_to_ration_mask(pred):
            max_start_arg = K.argmax(pred[:, :, 0])
            start = tf.one_hot([max_start_arg], depth=MAX_SEQ_LENGTH)
            start = tf.transpose(start, [1, 2, 0])
            max_end_arg = K.argmax(pred[:, :, 1])
            end = tf.one_hot([max_end_arg], depth=MAX_SEQ_LENGTH)
            end = tf.transpose(end, [1, 2, 0])
            interval = K.concatenate([start, end], axis=-1)
            interval = tf.dtypes.cast(interval, tf.float32)
            rations = convert_intervals_to_ration_mask(interval)
            return rations
        if exp_output == 'interval':
            y_true = convert_intervals_to_ration_mask(y_true)
            y_pred = convert_logits_to_ration_mask(y_pred)
        precision = precision(y_true, y_pred)
        recall = recall(y_true, y_pred)
        return 2*((precision*recall)/(precision+recall+K.epsilon()))
    return f1

def iou(y_true, y_pred):
    intersection = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    union = K.sum(K.round(K.clip(y_true + y_pred, 0, 1)))
    iou = intersection / (union + K.epsilon())
    return iou