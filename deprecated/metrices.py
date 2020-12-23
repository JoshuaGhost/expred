import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
from tensorflow.keras import backend as K
from config import *


def precision_wrapper(exp_output):
    def precision(y_true, y_pred):
        def convert_intervals_to_ration_mask(interval):
            p = K.cumsum(interval, axis=1)
            rations = K.clip(p[:, :, 0] - p[:, :, 1] + interval[:, :, 1], 0, 1)
            return rations

        def convert_rnr_matrix_to_ration_mask(y):
            starts = y[:, :, :1]
            ends_given_starts = y[:, :, 1:]
            starts = tf.math.round(tf.clip_by_value(starts, 0., 1.))
            ends_given_starts = tf.linalg.band_part(ends_given_starts, 0, -1)
            max_ends_arg = tf.math.argmax(ends_given_starts, axis=2)
            max_ends_arg = tf.cast(max_ends_arg, dtype=tf.int32)
            ends_given_starts = tf.math.multiply(tf.cast(tf.squeeze(starts), dtype=tf.int32), max_ends_arg)
            ends = tf.cast(tf.one_hot(ends_given_starts, depth=MAX_SEQ_LENGTH), dtype=tf.float32)
            ends = tf.math.cumsum(ends, axis=2, reverse=True)
            rations = tf.linalg.band_part(ends, 0, -1)
            rations = tf.math.multiply(tf.tile(starts, (1, 1, MAX_SEQ_LENGTH)), rations)
            rations = tf.math.reduce_sum(rations, axis=1, keepdims=True)
            rations = tf.clip_by_value(rations, 0., 1.)
            rations = tf.transpose(rations, perm=[0, 2, 1])
            return rations

        if exp_output == 'rnr':
            y_pred = convert_rnr_matrix_to_ration_mask(y_pred)
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    return precision


def recall_wrapper(exp_output):
    def recall(y_true, y_pred):
        def convert_intervals_to_ration_mask(interval):
            p = K.cumsum(interval, axis=1)
            rations = K.clip(p[:, :, 0] - p[:, :, 1] + interval[:, :, 1], 0, 1)
            return rations

        def convert_rnr_matrix_to_ration_mask(y):
            starts = y[:, :, :1]
            ends_given_starts = y[:, :, 1:]
            starts = tf.math.round(tf.clip_by_value(starts, 0., 1.))
            ends_given_starts = tf.linalg.band_part(ends_given_starts, 0, -1)
            max_ends_arg = tf.math.argmax(ends_given_starts, axis=2)
            max_ends_arg = tf.cast(max_ends_arg, dtype=tf.int32)
            ends_given_starts = tf.math.multiply(tf.cast(tf.squeeze(starts), dtype=tf.int32), max_ends_arg)
            ends = tf.cast(tf.one_hot(ends_given_starts, depth=MAX_SEQ_LENGTH), dtype=tf.float32)
            ends = tf.math.cumsum(ends, axis=2, reverse=True)
            rations = tf.linalg.band_part(ends, 0, -1)
            rations = tf.math.multiply(tf.tile(starts, (1, 1, MAX_SEQ_LENGTH)), rations)
            rations = tf.math.reduce_sum(rations, axis=1, keepdims=True)
            rations = tf.clip_by_value(rations, 0., 1.)
            rations = tf.transpose(rations, perm=[0, 2, 1])
            return rations

        if exp_output == 'rnr':
            y_pred = convert_rnr_matrix_to_ration_mask(y_pred)
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

        def convert_rnr_matrix_to_ration_mask(y):
            starts = y[:, :, :1]
            ends_given_starts = y[:, :, 1:]
            starts = tf.math.round(tf.clip_by_value(starts, 0., 1.))
            ends_given_starts = tf.linalg.band_part(ends_given_starts, 0, -1)
            max_ends_arg = tf.math.argmax(ends_given_starts, axis=2)
            max_ends_arg = tf.cast(max_ends_arg, dtype=tf.int32)
            ends_given_starts = tf.math.multiply(tf.cast(tf.squeeze(starts), dtype=tf.int32), max_ends_arg)
            ends = tf.cast(tf.one_hot(ends_given_starts, depth=MAX_SEQ_LENGTH), dtype=tf.float32)
            ends = tf.math.cumsum(ends, axis=2, reverse=True)
            rations = tf.linalg.band_part(ends, 0, -1)
            rations = tf.math.multiply(tf.tile(starts, (1, 1, MAX_SEQ_LENGTH)), rations)
            rations = tf.math.reduce_sum(rations, axis=1, keepdims=True)
            rations = tf.clip_by_value(rations, 0., 1.)
            rations = tf.transpose(rations, perm=[0, 2, 1])
            return rations

        if exp_output == 'rnr':
            y_pred = convert_rnr_matrix_to_ration_mask(y_pred)
        precision = precision(y_true, y_pred)
        recall = recall(y_true, y_pred)
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

    return f1


def sp_precision_wrapper(exp_output):
    def sp_precision(y_true, y_pred):
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

        if exp_output != 'rnr':
            return 0
        starts_pred = y_pred[:, :, :1]
        starts_true = y_true - tf.concat([tf.zeros_like(y_true[:, :1, :]), y_true[:, 1:, :]], axis=1)
        starts_true = tf.clip_by_value(starts_true, 0., 1.)
        precision = precision(y_true, y_pred)
        return precision

    return sp_precision


def sp_recall_wrapper(exp_output):
    def sp_recall(y_true, y_pred):
        def recall(y_true, y_pred):
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            recall = true_positives / (possible_positives + K.epsilon())
            return recall

        if exp_output != 'rnr':
            return 0
        starts_pred = y_pred[:, :, :1]
        starts_true = y_true - tf.concat([tf.zeros_like(y_true[:, :1, :]), y_true[:, 1:, :]], axis=1)
        starts_true = tf.clip_by_value(starts_true, 0., 1.)
        recall = recall(y_true, y_pred)
        return recall

    return sp_recall


def iou(y_true, y_pred):
    intersection = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    union = K.sum(K.round(K.clip(y_true + y_pred, 0, 1)))
    iou = intersection / (union + K.epsilon())
    return iou
