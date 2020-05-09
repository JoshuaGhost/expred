import os

import config
from metrices import *
from losses import imbalanced_bce_bayesian, imbalanced_bce_resampling
from losses import rnr_matrix_loss


class Config():
    def __init__(self, args):
        self.BATCH_SIZE = args.batch_size
        self.par_lambda = args.par_lambda
        self.NUM_EPOCHS = args.num_epochs
        self.gpu_id = args.gpu_id
        self.exp_structure = args.exp_structure
        self.dataset_name = args.dataset
        self.DO_DELETE = args.delete_checkpoints
        self.do_train = args.do_train
        self.load_best = not self.do_train
        self.evaluate = args.evaluate
        self.exp_visualize = args.exp_visualize
        self.exp_benchmark = args.exp_benchmark
        self.merge_evidences = args.merge_evidences
        self.BENCHMARK_SPLIT_NAME = args.benchmark_split
        self.train_on_portion = args.train_on_portion
        self.freeze_cls = args.freeze_cls
        self.freeze_exp = args.freeze_exp
        self.train_cls_first = args.train_cls_first
        self.train_exp_first = args.train_exp_first
        self.start_from_phase1 = args.start_from_phase1
        self.load_phase1 = args.load_phase1
        self.pooling = args.pooling
        self.EXP_OUTPUT = self.exp_structure
        self.bert_size = args.bert_size
        self.rebalance_approach = args.rebalance_approach
        self.USE_BUCKET = args.use_bucket
        self.data_dir = args.data_dir
        self.len_head = args.len_viz_head

        self.loss_function = self.decide_loss_function()
        self.suffix = self.decide_suffix()
        self.montage_output_dir()

        self.data_dir = os.path.join(self.data_dir, self.dataset_name)
        if self.par_lambda is None:
            self.loss_weights = None
        else:
            self.loss_weights = {'cls_output': 1,
                            'exp_output': self.par_lambda}
        self.metrics = {'cls_output': 'accuracy',
                   'exp_output': [f1_wrapper(self.EXP_OUTPUT),
                                  sp_precision_wrapper(self.EXP_OUTPUT),
                                  sp_recall_wrapper(self.EXP_OUTPUT),
                                  precision_wrapper(self.EXP_OUTPUT),
                                  recall_wrapper(self.EXP_OUTPUT)]}
        self.loss = {'cls_output': 'binary_crossentropy',
                     'exp_output': self.loss_function()}
        self.MAX_SEQ_LENGTH = MAX_SEQ_LENGTH

    def decide_loss_function(self):
        if self.EXP_OUTPUT == 'gru':
            if self.rebalance_approach == 'resampling':
                return imbalanced_bce_resampling
            else:
                return imbalanced_bce_bayesian
        elif self.EXP_OUTPUT == 'rnr':
            return rnr_matrix_loss

    def decide_suffix(self):
        if self.freeze_cls:
            return 'freeze_cls'
        elif self.freeze_exp:
            return 'freeze_exp'
        elif self.train_cls_first:
            return 'train_cls_first'
        elif self.train_exp_first:
            return 'train_exp_first'
        else:
            return ''

    def montage_output_dir(self):
        OUTPUT_DIR = ['bert_{}_seqlen_{}_{}_exp_output_{}'.format(self.bert_size, MAX_SEQ_LENGTH, self.dataset_name, self.EXP_OUTPUT)]
        OUTPUT_DIR.append('merged_evidences' if self.merge_evidences else 'separated_evidences')
        if self.train_on_portion != 0:
            OUTPUT_DIR += ['train_on_portion', str(self.train_on_portion)]
        self.DATASET_CACHE_NAME = '_'.join(OUTPUT_DIR) + '_inputdata_cache'
        if self.par_lambda is None:
            OUTPUT_DIR.append('no_weight')
        else:
            OUTPUT_DIR.append('par_lambda_{}'.format(self.par_lambda))
        OUTPUT_DIR.append(
            'no_padding_imbalanced_bce_{}_pooling_{}_learning_rate_{}'.format(self.rebalance_approach, self.pooling, config.LEARNING_RATE))
        OUTPUT_DIR.append(self.suffix)
        OUTPUT_DIR = '_'.join(OUTPUT_DIR)
        self.MODEL_NAME = OUTPUT_DIR
        self.OUTPUT_DIR = os.path.join('model_checkpoints', self.MODEL_NAME)