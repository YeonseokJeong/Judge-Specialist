# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialize_parser()

    def add_optim_options(self):
        self.parser.add_argument('--warmup_steps', type=int, default=1000)
        self.parser.add_argument('--total_steps', type=int, default=1000)
        self.parser.add_argument('--total_epochs',type=int,default=20)
        self.parser.add_argument('--scheduler_steps', type=int, default=None, 
                        help='total number of step for the scheduler, if None then scheduler_total_step = total_epochs * step_per_batch')
        self.parser.add_argument('--accumulation_steps', type=int, default=1)
        self.parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
        self.parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
        self.parser.add_argument('--clip', type=float, default=1., help='gradient clipping')
        self.parser.add_argument('--optim', type=str, default='adam')
        self.parser.add_argument('--scheduler', type=str, default='fixed')
        self.parser.add_argument('--weight_decay', type=float, default=0.1)
        self.parser.add_argument('--fixed_lr', action='store_true')

    def add_eval_options(self):
        self.parser.add_argument('--write_results', action='store_true', help='save results')
        self.parser.add_argument('--write_crossattention_scores', action='store_true', 
                        help='save dataset with cross-attention scores')

    def add_reader_options(self):
        self.parser.add_argument('--train_data', type=str, default='none', help='path of train data1')
        self.parser.add_argument('--train_data2', type=str, default='none', help='path of train data2')
        self.parser.add_argument('--eval_data', type=str, default='none', help='path of eval data 1')
        self.parser.add_argument('--eval_data2', type=str, default='none', help='path of eval data 2')
        self.parser.add_argument('--eval_data3', type=str, default='none', help='path of eval data 3')
        self.parser.add_argument('--eval_data4', type=str, default='none', help='path of eval data 4')

        self.parser.add_argument('--model_size', type=str, default='base')
        self.parser.add_argument('--use_checkpoint', action='store_true', help='use checkpoint in the encoder')
        self.parser.add_argument('--text_maxlength', type=int, default=200, 
                        help='maximum number of tokens in text segments (question+passage)')
        self.parser.add_argument('--answer_maxlength', type=int, default=-1, 
                        help='maximum number of tokens used to train the model, no truncation if -1')
        self.parser.add_argument('--no_title', action='store_true', 
                        help='article titles not included in passages')
        self.parser.add_argument('--n_context', type=int, default=1)
        self.parser.add_argument('--num_beams', type=int, default=1) # beam search
        self.parser.add_argument('--resume_checkpoint',action='store_true',help='resume checkpoint from the checkpoint dir')
        
        # precomputed batches
        self.parser.add_argument('--train_batches', type=str, default='none', help='path of train data1')
        self.parser.add_argument('--train_batches2', type=str, default='none', help='path of train data2')
        self.parser.add_argument('--train_batches3', type=str, default='none', help='path of train data3')
        self.parser.add_argument('--train_batches_unified', type=str, default='none', help='path of train data unified')
        self.parser.add_argument('--eval_batches', type=str, default='none', help='path of eval data1')
        self.parser.add_argument('--eval_batches2', type=str, default='none', help='path of eval data2')
        self.parser.add_argument('--eval_batches3', type=str, default='none', help='path of eval data3')
        self.parser.add_argument('--eval_batches_unified', type=str, default='none', help='path of eval data unified')

    def add_retriever_options(self):
        self.parser.add_argument('--train_data', type=str, default='none', help='path of train data')
        self.parser.add_argument('--eval_data', type=str, default='none', help='path of eval data')
        self.parser.add_argument('--indexing_dimension', type=int, default=768)
        self.parser.add_argument('--no_projection', action='store_true', 
                        help='No addition Linear layer and layernorm, only works if indexing size equals 768')
        self.parser.add_argument('--question_maxlength', type=int, default=40, 
                        help='maximum number of tokens in questions')
        self.parser.add_argument('--passage_maxlength', type=int, default=200, 
                        help='maximum number of tokens in passages')
        self.parser.add_argument('--no_question_mask', action='store_true')
        self.parser.add_argument('--no_passage_mask', action='store_true')
        self.parser.add_argument('--extract_cls', action='store_true')
        self.parser.add_argument('--no_title', action='store_true', 
                        help='article titles not included in passages')
        self.parser.add_argument('--n_context', type=int, default=1)


    def initialize_parser(self):
        # basic parameters
        self.parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment')
        self.parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint/', help='models are saved here')
        self.parser.add_argument('--model_path', type=str, default='none', help='path for retraining')
        self.parser.add_argument('--model_name', type=str, default='none', help='model card to import')
        self.parser.add_argument('--tmp_dir', type=str, default='none', help='tmp directory for analysis')

        # dataset parameters
        self.parser.add_argument("--per_gpu_batch_size", default=1, type=int, 
                        help="Batch size per GPU/CPU for training.")
        self.parser.add_argument('--maxload', type=int, default=-1)

        self.parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
        self.parser.add_argument("--main_port", type=int, default=-1,
                        help="Main port (for multi-node SLURM jobs)")
        self.parser.add_argument('--seed', type=int, default=0, help="random seed for initialization")
        # training parameters
        self.parser.add_argument('--eval_freq', type=float, default=1,
                        help='evaluate model every <eval_freq> epochs during training')
        self.parser.add_argument('--save_freq', type=float, default=5,
                        help='save model every <save_freq> epochs during training')
        self.parser.add_argument('--eval_print_freq', type=int, default=100,
                        help='print intermdiate results of evaluation every <eval_print_freq> steps in eval_print_freq')
        self.parser.add_argument('--print_dir',type=str, default='./', help='path to save intermediate (calibration) results')


    def print_options(self, opt):
        message = '\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default_value = self.parser.get_default(k)
            if v != default_value:
                comment = f'\t(default: {default_value})'
            message += f'{str(k):>30}: {str(v):<40}{comment}\n'

        expr_dir = Path(opt.checkpoint_dir)/ opt.name
        model_dir = expr_dir / 'models'
        model_dir.mkdir(parents=True, exist_ok=True)
        with open(expr_dir/'opt.log', 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

        logger.info(message)

    def parse(self):
        opt = self.parser.parse_args()
        return opt


def get_options(use_reader=False,
                use_retriever=False,
                use_optim=False,
                use_eval=False):
    options = Options()
    if use_reader:
        options.add_reader_options()
    if use_retriever:
        options.add_retriever_options()
    if use_optim:
        options.add_optim_options()
    if use_eval:
        options.add_eval_options()
    return options.parse()

