# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from cgi import print_arguments
import time
import sys
import torch
import transformers
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from src.options import Options

import src.slurm
import src.util
import src.evaluation
import src.data
import src.model
import os
from src.util_custom import *
import tqdm
import re
import torch.distributed as dist



def eval_wrapper(model, optimizer, scheduler, step, eval_batches_list, \
          opt, collator_list, best_dev_em, checkpoint_path, world_size):
    if opt.is_main:
        try:
            tb_logger = torch.utils.tensorboard.SummaryWriter(Path(opt.checkpoint_dir) / opt.name)
        except:
            tb_logger = None
            logger.warning('Tensorboard is not available.')

    torch.manual_seed(opt.local_rank + opt.seed)  # different seed for different sampling depending on global_rank

    # evaluate
    results = evaluate(model, eval_batches_list, tokenizer, collator_list[0], opt, step)

    ## save
    f_score = open(opt.tmp_dir + '/result_j_' + str(opt.local_rank) + '.txt','w')
    for row in results['row_j']:
        row_float = list(map(lambda x:float(x), row))
        row_str = list(map(lambda x:str(x), row_float))
        print(" || ".join(row_str), file=f_score)
    f_score.close()
    
    f_score = open(opt.tmp_dir + '/result_s_' + str(opt.local_rank) + '.txt','w')
    for row in results['row_s']:
        row_float = list(map(lambda x:float(x), row))
        row_str = list(map(lambda x:str(x), row_float))
        print(" || ".join(row_str), file=f_score)
    f_score.close()
    
    f_score = open(opt.tmp_dir + '/result_js_' + str(opt.local_rank) + '.txt','w')
    for row in results['row_js']:
        row_float = list(map(lambda x:float(x), row))
        row_str = list(map(lambda x:str(x), row_float))
        print(" || ".join(row_str), file=f_score)
    f_score.close()

    print(f'done:{opt.local_rank}')
    '''
    if opt.is_main:
        log = f"result |"
        log += f"eval_j : {100 * test_em_j:.2f}EM |"
        log += f"eval_s : {100 * test_em_s:.2f}EM |"
        log += f"eval_js : {100 * test_em_js:.2f}EM |"
        logger.info(log)

        f_w = open(opt.tmp_dir + '/eval_log.txt', 'a', encoding='utf-8')
        print(f'{100 * test_em_j} || {100 * test_em_s} || {100 * test_em_js}', file=f_w)
        f_w.close()
    '''



def evaluate(model, eval_batches_list, tokenizer, collator, opt, step):
    print('evaluation start: gpu', opt.local_rank)
    sampler = SequentialSampler(eval_batches_list[0])
    eval_batches1, eval_batches2, eval_batches3, eval_batches_unified = eval_batches_list
    all_batches = list(zip(eval_batches1, eval_batches2, eval_batches3, eval_batches_unified))
    

    model.eval()
    total = 0
    exactmatch_j = []
    exactmatch_s = []
    exactmatch_js = []
    selected_j = []
    selected_s = []
    selected_js = []
    row_j_list = []
    row_s_list = []
    row_js_list = []
    model = model.module

    with torch.no_grad():
        for b_idx in sampler:
            all_batch = all_batches[b_idx]
            single_batches = all_batch[:-1]
            unified_batch = all_batch[-1]

            # unpack unified batch
            _, context_ids, context_mask, _, _, _, _ = unified_batch;
            bsz, cand_num = context_ids.shape[0], len(single_batches)
            
            # get prob for each single batch
            cand_selfprobs = torch.zeros(bsz, cand_num, dtype=float)
            cand_probs = torch.zeros(bsz, cand_num, dtype=float)
            cand_ems = torch.zeros(bsz, cand_num, dtype=float)
            for i, single_batch in enumerate(single_batches):
                # unpack single batch
                _, _, _, cand_seq, token_scores, seq_lens, ems = single_batch
                seq_lens = seq_lens.long()
                cand_probs[:,i] = model.get_prob(input_ids=context_ids.cuda(),
                                               attention_mask=context_mask.cuda(),
                                               labels=cand_seq.cuda(),
                                               label_lens=seq_lens.cuda())

                seq_logprobs = torch.sum(token_scores, dim=-1).cuda()
                seq_logprobs = seq_logprobs / seq_lens # length normalization
                cand_selfprobs[:,i] = torch.exp(seq_logprobs)
                cand_ems[:,i] = ems
            
            # select with argmax
            selected_idx_j = torch.argmax(cand_probs, dim=1)
            selected_idx_s = torch.argmax(cand_selfprobs, dim=1)
            selected_idx_js = torch.argmax(cand_probs + cand_selfprobs, dim=1)
            
            # get ems
            for i in range(bsz):
                # idx
                idx_j = selected_idx_j[i].item()
                idx_s = selected_idx_s[i].item()
                idx_js = selected_idx_js[i].item()
                # prob
                prob_j = cand_probs[i,idx_j].item()
                prob_s = cand_selfprobs[i,idx_s].item()
                prob_js = (cand_probs[i,idx_js].item() + cand_selfprobs[i,idx_js].item()) / 2
                # score
                score_j = cand_ems[i,idx_j].item()
                score_s = cand_ems[i,idx_s].item()
                score_js = cand_ems[i,idx_js].item()
                # selected
                selected_j.append(idx_j)
                selected_s.append(idx_s)
                selected_js.append(idx_js)
                # exactmatch
                exactmatch_j.append(score_j)
                exactmatch_s.append(score_s)
                exactmatch_js.append(score_js)
                total += 1
                # data for analysis
                row_j = [score_j, prob_j] + list(cand_probs[i,:]) + list(cand_ems[i,:])
                row_s = [score_s, prob_s] + list(cand_selfprobs[i,:]) + list(cand_ems[i,:])
                row_js = [score_js, prob_js] + list((cand_probs[i,:] + cand_selfprobs[i,:])/2) + list(cand_ems[i,:])
                row_j_list.append(row_j)
                row_s_list.append(row_s)
                row_js_list.append(row_js)



    #avg_exactmatch_j, total = src.util.weighted_average(np.mean(exactmatch_j), total, opt)
    #avg_exactmatch_s, total = src.util.weighted_average(np.mean(exactmatch_s), total, opt)
    #avg_exactmatch_js, total = src.util.weighted_average(np.mean(exactmatch_js), total, opt)
    #result = {'em_j': exactmatch_j, 'em_s': exactmatch_s, 'em_js': exactmatch_js,
    #          'selected_j': selected_j, 'selected_s': selected_s,'selected_js': selected_js}
    result = {'row_j': row_j_list, 'row_s': row_s_list, 'row_js': row_js_list}
    
    return result


if __name__ == "__main__":
    options = Options()
    options.add_reader_options()
    options.add_optim_options()
    opt = options.parse()
    # opt = options.get_options(use_reader=True, use_optim=True)
    devices = [d for d in range(torch.cuda.device_count())]
    print(devices)

    torch.manual_seed(opt.seed)
    src.slurm.init_distributed_mode(opt)
    src.slurm.init_signal_handler()

    checkpoint_path = Path(opt.checkpoint_dir) / opt.name
    checkpoint_exists = checkpoint_path.exists()
    if opt.is_distributed:
        torch.distributed.barrier()
    torch.distributed.barrier()
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    #if not checkpoint_exists and opt.is_main:
    #    options.print_options(opt)
    # checkpoint_path, checkpoint_exists = util.get_checkpoint_path(opt)

    logger = src.util.init_logger(
        opt.is_main,
        None,
        checkpoint_path / 'run.log'
    )

    #model_name = 't5-' + opt.model_size
    model_class = src.model.FiDT5

    # load data
    tokenizer = transformers.T5Tokenizer.from_pretrained(opt.model_name)
    collator1 = src.data.Collator(opt.text_maxlength, tokenizer, answer_maxlength=opt.answer_maxlength)
    collator_list = []
    collator_list.append(collator1)


    # use golbal rank and world size to split the eval set on multiple gpus
    eval_batches1 = torch.load(opt.eval_batches + f'/batches_bs{opt.per_gpu_batch_size}_{opt.local_rank}.pt') # precomputed text batch
    eval_batches2 = torch.load(opt.eval_batches2 + f'/batches_bs{opt.per_gpu_batch_size}_{opt.local_rank}.pt') # precomputed table batch
    eval_batches3 = torch.load(opt.eval_batches3 + f'/batches_bs{opt.per_gpu_batch_size}_{opt.local_rank}.pt') # precomputed k batch
    eval_batches_unified = torch.load(opt.eval_batches_unified + f'/batches_bs{opt.per_gpu_batch_size}_{opt.local_rank}.pt') # precomputed unified batch

    eval_batches_list = []
    eval_batches_list.append(eval_batches1)
    eval_batches_list.append(eval_batches2)
    eval_batches_list.append(eval_batches3)
    eval_batches_list.append(eval_batches_unified)

    # initialize
    if opt.model_path == "none":
        t5 = transformers.T5ForConditionalGeneration.from_pretrained(opt.model_name)
        model = src.model.FiDT5(t5.config)
        model.load_t5(t5.state_dict())
        model = model.to(opt.local_rank)
        # scheduler steps setup
        if opt.scheduler_steps is None:
            scheduler_steps = opt.total_epochs * len(train_examples1) / opt.per_gpu_batch_size
        else:
            scheduler_steps = opt.scheduler_steps
        if opt.warmup_steps is None:
            warmup_steps = scheduler_steps // 15
        else:
            warmup_steps = opt.warmup_steps
        optimizer, scheduler = src.util.set_optim(opt, model, warmup_steps=warmup_steps,
                                                  scheduler_steps=scheduler_steps)
        step, best_dev_em = 0, 0.0
    else: # loading from checkpoint
        model, optimizer, scheduler, opt_checkpoint, step, best_dev_em = \
            src.util.load(model_class, opt.model_path, opt, reset_params=False)
        step = 0 # step is set to zero, for the sake of step display
        logger.info(f"Model loaded from {opt.model_path}")

    model.set_checkpoint(opt.use_checkpoint)
    
    if opt.is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            find_unused_parameters=False,
        )
        #model._set_static_graph()


    logger.info("Start training")
    eval_wrapper(
        model,
        optimizer,
        scheduler,
        step,
        eval_batches_list,
        opt,
        collator_list,
        best_dev_em,
        checkpoint_path,
        len(devices)
    )






