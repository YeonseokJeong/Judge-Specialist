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


def infer(model, optimizer, scheduler, step, eval_dataset_list, \
          opt, collator_list, best_dev_em, checkpoint_path, world_size):
    if opt.is_main:
        try:
            tb_logger = torch.utils.tensorboard.SummaryWriter(Path(opt.checkpoint_dir) / opt.name)
        except:
            tb_logger = None
            logger.warning('Tensorboard is not available.')

    torch.manual_seed(opt.seed)  # different seed for different sampling depending on global_rank


    test_em, results = evaluate(model, eval_dataset_list[0], tokenizer, collator_list[0], opt)

    ## save
    torch.save(results, f'{opt.tmp_dir}/batches_bs{opt.per_gpu_batch_size}_{opt.local_rank}.pt')

    if opt.is_main:
        curr_loss = 0
        if tb_logger is not None:
            tb_logger.add_scalar("Evaluation", test_em, step)
        f_w = open(opt.tmp_dir + '/eval_log.txt', 'a', encoding='utf-8')
        print(str(100 * test_em), file=f_w)
        f_w.close()
        print(f'done:{100*test_em}')

def evaluate(model, dataset, tokenizer, collator, opt, step=None, num_batch=None):
    print('evaluation start: gpu', opt.local_rank)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
                            sampler=sampler,
                            batch_size=opt.per_gpu_batch_size,
                            drop_last=False,
                            num_workers=10,
                            collate_fn=collator
                            )
    model.eval()
    total = 0
    exactmatch = []
    extended_batch_list = []
    model = model.module

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            (idx, _, _, context_ids, context_mask) = batch
            num_beams = opt.num_beams

            # STEP 1: extract (seq, token logprobs, seq lens)
            outputs, token_scores, seq_lens = model.cand_generate(
                    input_ids=context_ids.cuda(),
                    attention_mask=context_mask.cuda(),
                    max_length=opt.answer_maxlength,
                    num_beams=num_beams,
            )
            # TODO -- take care of beam search
            #print('outputs shape:',outputs.shape)
            #print('token_scores shape:',token_scores.shape)
            #print('seq_lens:',seq_lens)
            
            # STEP 2: get ems tensor
            ems = []
            for k, o in enumerate(outputs):
                # ans = tokenizer.decode(o, skip_special_tokens=True)
                # ans = tokenizer.decode(o, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                ans = tokenizer.decode(o, clean_up_tokenization_spaces=False)

                gold = dataset.get_example(idx[k//num_beams])['answers']
                score = src.evaluation.ems(ans, gold)
                total += 1
                exactmatch.append(score)
                if score:
                    ems.append(1)
                else:
                    ems.append(0)
            ems = torch.tensor(ems)

            # STEP 3: make augmented batch, export
            # (idx, context_ids, context_mask, seq, token_scores, seq_lens, ems)
            
            # preprocess / postprocess outputs (remove <pad> heading & max padding)
            outputs = outputs[:,1:]
            bsz, outputs_len = outputs.shape
            pad_frame = torch.zeros(bsz,opt.answer_maxlength, dtype=torch.long)
            pad_frame[:,:outputs_len] = outputs
            outputs = pad_frame
            
            # preprocess / preprocess token_scores
            token_scores = token_scores[:,1:]
            bsz, token_scores_len = token_scores.shape
            pad_frame = torch.zeros(bsz,opt.answer_maxlength, dtype=torch.float)
            pad_frame[:,:token_scores_len] = token_scores
            token_scores = pad_frame
            
            extended_batch = (idx, context_ids, context_mask, outputs, token_scores, seq_lens, ems)
            extended_batch_list.append(extended_batch)
            
            if (i+1)%10 == 0 and opt.is_main:
                print(i+1,'/',len(dataloader))
                    
    avg_exactmatch, total = src.util.weighted_average(np.mean(exactmatch), total, opt)
    result = extended_batch_list
    return avg_exactmatch, result


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
    # if not checkpoint_exists and opt.is_main:
    #    options.print_options(opt)
    # checkpoint_path, checkpoint_exists = util.get_checkpoint_path(opt)

    logger = src.util.init_logger(
        opt.is_main,
        None,
        checkpoint_path / 'run.log'
    )

    # model_name = 't5-' + opt.model_size
    model_class = src.model.FiDT5

    # load data
    tokenizer = transformers.T5Tokenizer.from_pretrained(opt.model_name)
    collator1 = src.data.Collator(opt.text_maxlength, tokenizer, answer_maxlength=opt.answer_maxlength)
    collator_list = []
    collator_list.append(collator1)

    # use golbal rank and world size to split the eval set on multiple gpus
    eval_examples1 = src.data.load_data(
        opt.eval_data,
        global_rank=opt.local_rank,
        world_size=len(devices),
    )
    eval_dataset1 = src.data.Dataset(eval_examples1, opt.n_context)

    eval_dataset_list = []
    eval_dataset_list.append(eval_dataset1)

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
    else:  # loading from checkpoint
        model, optimizer, scheduler, opt_checkpoint, step, best_dev_em = \
            src.util.load(model_class, opt.model_path, opt, reset_params=False)
        step = 0  # step is set to zero, for the sake of step display
        logger.info(f"Model loaded from {opt.model_path}")

    model.set_checkpoint(opt.use_checkpoint)

    if opt.is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            find_unused_parameters=False,
        )

    logger.info("Start training")
    infer(
        model,
        optimizer,
        scheduler,
        step,
        eval_dataset_list,
        opt,
        collator_list,
        best_dev_em,
        checkpoint_path,
        len(devices)
    )
