import glob
import os
import shutil
import numpy as np
from argparse import ArgumentParser

# util functions
def read_beam(bid, src, data_type, root):
    # data type = ans / ems
    path = root
    with open(path, 'r') as f:
        content = f.readlines()
        content = list(map(lambda s: s.rstrip(), content))
    return content

def read_lmprob(bid, reader, ans, world_size, root):
    stem = f'{root}/lm_prob/reader_{reader}_ans_{ans}/beam_{bid}/lm_prob_'
    integ_list = []
    lines_list = []
    num_lines = 0
    rank_list = list(range(world_size))
    
    for rank in rank_list:
        with open(stem+str(rank)+'.txt', 'r') as f:
            line_list = f.readlines()
            line_list = list(map(lambda s: s.rstrip(), line_list))
            lines_list.append(line_list)
            num_lines += len(line_list)

    for i in range(num_lines):
        src_idx = i % world_size
        idx = i // world_size
        integ_list.append(lines_list[src_idx][idx])
    return integ_list

def to_bool(str_input):
    if str_input == 'True':
        return True
    elif str_input == 'False':
        return False
    else:
        print('What is this?',str_input)

def merge_result(root_dir, output_dir, beam, n_gpu=8):

    score_types = ['j','s','js']
    head = ['selected_ems','selected_prob','text_prob','table_prob','k_prob','text_ems','table_ems','k_ems']

    for t in score_types:
        rows = []
        for i in range(n_gpu):
            with open(root_dir+f'/result_{t}_{i}.txt', 'r') as f:
                rows += f.readlines()
        with open(root_dir+f'/result_{t}_all.txt', 'w') as f:
            print(" || ".join(head), file=f)
            for r in rows:
                print(r.rstrip(), file=f)
        shutil.copy(root_dir+f'/result_{t}_all.txt', output_dir+f'/result_{t}_all_beam{beam}.txt')

def beam_allocation(output_dir, max_beams=5, n_gpu=8, dataset_size=3610):
    p_j_list = glob.glob(output_dir+f'/result_j_all_beam*')
    p_s_list = glob.glob(output_dir+f'/result_s_all_beam*')
    
    p_j_list.sort()
    p_s_list.sort()
    p_j_list.insert(0, 'empty')
    p_s_list.insert(0, 'empty')

    beam_lists =[[] for _ in range(max_beams+1)]
    for num_beams in range(1, max_beams+1):
        with open(p_j_list[num_beams], 'r') as f:
            rows_j = f.readlines()
            del rows_j[0]
        with open(p_s_list[num_beams], 'r') as f:
            rows_s = f.readlines()
            del rows_s[0]
        for bid in range(num_beams):
            text_p_self = [float(r.split('||')[bid+2]) for r in rows_s]
            table_p_self = [float(r.split('||')[num_beams+bid+2]) for r in rows_s]
            k_p_self = [float(r.split('||')[num_beams*2+bid+2]) for r in rows_s]
            text_p_tt = [float(r.split('||')[bid+2]) for r in rows_j]
            table_p_tt = [float(r.split('||')[num_beams+bid+2]) for r in rows_j]
            k_p_tt = [float(r.split('||')[num_beams*2+bid+2]) for r in rows_j]

            text_ems = [float(r.split('||')[num_beams*3+bid+2]) for r in rows_s]
            table_ems = [float(r.split('||')[num_beams*4+bid+2]) for r in rows_s]
            k_ems = [float(r.split('||')[num_beams*5+bid+2]) for r in rows_s]

            ## make beam
            beam = {'bid':bid}
            beam['text'] = {'ems':text_ems, 'p_self':text_p_self, 'p_tt':text_p_tt}
            beam['table'] = {'ems':table_ems, 'p_self':table_p_self, 'p_tt':table_p_tt}
            beam['k'] = {'ems':k_ems, 'p_self':k_p_self, 'p_tt':k_p_tt}
            beam_lists[num_beams].append(beam)

    beam_alloc_array = np.repeat(np.array([[5,4,3]]), dataset_size, 0)

    ems_list_tt = []
    for i in range(dataset_size):
        # i-th question
        # integrate ems, lmprob through beam results
        ems_integ = []
        prob_integ = []
        bpd = beam_alloc_array[i] # beams_per_data
        for k in range(bpd[0]):
            ems_integ.append(beam_lists[bpd[0]][k]['text']['ems'][i])
            prob_integ.append(beam_lists[bpd[0]][k]['text']['p_self'][i])
            
        for k in range(bpd[1]):
            ems_integ.append(beam_lists[bpd[1]][k]['table']['ems'][i])
            prob_integ.append(beam_lists[bpd[1]][k]['table']['p_self'][i])
            
        for k in range(bpd[2]):
            ems_integ.append(beam_lists[bpd[2]][k]['k']['ems'][i])
            prob_integ.append(beam_lists[bpd[2]][k]['k']['p_self'][i])

        # do selection (argmax)
        max_idx = max(range(len(prob_integ)), key=lambda i: prob_integ[i])
        ems_select = ems_integ[max_idx]
        if ems_select:
            ems_list_tt.append(1)
        else:
            ems_list_tt.append(0)
    em_tt = np.mean(ems_list_tt)
    print('P_self selection:',em_tt)

    ems_list_tt = []
    for i in range(dataset_size):
        # i-th question
        # integrate ems, lmprob through beam results
        ems_integ = []
        prob_integ = []
        bpd = beam_alloc_array[i] # beams_per_data
        for k in range(bpd[0]):
            ems_integ.append(beam_lists[bpd[0]][k]['text']['ems'][i])
            prob_integ.append(beam_lists[bpd[0]][k]['text']['p_tt'][i])
            
        for k in range(bpd[1]):
            ems_integ.append(beam_lists[bpd[1]][k]['table']['ems'][i])
            prob_integ.append(beam_lists[bpd[1]][k]['table']['p_tt'][i])
            
        for k in range(bpd[2]):
            ems_integ.append(beam_lists[bpd[2]][k]['k']['ems'][i])
            prob_integ.append(beam_lists[bpd[2]][k]['k']['p_tt'][i])

        # do selection (argmax)
        max_idx = max(range(len(prob_integ)), key=lambda i: prob_integ[i])
        ems_select = ems_integ[max_idx]
        if ems_select:
            ems_list_tt.append(1)
        else:
            ems_list_tt.append(0)
    em_tt = np.mean(ems_list_tt)
    print('P_tt selection:',em_tt)

    # P_ttk + P_self
    ems_list_tt_self = []
    for i in range(dataset_size):
        # i-th question
        # integrate ems, lmprob through beam results
        ems_integ = []
        prob_integ = []
        bpd = beam_alloc_array[i]
        for k in range(beam_alloc_array[i][0]):
            ems_integ.append(beam_lists[bpd[0]][k]['text']['ems'][i])
            prob_integ.append(np.mean([beam_lists[bpd[0]][k]['text']['p_tt'][i], beam_lists[bpd[0]][k]['text']['p_self'][i]]))
            
        for k in range(beam_alloc_array[i][1]):
            ems_integ.append(beam_lists[bpd[1]][k]['table']['ems'][i])
            prob_integ.append(np.mean([beam_lists[bpd[1]][k]['table']['p_tt'][i], beam_lists[bpd[1]][k]['table']['p_self'][i]]))
            
        for k in range(beam_alloc_array[i][2]):
            ems_integ.append(beam_lists[bpd[2]][k]['k']['ems'][i])
            prob_integ.append(np.mean([beam_lists[bpd[2]][k]['k']['p_tt'][i], beam_lists[bpd[2]][k]['k']['p_self'][i]]))

        # do selection (argmax)
        max_idx = max(range(len(prob_integ)), key=lambda i: prob_integ[i])
        ems_select = ems_integ[max_idx]
        if ems_select:
            ems_list_tt_self.append(1)
        else:
            ems_list_tt_self.append(0)
    em_tt_self = np.mean(ems_list_tt_self)
    print('P_tt_self selection:',em_tt_self)

def main(args):
    root_list = glob.glob(args.root_dir)
    root_list.sort()
    os.makedirs('./tmp_result/select/beam_allocation', exist_ok=True)
    output_dir = './tmp_result/select/beam_allocation'
    for beam, root_dir in enumerate(root_list):
        merge_result(root_dir, output_dir, beam, args.n_gpu)

    beam_allocation(output_dir)
   

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="./tmp_result/select/compare_nq_large_beam*")
    parser.add_argument("--n_gpu", type=int, default=8)
    
    args = parser.parse_args()
    main(args)
