import matplotlib.pyplot as plt
import re, string
import numpy as np
import json


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def calc_em_norm(li):
  denom = len(li)
  if denom == 0:
    return 0
  cor = 0
  for l in li:
    norm_aliases = list(map(lambda x:normalize_answer(x),l[2]))
    if normalize_answer(l[1]) in norm_aliases:
      cor +=1
  return cor/denom

# point analysis
def pt_anal_norm(all_list, n):
  stride = int(len(all_list)/n)
  idx_list = list(range(n))
  idx_list = list(map(lambda x:(x+1)*stride, idx_list))
  idx_list[-1] = len(all_list)
  em_list = []
  for i in range(n):
    t = idx_list[i]
    sample = all_list[:t]
    em_list.append(calc_em_norm(sample))
  return em_list

# point analysis for (score, ems) tuples
def pt_anal_norm_tuple(all_list, n):
  stride = int(len(all_list)/n)
  idx_list = list(range(n))
  idx_list = list(map(lambda x:(x+1)*stride, idx_list))
  idx_list[-1] = len(all_list)
  em_list = []
  for i in range(n):
    t = idx_list[i]
    sample = all_list[:t]
    em_list.append(np.mean(list(map(lambda x:x[1],sample))))
  return em_list

# point analysis 2 (non-cumulative)
def pt_anal2_norm(all_list, n):
  stride = int(len(all_list)/n)
  idx_list = list(range(n+1))
  idx_list = list(map(lambda x:x*stride, idx_list))
  idx_list[-1] = len(all_list)
  em_list = []
  for i in range(n):
    s = idx_list[i]
    t = idx_list[i+1]
    sample = all_list[s:t]
    em_list.append(calc_em_norm(sample))
  return em_list

# point analysis 2 (non-cumulative) for tuples
def pt_anal2_norm_tuple(all_list, n):
  stride = int(len(all_list)/n)
  idx_list = list(range(n+1))
  idx_list = list(map(lambda x:x*stride, idx_list))
  idx_list[-1] = len(all_list)
  em_list = []
  for i in range(n):
    s = idx_list[i]
    t = idx_list[i+1]
    sample = all_list[s:t]
    em_list.append(np.mean(list(map(lambda x:x[1],sample))))
  return em_list


# point analysis with (score, pred, truth) triples
def analyze_triples(triple_list, n, epoch, result_dir):
    print('Start analysis & plotting...')
    sorted_triples = sorted(triple_list, key=lambda x:x[0], reverse=True) # descending order
    analysis_result = pt_anal_norm(sorted_triples, n)

    # figure
    base = list(range(n))
    base = list(map(lambda x:(x+1)*int(100/n), base))
    plt.figure(figsize=(12,8))
    plt.plot(base, analysis_result, 'b')
    plt.plot(base, analysis_result, 'bo')
    plt.title('Epoch '+str(epoch))
    plt.savefig(result_dir+'/plot_epoch'+str(epoch)+'.png')

    # list
    with open(result_dir+'/em_epoch'+str(epoch)+'.json', 'w') as f:
        json.dump(analysis_result,f)
    print('Analysis result saved at',result_dir)

# point analysis with (score, ems) tuples
def analyze_tuples(tuple_list, n, epoch, result_dir):
    print('Start analysis & plotting...')
    sorted_tuples = sorted(tuple_list, key=lambda x:x[0], reverse=True) # descending order
    analysis_result = pt_anal_norm_tuple(sorted_tuples, n)

    # figure
    base = list(range(n))
    base = list(map(lambda x:(x+1)*int(100/n), base))
    plt.figure(figsize=(12,8))
    plt.plot(base, analysis_result, 'b')
    plt.plot(base, analysis_result, 'bo')
    plt.title('Epoch '+str(epoch))
    plt.savefig(result_dir+'/plot_epoch'+str(epoch)+'.png')

    # list
    with open(result_dir+'/em_epoch'+str(epoch)+'.json', 'w') as f:
        json.dump(analysis_result,f)
    print('Analysis result saved at',result_dir)


# point analysis with (score, ems) tuples
def analyze_tuples_nc(tuple_list, n, epoch, result_dir):
    print('Start analysis & plotting...')
    sorted_tuples = sorted(tuple_list, key=lambda x:x[0], reverse=True) # descending order
    analysis_result = pt_anal2_norm_tuple(sorted_tuples, n)

    # figure
    base = list(range(n))
    base = list(map(lambda x:(x+1)*int(100/n), base))
    plt.figure(figsize=(12,8))
    plt.plot(base, analysis_result, 'b')
    plt.plot(base, analysis_result, 'bo')
    plt.title('Epoch '+str(epoch))
    plt.savefig(result_dir+'/plot_epoch'+str(epoch)+'.png')

    # list
    with open(result_dir+'/em_epoch'+str(epoch)+'.json', 'w') as f:
        json.dump(analysis_result,f)
    print('Analysis result saved at',result_dir)

