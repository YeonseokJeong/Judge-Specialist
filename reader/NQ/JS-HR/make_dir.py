import os, sys

if len(sys.argv) < 2:
    raise Exception("not enough arguments")

PROC_NAME = sys.argv[0]
dirname = sys.argv[1]

root = 'tmp_result'
subdir1 = ['test','dev','train']
subdir2 = ['Text','Table','K','Unified']
dirname = sys.argv[1]

for s1 in subdir1:
    for s2 in subdir2:
        os.makedirs(f'{root}/gen/{dirname}/{s1}/{s2}',exist_ok=True)
os.makedirs(f'{root}/select/{dirname}', exist_ok=True)
