num_beams=$1
dir_name="compare_nq_large_beam${num_beams}"

# STEP 1: make directory structure for experiment
python make_dir.py $dir_name

# STEP 2: beam search inference
if [ $num_beams -gt 1 ]
then
  sh cand_beam.sh $dir_name $num_beams
else
  sh cand_greedy.sh $dir_name
fi
#sh cand_beam.sh $dir_name $num_beams
#sh cand_greedy.sh $dir_name

# STEP 3: selection
if [ $num_beams -gt 1 ]
then
  sh test_select_beam.sh $dir_name $num_beams
else
  sh test_select.sh $dir_name
fi

# STEP 4: beam allocation
python beam_allocation.py