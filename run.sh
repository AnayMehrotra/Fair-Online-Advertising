
PYTHON="/usr/local/Cellar/python/3.6.5/bin/python3.6"
num_cores=13 #Number of cores-1

is_bgs_running () {
  pids=$1
  for i in `seq 0 $2`
  do
    while [[ $(ps aux | grep ${pids[$i]} | wc -l) -gt 1 ]]
    do
      echo "waiting for $i-th background process to finish..."
      echo "sleeping for 30mins"
      sleep 30m
      echo "woke!!"
    done
  done
}

is_bg_running () {
  while [[ $(ps aux | grep $1 | wc -l) -gt 1 ]]
  do
    echo "waiting for the background process to finish..."
    echo "sleeping for 30mins"
    sleep 30m
    echo "woke!!"
  done
}

echo "cleaning data..."
$PYTHON cleanData.py > output_cd 2>&1
echo "fitting distributions..."
$PYTHON distributionFitting.py > output_df 2>&1

echo "setting up logistic things for gradient framework..."
pids=()
for i in `seq 0 $num_cores`
do
  $PYTHON logistic_gradient_framework.py $i > output_lgf_$i 2>&1 &
  pids+=($!)
done

is_bgs_running $pids $num_cores

echo "getting implicit unbalance of keyword pairs..."
$PYTHON -c 'import run_gradient_framework as rgf; rgf.get_unbalanced_keys();' > output_3 2>&1 &

is_bg_running $!

echo "running experiment..."
pids=()
for i in `seq 0 $num_cores`
do
  $PYTHON run_gradient_framework.py $i > output_rgf_$i 2>&1 &
  pids+=($!)
done
is_bgs_running $pids $num_cores

echo "generating plots..."
$PYTHON plot_gradient_framework.py > output_plot 2>&1

echo "plots saved. done."
