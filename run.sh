
PYTHON="/opt/rh/rh-python36/root/bin/python3"
num_cores=13
num_cores_minus=$((num_cores--))
echo $num_cores_minus

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
for i in `seq 0 $num_cores_minus`
do
  $PYTHON logistic_gradient_framework.py $i $num_cores_minus > output_lgf_$i 2>&1 &
  pids+=($!)
done

is_bgs_running $pids $num_cores_minus

echo "getting implicit unbalance of keyword pairs..."
$PYTHON -c 'import run_gradient_framework as rgf; rgf.get_unbalanced_keys();' > output_rgf_ini 2>&1 &

is_bg_running $!

echo "running experiment..."
pids=()
for i in `seq 0 $num_cores_minus`
do
  $PYTHON run_gradient_framework.py $i $num_cores_minus > output_rgf_$i 2>&1 &
  pids+=($!)
done
is_bgs_running $pids $num_cores_minus

echo "generating plots..."
$PYTHON plot_gradient_framework.py > output_plot 2>&1

echo "plots saved. done."
