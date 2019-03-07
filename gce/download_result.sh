#!/bin/bash

sdir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source $sdir/config.sh

cd $dir_local_code/result
pwd

tmp=`gcloud compute ssh $gce -- "cd $dir_rmt_code && ls -td train_* | head -n1"`
train_dir=`echo $tmp |sed "s/$(printf '\r')\$//"`

tmp=`gcloud compute ssh $gce -- "cd $dir_rmt_code/$train_dir && ls -t ddpg.ckpt* | head -n1"`
ckpt=`echo $tmp | grep -o '.*\-[0-9]\+'`

echo --------------------------------------------------------------------------
echo $train_dir $ckpt

mkdir -p $train_dir/
mkdir -p $train_dir/tb

if [ ! -d $train_dir/code ]; then
  mkdir -p $train_dir/code
  cd $dir_local_code/result/$train_dir/code

  gcloud compute scp $gce:$dir_rmt_code/run.sh  .
  gcloud compute scp  $gce:$dir_rmt_code/source_code.py . # an example
fi


gcloud compute scp $gce:$dir_rmt_code/$train_dir/{*[!0-9]0.monitor*.csv,0.monitor*.csv} $train_dir/.

cd $dir_local_code/result/$train_dir
python $dir_local_code/tools/results_plotter.py --dirs . --output ./plot.png


cd $dir_local_code/result/$train_dir/code
if [ ! -z "$ckpt" ]; then
  gcloud compute scp $gce:$dir_rmt_code/$train_dir/$ckpt* $dir_local_code/result/$train_dir/.
  echo python main.py --mode 2 --render_eval --start-ckpt ../$ckpt --verbose_eval
  pwd
  python main.py --mode 2 --render_eval --start_ckpt ../$ckpt --verbose_eval --nb_epochs 2 --demo_files '../../../demo_train.csv,../../../demo_test0.csv' &

fi


gcloud compute scp $gce:$dir_rmt_code/$train_dir/log.txt $dir_local_code/result/$train_dir/.
gcloud compute scp $gce:$dir_rmt_code/$train_dir/tb/* $dir_local_code/result/$train_dir/tb/. 

cd $dir_local_code/result/$train_dir/tb
tensorboard --logdir=. --port 6007
