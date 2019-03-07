#!/bin/bash

sdir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source $sdir/config.sh

./start_gce.sh
./upload_code.sh
echo running tmux on $gce

gcloud compute ssh $gce -- "cd jing_walk &&  tmux new-session -d ./run.sh"
date
