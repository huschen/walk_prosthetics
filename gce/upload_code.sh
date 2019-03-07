#!/bin/bash

sdir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source $sdir/config.sh

echo updating $gce

cd $dir_local_code
pwd

#find . -type d -name "__pycache__" | xargs rm -rf

gcloud compute scp run.sh $gce:$dir_rmt_code/.
gcloud compute scp source_code.py $gce:$dir_rmt_code/. # an example
