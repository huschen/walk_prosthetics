#!/bin/bash

set -e

sdir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source $sdir/config.sh

echo starting $gce

gcloud compute instances list
gcloud compute instances start $gce
gcloud compute instances list

echo waiting for gce to be ready
gce_ready=0
for i in {1..6}; do
  if [ $gce_ready -eq 0 ]; then
    echo ...
    if ! gcloud compute ssh $gce -- "pwd" > /dev/null 2>&1; then
      echo sleep...; sleep 3
    else
      gce_ready=1
    fi
  fi
done

if [ $gce_ready -eq 0 ]; then
  echo gce is not ready, exiting...
  exit 1
fi

echo $gce is ready
