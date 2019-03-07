#!/bin/bash

sdir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

#export GCEC=1
echo gce chosen: $GCEC, instance-$GCEC

if [ -z $GCEC ]; then
  gce=instance-2
else
  gce=instance-$GCEC
fi

echo using $gce

dir_project="~/project_walk/"
dir_local_code="$sdir/../code/"
dir_rmt_code="$dir_project/"
