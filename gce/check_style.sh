#!/bin/bash

sdir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $sdir/..


if [ -z "$1" ]; then
  files="`find . -type f -name '*.py' -maxdepth 1` `find ddpg -type f -name '*.py' -maxdepth 1` common/misc_util.py common/dataset.py"
else
  files=$1
fi


for f in $files; do
  echo "****$f*****"
  pycodestyle --first --show-pep8 --ignore=E114,E111 --max-line-length 120 $f
  pylint --disable=bad-indentation,no-member,no-name-in-module,too-many-public-methods,too-many-arguments,too-many-instance-attributes,too-many-locals,line-too-long,missing-docstring,invalid-name,too-many-branches,too-many-statements --reports=n $f
done
