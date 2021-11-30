#!/bin/bash

work_dir='/tmp/dir_test'
# 注意这里是没有空格的
if [[ ! -d ${work_dir} ]]; then
  mkdir ${work_dir}
else
  echo "file existed"
fi

echo "mkdir successful"

pids=$(ps -elf | grep python | grep -v 'grep' | awk '{printf("%s ",$2)}')

for pid in ${pids}; do
  echo "It is ${pid}"
done
