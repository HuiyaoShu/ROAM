#! /usr/bin/sh
export LD_LIBRARY_PATH=""
export GRB_LICENSE_FILE=""

usage() {
  echo "Usage: $0 [options]
    options:
      --model   models to be benchmarked
      --batch-size  batch sizes to be benchmarked
      --generate-address  optimize address for tensors
      --skip-node-ordering  execute no optimization, provided for pytorch profile mode
      --gpu-profile   profile pytorch eager mode
  "
}

args=""
for x in "$*"
do
  args=$args"$x"
done


python3 ./baselines/no_acc_MODeL/benchmarks.py $args