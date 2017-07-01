cur_dir=$(cd "$(dirname $0)";pwd)
echo ${cur_dir}

mkldnn_path=${cur_dir}/../external/mkldnn/install/lib
echo ${mkldnn_path}

export PYTHONPATH=$PYTHONPATH:${cur_dir}/../python
export LD_LIBRARY_PATH=${mkldnn_path}:$LD_LIBRARY_PATH
export KMP_AFFINITY=compact,1,0,granularity=fine
echo $@
../build/tools/caffe time -model=./test.prototxt -iterations=20  $@
