#!/bin/bash

BASE_DIR=$(cd $(dirname $0);pwd)
echo "BASE_DIR: ${BASE_DIR}"

cd ${BASE_DIR}/

mkdir build
cd build
cmake ..
make -j
make install

# rm -r ../../../tmp/test/*
# ./slam_components/apps/seq_data_test

# cd ..
# rm -rf build

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/mnt/i/project/slam/tmp/libtorch/lib