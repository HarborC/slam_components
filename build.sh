#!/bin/bash

BASE_DIR=$(cd $(dirname $0);pwd)
echo "BASE_DIR: ${BASE_DIR}"

cd ${BASE_DIR}/

mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=${BASE_DIR}/../../tmp/libtorch .. 
make -j 
make install
# cd ..
# rm -rf build

# rm -r ../../../tmp/test/*
# ./slam_components/apps/eruco_val
./slam_components/apps/seq_data_test