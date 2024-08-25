#!/bin/bash

BASE_DIR=$(cd $(dirname $0);pwd)
echo "BASE_DIR: ${BASE_DIR}"

source activate && conda activate slam4labeling 

cd ${BASE_DIR}/

# rm -rf build
mkdir build
cd build
cmake ..
make -j
make install
