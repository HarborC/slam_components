#!/bin/bash

BASE_DIR=$(cd $(dirname $0);pwd)
echo "BASE_DIR: ${BASE_DIR}"

cd ${BASE_DIR}/

clang-format -i ./slam_components/include/components/*.h
clang-format -i ./slam_components/include/components/network/droid_net/*.h
clang-format -i ./slam_components/src/components/*.cpp
clang-format -i ./slam_components/src/components/network/droid_net/*
clang-format -i ./slam_components/slam_components/apps/*.cpp