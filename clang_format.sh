#!/bin/bash

BASE_DIR=$(cd $(dirname $0);pwd)
echo "BASE_DIR: ${BASE_DIR}"

cd ${BASE_DIR}/

clang-format -i ./slam_components/include/components/*.h
clang-format -i ./slam_components/include/components/network/droid/*.h
clang-format -i ./slam_components/include/components/network/superpoint/*.h
clang-format -i ./slam_components/include/components/network/*.h
clang-format -i ./slam_components/src/components/*.cpp
clang-format -i ./slam_components/src/components/network/droid/*
clang-format -i ./slam_components/src/components/network/superpoint/*
clang-format -i ./slam_components/src/components/network/*.cpp
clang-format -i ./slam_components/apps/*.cpp
clang-format -i ./slam_components/src/utils/*.cpp
clang-format -i ./slam_components/include/utils/*.h
clang-format -i ./slam_components/include/components/utils/*.h