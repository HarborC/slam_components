#include <string>
#include <iostream>
#include <vector>
#include <Eigen/Core>
#include <unistd.h>

#include "utils/io_utils.h"
#include "utils/log_utils.h"
#include "sparse_map/sparse_map.h"
#include "sparse_map/matcher.h"
#include "sparse_map/dem.h"

int main (int argc, char** argv) {
    DEM dem;
    dem.load("../../../datasets/TXPJ/test2/extract/dem.txt");
    dem.save("../../../datasets/TXPJ/test2/extract/dem3.txt");
    dem.save2("../../../datasets/TXPJ/test2/extract/dem2.bin");
    return 0;
}