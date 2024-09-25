#include <Eigen/Core>
#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>

#include "sparse_map/dem.h"
#include "sparse_map/matcher.h"
#include "sparse_map/sparse_map.h"
#include "utils/io_utils.h"
#include "utils/log_utils.h"

int main(int argc, char **argv) {
  DEM dem;
  dem.loadFromArcGrid("../../../datasets/TXPJ/test2/extract/dem.txt");
  dem.saveAsArcGrid("../../../datasets/TXPJ/test2/extract/dem3.txt");
  dem.save2("../../../datasets/TXPJ/test2/extract/dem2.bin");
  return 0;
}