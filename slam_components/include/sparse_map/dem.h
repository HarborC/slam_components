#pragma once

#include <Eigen/Dense>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <vector>

struct DEM {
  DEM() = default;
  DEM(const double &_grid_resolution,
      const std::vector<Eigen::Vector3d> &_pointcloud);

  void fillWithPlane(const std::vector<double> &plane_coefficients);

  void saveAsArcGrid(const std::string &filename);

  bool loadFromArcGrid(const std::string &filename);

  void save2(const std::string &filename);

  void print();

  double grid_resolution;
  double minX, minY;
  std::vector<std::vector<double>> data;
  std::vector<std::vector<bool>> value_mask;
  long long nodata_value = -99999;
};