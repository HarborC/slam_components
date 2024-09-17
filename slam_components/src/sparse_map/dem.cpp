#include "sparse_map/dem.h"
#include "utils/io_utils.h"

DEM::DEM(const double &_grid_resolution,
         const std::vector<Eigen::Vector3d> &_pointcloud) {
  grid_resolution = _grid_resolution;
  minX = minY = std::numeric_limits<double>::max();
  double maxX = -std::numeric_limits<double>::max();
  double maxY = -std::numeric_limits<double>::max();
  for (const auto &pt : _pointcloud) {
    minX = std::min(minX, pt(0));
    maxX = std::max(maxX, pt(0));
    minY = std::min(minY, pt(1));
    maxY = std::max(maxY, pt(1));
  }

  int grid_height = (maxY - minY) / grid_resolution + 1;
  int grid_width = (maxX - minX) / grid_resolution + 1;
  data.resize(
      grid_height,
      std::vector<double>(grid_width, std::numeric_limits<double>::lowest()));
  value_mask.resize(grid_height, std::vector<bool>(grid_width, false));
  for (const auto &pt : _pointcloud) {
    int ix = static_cast<int>((pt.x() - minX) / grid_resolution);
    int iy = static_cast<int>((pt.y() - minY) / grid_resolution);

    // Use the highest Z value (elevation)
    data[iy][ix] = std::max(data[iy][ix], pt.z());
    value_mask[iy][ix] = true;

    if (nodata_value > pt.z()) {
      nodata_value = pt.z() - 10000;
    }
  }
}

void DEM::fillWithPlane(const std::vector<double> &plane_coefficients) {
  double a = plane_coefficients[0];
  double b = plane_coefficients[1];
  double c = plane_coefficients[2];
  double d = plane_coefficients[3];

  for (int i = 0; i < data.size(); i++) {
    for (int j = 0; j < data[i].size(); j++) {
      double x = minX + j * grid_resolution;
      double y = minY + i * grid_resolution;
      double z = (-a * x - b * y - d) / c;
      data[i][j] = z;
      value_mask[i][j] = true;

      if (nodata_value > z) {
        nodata_value = z - 10000;
      }
    }
  }
}

void DEM::saveAsArcGrid(const std::string &filename) {
  std::ofstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Error: Could not open the file " << filename << std::endl;
    return;
  }

  file << "ncols " << data[0].size() << std::endl;
  file << "nrows " << data.size() << std::endl;
  file << "xllcorner " << minX << std::endl;
  file << "yllcorner " << minY << std::endl;
  file << "cellsize " << grid_resolution << std::endl;
  file << "NODATA_value " << nodata_value << std::endl;

  for (int i = 0; i < data.size(); i++) {
    for (int j = 0; j < data[i].size(); j++) {
      if (value_mask[i][j]) {
        file << data[i][j] << " ";
      } else {
        file << nodata_value << " ";
      }
    }
    file << std::endl;
  }

  file.close();
  std::cout << "DEM saved to " << filename << std::endl;
}

bool DEM::loadFromArcGrid(const std::string &filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cout << "Error: Could not open the file " << filename << std::endl;
    return false;
  }

  int ncols, nrows;
  int idx = 0;

  std::string line;
  while (std::getline(file, line)) {
    if (line.empty()) {
      continue;
    }

    std::vector<std::string> items = Utils::StringSplit(line, " ");
    if (items[0] == "ncols") {
      ncols = std::stoi(items[1]);
    } else if (items[0] == "nrows") {
      nrows = std::stoi(items[1]);
      data.resize(nrows);
      value_mask.resize(nrows);
      for (int i = 0; i < data.size(); i++) {
        data[i].resize(ncols);
        value_mask[i].resize(ncols);
      }
    } else if (items[0] == "xllcorner") {
      minX = std::stod(items[1]);
    } else if (items[0] == "yllcorner") {
      minY = std::stod(items[1]);
    } else if (items[0] == "cellsize") {
      grid_resolution = std::stod(items[1]);
    } else if (items[0] == "NODATA_value") {
      nodata_value = std::stod(items[1]);
    } else {
      for (int i = 0; i < ncols; i++) {
        double value = std::stod(items[i]);
        if (value != nodata_value) {
          data[idx][i] = value;
          value_mask[idx][i] = true;
        } else {
          data[idx][i] = nodata_value;
          value_mask[idx][i] = false;
        }
      }
      idx++;
    }
  }
  file.close();
  std::cout << "DEM loaded from " << filename << std::endl;
  print();
  return true;
}

void DEM::save2(const std::string &filename) {
  FILE *fp = fopen(filename.c_str(), "wb");
  if (!fp) {
    printf("无法打开文件进行写入。\n");
    return;
  }

  int nRow = data.size();
  int nCol = data[0].size();
  double X0 = minX;
  double Y0 = minY;
  double dX = grid_resolution;
  double dY = grid_resolution;

  // 计算最大最小高程
  float minZ = std::numeric_limits<float>::max();
  float maxZ = std::numeric_limits<float>::lowest();
  float avgZ = 0;
  int count = 0;

  for (int i = 0; i < nRow; i++) {
    for (int j = 0; j < nCol; j++) {
      if (value_mask[i][j]) {
        minZ = std::min(minZ, float(data[i][j]));
        maxZ = std::max(maxZ, float(data[i][j]));
        avgZ += data[i][j];
        count++;
      }
    }
  }

  avgZ /= count;

  // 写入头部信息
  fwrite(&nCol, sizeof(int), 1, fp);
  fwrite(&nRow, sizeof(int), 1, fp);
  fwrite(&X0, sizeof(double), 1, fp);
  fwrite(&Y0, sizeof(double), 1, fp);
  fwrite(&dX, sizeof(double), 1, fp);
  fwrite(&dY, sizeof(double), 1, fp);
  fwrite(&avgZ, sizeof(float), 1, fp);
  fwrite(&minZ, sizeof(float), 1, fp);
  fwrite(&maxZ, sizeof(float), 1, fp);

  // 写入高程数据
  for (int i = 0; i < nRow; i++) {
    for (int j = 0; j < nCol; j++) {
      float z = value_mask[i][j] ? data[i][j] : nodata_value;
      fwrite(&z, sizeof(float), 1, fp);
    }
  }

  fclose(fp);
  printf("DEM saved to %s\n", filename.c_str());
}

void DEM::print() {
  std::cout << std::fixed << "Grid resolution: " << grid_resolution
            << std::endl;
  std::cout << std::fixed << "MinX: " << minX << " MinY: " << minY << std::endl;
  std::cout << std::fixed << "Nodata value: " << nodata_value << std::endl;
  std::cout << std::fixed << "Data(" << data.size() << "," << data[0].size()
            << ")" << std::endl;
}
