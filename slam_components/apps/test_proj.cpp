#include <Eigen/Core>
#include <iostream>
#include <proj.h>

int main() {

  double latitude = 35.3059;
  double longitude = 110.368;
  PJ_CONTEXT *context = proj_context_create();

  // 定义源坐标系 (WGS 84 经纬度, EPSG:4326)
  PJ *wgs84 = proj_create(context, "EPSG:4326");

  if (!wgs84) {
    std::cerr << "Error: Failed to create WGS 84 object!" << std::endl;
    return 0;
  }

  // 定义目标坐标系 (Pseudo-Mercator, EPSG:3857)
  PJ *mercator = proj_create(context, "EPSG:3857");

  if (!mercator) {
    std::cerr << "Error: Failed to create Mercator object!" << std::endl;
    return 0;
  }

  // 创建从 WGS 84 到 EPSG:3857 的转换对象
  PJ *transformation =
      proj_create_crs_to_crs(context, "EPSG:4326", "EPSG:3857", NULL);

  if (transformation == NULL) {
    std::cerr << "Error: Failed to create transformation object!" << std::endl;
    return 0; // 或其他错误处理
  }

  // 定义经纬度作为输入 (注意：WGS 84 坐标顺序是 经度, 纬度)
  std::cout << "latitude: " << latitude << ", longitude: " << longitude
            << std::endl;
  PJ_COORD input_coord = proj_coord(latitude, longitude, 0, 0);

  // 执行转换：从 WGS 84 经纬度转换为 EPSG:3857
  PJ_COORD output_coord = proj_trans(transformation, PJ_FWD, input_coord);

  int err = proj_errno(transformation);
  if (err != 0) {
    const char *error_message = proj_errno_string(err);
    std::cerr << "Error: Projection transformation failed! Reason: "
              << error_message << std::endl;
    return 0;
  }

  Eigen::Vector3d xyz =
      Eigen::Vector3d(output_coord.xy.x, output_coord.xy.y, 0);

  std::cout << std::fixed << "x: " << xyz.x() << ", y: " << xyz.y()
            << std::endl;

  proj_destroy(wgs84);
  proj_destroy(mercator);
  proj_destroy(transformation);
  proj_context_destroy(context);
  return 0;
}
