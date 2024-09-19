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

#include "foxglove/visualizer.h"
#include "general_camera_model/function.hpp"

using namespace Eigen;

// Step 1: read images sequence from the directory
// Step 2: extract features from the images
// Step 3: match features between images

using namespace foxglove_viz;

#include <Eigen/Dense>
#include <iostream>

Eigen::Vector3d llhToECEF(double lat, double lon, double alt) {
  // WGS84楠球参数
  const double a = 6378137.0;
  const double b = 6356752.3142;
  const double f = (a - b) / a;
  double e2 = f * (2 - f);

  double N = a / std::sqrt(1 - e2 * std::sin(lat) * std::sin(lat));

  Eigen::Vector3d xyz;
  xyz(0) = (N + alt) * std::cos(lat) * std::cos(lon);
  xyz(1) = (N + alt) * std::cos(lat) * std::sin(lon);
  xyz(2) = (N * (1 - e2) + alt) * std::sin(lat);

  return xyz;
}

// llh to NUE(北天东)坐标系
Eigen::Vector3d llhToENU(double lat, double lon, double hei, double lat0,
                         double lon0, double hei0 = 0) {

  Eigen::Vector3d xyz0 = llhToECEF(lat0, lon0, hei0);

  Eigen::Vector3d xyz = llhToECEF(lat, lon, hei);

  Eigen::Vector3d dxyz = xyz - xyz0;

  Eigen::Matrix3d R;
  R << -std::sin(lon0), std::cos(lon0), 0, -std::sin(lat0) * std::cos(lon0),
      -std::sin(lat0) * std::sin(lon0), std::cos(lat0),
      std::cos(lat0) * std::cos(lon0), std::cos(lat0) * std::sin(lon0),
      std::sin(lat0);

  Eigen::Vector3d nue = R * dxyz;

  return nue;
}

Eigen::Matrix3d nueToBody(double yaw, double pitch, double roll) {
  Eigen::Matrix3d R;

  double cos_yaw = std::cos(yaw);
  double sin_yaw = std::sin(yaw);
  double cos_pitch = std::cos(pitch);
  double sin_pitch = std::sin(pitch);
  double cos_roll = std::cos(roll);
  double sin_roll = std::sin(roll);

  R(0, 0) = cos_pitch * cos_yaw;
  R(0, 1) = sin_pitch;
  R(0, 2) = -cos_pitch * sin_yaw;

  R(1, 0) = sin_yaw * sin_roll - cos_yaw * sin_pitch * cos_roll;
  R(1, 1) = cos_pitch * cos_roll;
  R(1, 2) = sin_yaw * cos_roll + sin_yaw * sin_roll * cos_yaw;

  R(2, 0) = sin_yaw * cos_roll + cos_yaw * sin_pitch * sin_roll;
  R(2, 1) = -cos_pitch * sin_roll;
  R(2, 2) = cos_yaw * cos_roll - sin_yaw * sin_pitch * sin_roll;

  return R;
}

Eigen::Matrix3d bodyToCamera(double qy, double qz) {
  Eigen::Matrix3d R;

  double cos_qy = std::cos(qy);
  double sin_qy = std::sin(qy);
  double cos_qz = std::cos(qz);
  double sin_qz = std::sin(qz);

  R(0, 0) = cos_qy * cos_qz;
  R(0, 1) = sin_qy;
  R(0, 2) = -cos_qy * sin_qz;

  R(1, 0) = -sin_qy * cos_qz;
  R(1, 1) = cos_qy;
  R(1, 2) = sin_qy * sin_qz;

  R(2, 0) = sin_qz;
  R(2, 1) = 0;
  R(2, 2) = cos_qz;

  return R;
}

std::map<std::string, std::vector<double>>
readCSV(const std::string &filename) {
  std::map<std::string, std::vector<double>> ins_data;

  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Error: Could not open the file " << filename << std::endl;
    return ins_data;
  }

  std::string line;
  std::getline(file, line);
  while (std::getline(file, line)) {
    if (line.empty()) {
      continue;
    }

    std::vector<std::string> items = Utils::StringSplit(line, ",");
    std::string name = items[0];
    std::vector<double> data;
    for (int i = 1; i < 9; i++) {
      data.push_back(std::stod(items[i]));
    }
    ins_data[name] = data;
  }

  file.close();

  return ins_data;
}

double distance(const cv::Point2f &p1, const cv::Point2f &p2) {
  return std::sqrt((p1.x - p2.x) * (p1.x - p2.x) +
                   (p1.y - p2.y) * (p1.y - p2.y));
}

cv::Mat getVaildMask(const std::vector<cv::Point2f> &pts, const cv::Mat &mask) {
  int min_window_size = 5;

  cv::Mat mask_updated = mask.clone();

  int width = mask_updated.cols;
  int height = mask_updated.rows;

  for (size_t i = 0; i < pts.size(); i++) {
    // Get current left keypoint, check that it is in bounds
    const cv::Point2f &pt = pts[i];
    int x = (int)pt.x;
    int y = (int)pt.y;
    if (x < min_window_size || x >= width - min_window_size ||
        y < min_window_size || y >= height - min_window_size) {
      continue;
    }

    // Now check if it is in a mask area or not
    // NOTE: mask has max value of 255 (white) if it should be
    if (mask.at<uint8_t>(y, x) < 127) {
      continue;
    }

    // Append this to the local mask of the image
    cv::Point pt1(x - min_window_size, y - min_window_size);
    cv::Point pt2(x + min_window_size, y + min_window_size);
    cv::rectangle(mask_updated, pt1, pt2, cv::Scalar(0));
  }

  return mask_updated.clone();
}

std::vector<std::pair<int, int>>
opticalFlow(const cv::Mat &prev_img, const cv::Mat &curr_img,
            const std::vector<cv::Point2f> &prev_pts,
            std::vector<cv::Point2f> &curr_pts, const cv::Mat &curr_mask,
            bool double_check = false) {
  int need_pts_num = 500;
  std::vector<std::pair<int, int>> matches;

  if (prev_pts.size()) {
    std::vector<uchar> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(prev_img, curr_img, prev_pts, curr_pts, status,
                             err, cv::Size(21, 21), 3);

    if (double_check) {
      std::vector<uchar> reverse_status;
      std::vector<cv::Point2f> reverse_pts = prev_pts;
      cv::calcOpticalFlowPyrLK(
          curr_img, prev_img, curr_pts, reverse_pts, reverse_status, err,
          cv::Size(21, 21), 1,
          cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30,
                           0.01),
          cv::OPTFLOW_USE_INITIAL_FLOW);

      for (size_t i = 0; i < status.size(); i++) {
        if (status[i] && reverse_status[i] &&
            distance(prev_pts[i], reverse_pts[i]) <= 0.5f)
          status[i] = 1;
        else
          status[i] = 0;
      }
    }

    std::vector<cv::Point2f> curr_pts2;
    for (int i = 0; i < status.size(); i++) {
      if (status[i]) {
        curr_pts2.push_back(curr_pts[i]);
        matches.push_back(std::make_pair(i, curr_pts2.size() - 1));
      }
    }
    curr_pts = curr_pts2;
  }

  cv::Mat mask = getVaildMask(curr_pts, curr_mask);

  if (curr_pts.size() + 20 < need_pts_num) {
    std::vector<cv::Point2f> add_pts;
    cv::goodFeaturesToTrack(curr_img, add_pts, need_pts_num - curr_pts.size(),
                            0.01, 10, mask);
    for (int i = 0; i < add_pts.size(); i++) {
      curr_pts.push_back(add_pts[i]);
    }
  }

  return matches;
}

std::vector<Eigen::Vector2d> toEigen(const std::vector<cv::Point2f> &pts) {
  std::vector<Eigen::Vector2d> eigen_pts(pts.size());
  for (int i = 0; i < pts.size(); i++) {
    eigen_pts[i] << pts[i].x, pts[i].y;
  }
  return eigen_pts;
}

Eigen::Vector3d R2Omega(const Eigen::Matrix3d &R) {
  Eigen::Vector3d euler;
  euler(0) = -std::atan2(R(0, 2), R(2, 2));
  euler(1) = std::asin(-R(1, 2));
  euler(2) = std::atan2(R(1, 0), R(1, 1));
  return euler;
}

Eigen::Matrix3d Omega2R(const Eigen::Vector3d &r) {
  Eigen::Matrix3d R1;
  R1 << cos(r(0)), 0, -sin(r(0)), 0, 1, 0, sin(r(0)), 0, cos(r(0));

  Eigen::Matrix3d R2;
  R2 << 1, 0, 0, 0, cos(r(1)), -sin(r(1)), 0, sin(r(1)), cos(r(1));

  Eigen::Matrix3d R3;
  R3 << cos(r(2)), -sin(r(2)), 0, sin(r(2)), cos(r(2)), 0, 0, 0, 1;

  return R1 * R2 * R3;
}

Eigen::Matrix3d RNormalized(const Eigen::Matrix3d &R) {
  Eigen::Quaterniond q(R);
  Eigen::Matrix3d R_normalized = q.normalized().toRotationMatrix();
  return R_normalized;
}

void preprocessImage(const cv::Mat &img, cv::Mat &img_processed,
                     cv::Mat &mask) {
  cv::Mat img_gray;
  if (img.channels() == 3) {
    cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
  } else {
    img_gray = img.clone();
  }
  cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();

  clahe->apply(img_gray, img_processed);

  mask = cv::Mat::ones(img.size(), CV_8UC1) * 255;

  for (int row = 0; row < img.rows; ++row) {
    for (int col = 0; col < img.cols; ++col) {
      cv::Vec3b pixelColor = img.at<cv::Vec3b>(row, col);
      if (pixelColor(0) != pixelColor(1) || pixelColor(0) != pixelColor(2)) {
        mask.at<uchar>(row, col) = 0;
      }
    }
  }

  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(10, 10));
  cv::erode(mask, mask, kernel);
}

bool solveRelativeRT(
    const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> &corres,
    Eigen::Matrix3d &Rotation, Eigen::Vector3d &Translation) {
  if (corres.size() >= 15) {
    std::vector<cv::Point2f> ll, rr;
    for (int i = 0; i < int(corres.size()); i++) {
      ll.push_back(cv::Point2f(corres[i].first(0) / corres[i].first(2),
                               corres[i].first(1) / corres[i].first(2)));
      rr.push_back(cv::Point2f(corres[i].second(0) / corres[i].second(2),
                               corres[i].second(1) / corres[i].second(2)));
    }
    cv::Mat mask;
    cv::Mat E =
        cv::findFundamentalMat(ll, rr, cv::FM_RANSAC, 0.3 / 460, 0.99, mask);
    cv::Mat cameraMatrix =
        (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
    cv::Mat rot, trans;
    int inlier_cnt = cv::recoverPose(E, ll, rr, cameraMatrix, rot, trans, mask);

    Eigen::Matrix3d R;
    Eigen::Vector3d T;
    for (int i = 0; i < 3; i++) {
      T(i) = trans.at<double>(i, 0);
      for (int j = 0; j < 3; j++)
        R(i, j) = rot.at<double>(i, j);
    }

    Rotation = R.transpose();
    Translation = -R.transpose() * T;
    if (inlier_cnt > 12)
      return true;
    else
      return false;
  }
  return false;
}

bool solveFrameByPnP(
    const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>
        &corres_2d_3d,
    Matrix3d &R_initial, Vector3d &P_initial) {
  std::vector<cv::Point2f> pts_2_vector;
  std::vector<cv::Point3f> pts_3_vector;

  size_t feature_num = corres_2d_3d.size();
  for (size_t j = 0; j < feature_num; j++) {
    Eigen::Vector3d pt_2d = corres_2d_3d[j].first;
    Eigen::Vector3d pt_3d = corres_2d_3d[j].second;

    cv::Point2f pts_2(pt_2d(0) / pt_2d(2), pt_2d(1) / pt_2d(2));
    pts_2_vector.push_back(pts_2);
    cv::Point3f pts_3(pt_3d(0), pt_3d(1), pt_3d(2));
    pts_3_vector.push_back(pts_3);
  }

  if (int(pts_2_vector.size()) < 15) {
    printf("unstable features tracking, please slowly move you device!\n");
    if (int(pts_2_vector.size()) < 10)
      return false;
  }

  cv::Mat r, rvec, t, D, tmp_r;
  cv::eigen2cv(R_initial, tmp_r);
  cv::Rodrigues(tmp_r, rvec);
  cv::eigen2cv(P_initial, t);
  cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
  bool pnp_succ;
  pnp_succ = cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1);
  if (!pnp_succ) {
    return false;
  }
  cv::Rodrigues(rvec, r);
  // cout << "r " << endl << r << endl;
  MatrixXd R_pnp;
  cv::cv2eigen(r, R_pnp);
  MatrixXd T_pnp;
  cv::cv2eigen(t, T_pnp);
  R_initial = R_pnp;
  P_initial = T_pnp;
  return true;
}

Eigen::Matrix4d getENUPose(const std::vector<double> &ins) {
  double lat0, lon0, alt0;
  lon0 = 110.365814 * M_PI / 180;
  lat0 = 35.304042 * M_PI / 180;
  alt0 = 0;

  Eigen::Matrix4d T_wc = Eigen::Matrix4d::Identity();
  double qy = ins[0] * M_PI / 180;
  double qz = ins[1] * M_PI / 180;
  double pitch = ins[2] * M_PI / 180;
  double yaw = ins[3] * M_PI / 180;
  double roll = ins[4] * M_PI / 180;
  double lon = ins[5] * M_PI / 180;
  double lat = ins[6] * M_PI / 180;
  double alt = ins[7];

  Eigen::Matrix3d R_nue2body = nueToBody(yaw, pitch, roll);
  Eigen::Matrix3d R_body2camera = bodyToCamera(qy, qz);
  Eigen::Vector3d t_enu = llhToENU(lat, lon, alt, lat0, lon0, alt0);
  Eigen::Matrix3d R_nue = R_nue2body * R_body2camera;

  Eigen::Matrix3d R_enu_nue;
  R_enu_nue << 0, 0, 1, 1, 0, 0, 0, 1, 0;

  Eigen::Matrix3d R_2;
  R_2 << -1, 0, 0, 0, -1, 0, 0, 0, 1;

  T_wc.block(0, 0, 3, 3) = RNormalized(R_enu_nue * R_nue * R_2);
  T_wc.block(0, 3, 3, 1) = t_enu;

  return T_wc;
}

DEM generateDEMByPlane(const std::vector<Eigen::Vector3d> &pointcloud,
                       const std::vector<double> &plane_coefficients,
                       double grid_resolution) {
  DEM dem(grid_resolution, pointcloud);
  dem.fillWithPlane(plane_coefficients);
  return dem;
}

int main(int argc, char **argv) {
  Visualizer server(8088);

  TimeTicToc time;

  std::string img_dir = "../../../datasets/TXPJ/test2/raw_data/img2_raw/";
  auto ins_data = readCSV("../../../datasets/TXPJ/test2/raw_data/ins2.csv");

  bool is_publised = false;

  // std::string img_dir = "../../../datasets/TXPJ/test1/img1/";
  // auto ins_data = readCSV("../../../datasets/TXPJ/test1/ins1.csv");

  std::vector<std::string> img_files = Utils::GetFileList(img_dir);
  std::sort(img_files.begin(), img_files.end());

  double scale = 4;
  int width = 4096 / scale;
  int height = 3072 / scale;
  double f = 33400 / scale;
  double cx = width / 2.0;
  double cy = height / 2.0;

  Calibration::Ptr calib(new Calibration());
  general_camera_model::GeneralCameraModel pinhole_camera =
      general_camera_model::getSimplePinhole(f, f, cx, cy, width, height);

  Camera::Ptr camera(new Camera());
  camera->setCameraModel(pinhole_camera.makeShared());

  calib->addCamera(camera);

  std::vector<Eigen::Matrix4f> poses_prior;

  SparseMap::Ptr flow_sparse_map(new SparseMap(calib, true));
  SparseMap::Ptr sparse_map(new SparseMap(calib, true));

  Frame::Ptr prev_flow_frame;
  cv::Mat prev_flow_img, curr_flow_img;
  std::vector<cv::Point2f> prev_flow_pts, curr_flow_pts;
  FrameIDType last_kf_id;

  bool is_init = false;

  std::map<FrameIDType, std::string> kf_id_and_name;
  for (int i = 0; i < img_files.size(); i++) {
    // read ins data
    std::string basename = Utils::GetPathBaseName(img_files[i]);

    // get ins prior pose
    std::vector<double> ins = ins_data[basename];
    Eigen::Matrix4d T_wc = getENUPose(ins);

    // optical flow
    Frame::Ptr frame(
        new Frame(flow_sparse_map->frame_next_id++, calib->camNum()));
    curr_flow_img = cv::imread(img_files[i], cv::IMREAD_COLOR);
    cv::resize(
        curr_flow_img, curr_flow_img,
        cv::Size(curr_flow_img.cols / scale, curr_flow_img.rows / scale));

    cv::Mat curr_flow_img_processed, curr_mask;
    preprocessImage(curr_flow_img, curr_flow_img_processed, curr_mask);

    auto flow_matches = opticalFlow(prev_flow_img, curr_flow_img_processed,
                                    prev_flow_pts, curr_flow_pts, curr_mask);

    std::vector<Eigen::Vector3d> flow_bearings;
    for (int j = 0; j < curr_flow_pts.size(); j++) {
      Eigen::Vector3d bearing;
      pinhole_camera.planeToSpace(
          Eigen::Vector2d(curr_flow_pts[j].x, curr_flow_pts[j].y), &bearing);
      flow_bearings.push_back(bearing);
    }

    frame->addData({curr_flow_img}, {toEigen(curr_flow_pts)}, {flow_bearings});
    flow_sparse_map->addKeyFrame(frame);
    if (flow_matches.size() > 0)
      flow_sparse_map->addIntraMatches(prev_flow_frame->id(), frame->id(),
                                       {flow_matches});

    {
      prev_flow_frame = frame;
      prev_flow_img = curr_flow_img_processed.clone();
      prev_flow_pts = curr_flow_pts;
    }

    // pick keyframe
    if (i > 0) {
      std::vector<std::pair<size_t, size_t>> matches =
          flow_sparse_map->getMatches(last_kf_id, 0, frame->id(), 0);
      double kf_num1 = flow_sparse_map->getKeypointSize(last_kf_id, 0);
      double kf_num2 = flow_sparse_map->getKeypointSize(frame->id(), 0);
      double ratio1 = matches.size() / kf_num1;
      double ratio2 = matches.size() / kf_num2;

      double TH_ratio = 0.5;
      if (ratio1 < TH_ratio || ratio2 < TH_ratio) {
        // std::cout << std::fixed << "keyframe: " << frame->id() << " ratio1: "
        // << ratio1 << " ratio2: " << ratio2 << std::endl;
      } else {
        continue;
      }
    }

    last_kf_id = frame->id();

    poses_prior.push_back(T_wc.cast<float>());

    Frame::Ptr new_frame(
        new Frame(sparse_map->frame_next_id++, calib->camNum()));
    new_frame->Twb_prior_ = T_wc;
    new_frame->extractFeature({curr_flow_img}, "ORB", {curr_mask});

    std::vector<std::vector<Eigen::Vector3d>> bearings(1);
    bearings[0].resize(new_frame->keypoints()[0].size());
    for (int j = 0; j < new_frame->keypoints()[0].size(); j++) {
      Eigen::Vector2d kp = new_frame->keypoints()[0][j];
      Eigen::Vector3d bearing;
      pinhole_camera.planeToSpace(kp, &bearing);
      bearings[0][j] = bearing;
    }
    new_frame->addData({}, {}, {bearings});

    new_frame->setBodyPose(T_wc);

    if (sparse_map->last_frame_) {
      auto prev_kf_id = sparse_map->last_frame_->id();
      Frame::Ptr prev_frame = sparse_map->getFrame(prev_kf_id);

      sparse_map->addKeyFrame(new_frame);

      sparse_map->matchTwoFrames(prev_kf_id, 0, new_frame->id(), 0);

      sparse_map->triangulate2();

      sparse_map->bundleAdjustment(true);

      sparse_map->matchByPolyArea(new_frame->id(), 0);

      sparse_map->triangulate2();

      sparse_map->bundleAdjustment(true);

      if (is_publised) {
        cv::Mat timg = frame->drawKeyPoint(0);
        server.showImage("image", frame->id(), timg);

        cv::Mat timg3 = flow_sparse_map->drawFlow(frame->id(), 0);
        server.showImage("flow_image", frame->id(), timg3);

        cv::Mat timg4 =
            sparse_map->drawFlow(new_frame->id(), 0, new_frame->id() - 1);
        server.showImage("flow_image2", new_frame->id(), timg4);

        std::vector<Eigen::Vector3d> world_points =
            sparse_map->getWorldPoints();
        std::vector<std::vector<float>> world_points_f;
        for (int i = 0; i < world_points.size(); i++) {
          std::vector<float> pt = {world_points[i].x(), world_points[i].y(),
                                   world_points[i].z()};
          world_points_f.push_back(pt);
        }

        cv::Mat timg2 = sparse_map->drawReprojKeyPoint(new_frame->id(), 0);
        server.showImage("reproj_image2", new_frame->id(), timg2);

        cv::Mat timg1 = sparse_map->drawReprojKeyPoint(new_frame->id() - 1, 0);
        server.showImage("reproj_image1", new_frame->id() - 1, timg1);

        // server.showImage("reproj_image3", new_frame->id(), timg9);
        // server.showImage("reproj_image4", new_frame->id()-1, timg8);

        server.showPointCloud("world_points", new_frame->id(), world_points_f,
                              {}, "ENU");

        server.showPath("ins_path", frame->id(), poses_prior, "ENU");

        std::vector<Eigen::Matrix4f> poses_camera;
        for (auto it = sparse_map->frame_map_.begin();
             it != sparse_map->frame_map_.end(); ++it) {
          Frame::Ptr frame = it->second;
          poses_camera.push_back(frame->getBodyPose().cast<float>());
        }
        server.showPath("cam_path", frame->id(), poses_camera, "ENU");
      }
    } else {
      sparse_map->addKeyFrame(new_frame);

      if (is_publised) {
        std::vector<Eigen::Matrix4f> poses_camera;
        for (auto it = sparse_map->frame_map_.begin();
             it != sparse_map->frame_map_.end(); ++it) {
          Frame::Ptr frame = it->second;
          poses_camera.push_back(frame->getBodyPose().cast<float>());
        }
        server.showPath("cam_path", frame->id(), poses_camera, "ENU");
      }
    }

    kf_id_and_name[new_frame->id()] = basename;
  }

  sparse_map->bundleAdjustment(false, 100);

  double all_time = time.toc();
  std::cout << "all time: " << all_time / 1000 << " s" << std::endl;

  sparse_map->save("../../../datasets/TXPJ/test2/extract/sparse_map.bin");

  std::ofstream out_file("../../../datasets/TXPJ/test2/extract/poses.txt");
  std::string save_dir = "../../../datasets/TXPJ/test2/extract/imgs/";
  out_file << std::fixed << kf_id_and_name.size() << " 1 0" << std::endl;
  out_file << "0      4096 	3072	1.0	33400	0      0      0      0 "
              "     0      0      0      0      0      0  "
           << std::endl;

  Eigen::Vector3d euler0 = Eigen::Vector3d::Zero();
  int index = 0;
  for (auto it = kf_id_and_name.begin(); it != kf_id_and_name.end(); it++) {
    std::string basename = it->second;
    Frame::Ptr new_frame = sparse_map->getFrame(it->first);
    Eigen::Matrix4d T_wc_new = new_frame->getBodyPose();

    Eigen::Matrix3d R_3;
    R_3 << 1, 0, 0, 0, -1, 0, 0, 0, -1;

    Eigen::Vector3d euler = R2Omega(T_wc_new.block(0, 0, 3, 3) * R_3);
    euler0 += euler;
    index++;
  }
  euler0 /= index;

  Eigen::Matrix4d T_wc0 =
      sparse_map->getFrame(kf_id_and_name.begin()->first)->getBodyPose();
  T_wc0.block(0, 0, 3, 3) = Omega2R(euler0);

  Eigen::Matrix4d T_wc0_inv = T_wc0.inverse();

  std::vector<Eigen::Vector3d> points;
  for (auto it = sparse_map->frame_map_.begin();
       it != sparse_map->frame_map_.end(); ++it) {
    Frame::Ptr frame = it->second;
    auto camera = calib->getCamera(0);
    auto camera_model = camera->getCameraModel();
    Eigen::Matrix4d Twc = frame->getBodyPose() * camera->getExtrinsic();
    Eigen::Vector3d twc = Twc.block<3, 1>(0, 3);
    Eigen::Matrix3d Rwc = Twc.block<3, 3>(0, 0);

    std::vector<Eigen::Vector2d> corner_points = camera_model->getCornerPoints();
    for (int i = 0; i < corner_points.size(); ++i) {
      Eigen::Vector3d bearing;
      camera_model->planeToSpace(corner_points[i], &bearing);
      Eigen::Vector3d bearing0 = Rwc * bearing.normalized();
      double fi = -twc(2) / bearing0(2);
      double x = twc(0) + fi * bearing0(0);
      double y = twc(1) + fi * bearing0(1);
      points.push_back(Eigen::Vector3d(x, y, 0));
    }
  }

  std::vector<Eigen::Vector3d> new_points;
  for (int i = 0; i < points.size(); i++) {
    Eigen::Vector3d pt = points[i];
    pt = T_wc0_inv.block(0, 0, 3, 3) * pt + T_wc0_inv.block(0, 3, 3, 1);
    new_points.push_back(pt);
  }

  Eigen::Vector3d normal(0, 0, 1);
  Eigen::Vector3d new_normal = T_wc0_inv.block(0, 0, 3, 3) * normal;
  Eigen::Vector3d new_center = T_wc0_inv.block(0, 3, 3, 1);
  double new_d = -new_normal.dot(new_center);
  std::vector<double> plane_coefficients = {new_normal.x(), new_normal.y(),
                                            new_normal.z(), new_d};

  auto dem = generateDEMByPlane(new_points, plane_coefficients, 5);
  dem.saveAsArcGrid("../../../datasets/TXPJ/test2/extract/dem.grd");

  index = 0;
  for (auto it = kf_id_and_name.begin(); it != kf_id_and_name.end(); it++) {
    std::string basename = it->second;
    Frame::Ptr new_frame = sparse_map->getFrame(it->first);
    Eigen::Matrix4d T_wc_new = new_frame->getBodyPose();

    Eigen::Matrix3d R_3;
    R_3 << 1, 0, 0, 0, -1, 0, 0, 0, -1;

    T_wc_new.block(0, 0, 3, 3) = T_wc_new.block(0, 0, 3, 3) * R_3;
    T_wc_new = T_wc0_inv * T_wc_new;

    Eigen::Vector3d euler = R2Omega(T_wc_new.block(0, 0, 3, 3));

    out_file << index++ << " 0 0 " << std::endl;
    out_file << basename << std::endl;
    out_file << " 0                   0                   0                   "
                "0                   0                   0                   0 "
                "                  0                   0"
             << std::endl;
    out_file << std::fixed << " " << std::setprecision(6) << T_wc_new(0, 3)
             << " " << T_wc_new(1, 3) << " " << T_wc_new(2, 3) << " ";
    out_file << std::fixed << std::setprecision(6) << euler(0) << " "
             << euler(1) << " " << euler(2) << " " << std::endl;

    std::string img_path = img_dir + "/" + basename;
    cv::Mat img = cv::imread(img_path, cv::IMREAD_COLOR);
    cv::imwrite(save_dir + "/" + basename, img);

    cv::Mat timg1 = sparse_map->drawReprojKeyPoint(new_frame->id(), 0);
    cv::imwrite("../../../datasets/TXPJ/test2/extract/imgs_reproj/" +
                    std::to_string(new_frame->id()) + ".png",
                timg1);
  }
  out_file.close();

  std::ofstream out_file2("../../../datasets/TXPJ/test2/extract/points.txt");
  out_file2 << "# (id x y z): (img_name, pt_x, pt_y) ...\n";
  for (auto it = sparse_map->feature_map_.begin();
       it != sparse_map->feature_map_.end(); ++it) {
    Feature::Ptr feature = it->second;
    if (!feature->isTriangulated())
      continue;

    if (feature->observationSize() < 2)
      continue;

    out_file2 << feature->id() << " " << feature->getWorldPoint().transpose()
              << ": ";

    for (const auto &obs : feature->observations()) {
      FrameIDType frame_id = obs.first;
      Frame::Ptr frame = sparse_map->frame_map_[frame_id];
      for (auto cam_obs : obs.second) {
        int cam_id = cam_obs.first;
        int pt_id = cam_obs.second;
        out_file2 << "(" << kf_id_and_name[frame_id] << " "
                  << frame->keypoints()[cam_id][pt_id].transpose() * scale
                  << ") ";
      }
    }

    out_file2 << std::endl;
  }
  out_file2.close();

  while (1) {
    std::vector<Eigen::Vector3d> world_points = sparse_map->getWorldPoints();
    std::vector<std::vector<float>> world_points_f;
    for (int i = 0; i < world_points.size(); i++) {
      std::vector<float> pt = {world_points[i].x(), world_points[i].y(),
                               world_points[i].z()};
      world_points_f.push_back(pt);
    }

    server.showPointCloud("world_points", 1, world_points_f, {}, "ENU");

    server.showPath("ins_path", 1, poses_prior, "ENU");

    std::vector<Eigen::Matrix4f> poses_camera;
    for (auto it = sparse_map->frame_map_.begin();
         it != sparse_map->frame_map_.end(); ++it) {
      Frame::Ptr frame = it->second;
      poses_camera.push_back(frame->getBodyPose().cast<float>());
    }
    server.showPath("cam_path", 1, poses_camera, "ENU");

    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  }

  return 0;
}

// if (!is_init) {
//     Eigen::Matrix3d Rotation;
//     Eigen::Vector3d Translation;
//     bool success =
//     solveRelativeRT(sparse_map->getCorrespondences2D2D(prev_kf_id, 0,
//     new_frame->id(), 0), Rotation, Translation);

//     Eigen::Matrix4d prev_T = prev_frame->getBodyPose();

//     Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
//     T.block(0, 0, 3, 3) = Rotation;
//     T.block(0, 3, 3, 1) = Translation;
//     Eigen::Matrix4d curr_T = prev_T * T;

//     new_frame->setBodyPose(curr_T);
//     new_frame->Tcw_[0] = curr_T.inverse();

//     // std::cout << "1:\n" << new_frame->getBodyPose() << std::endl;
//     // std::cout << "2:\n" << prev_frame->Twb_prior_.inverse() *
//     new_frame->Twb_prior_ << std::endl;

//     // std::exit(0);

//     // if (!success) {
//     //     std::cout << "solve relative RT failed" << std::endl;
//     //     std::exit(0);
//     // } else {
//     //     std::cout << "solve relative RT success" << std::endl;
//     //     std::exit(0);
//     // }

//     is_init = true;
// } else {
//     Eigen::Matrix3d Rotation;
//     Eigen::Vector3d Translation;
//     bool success =
//     solveFrameByPnP(sparse_map->getCorrespondences2D3D(new_frame->id(), 0),
//     Rotation, Translation);

//     if (!success) {
//         std::cout << "solve Frame By PnP failed" << std::endl;
//         std::exit(0);
//         new_frame->setBodyPose(prev_frame->getBodyPose());
//         new_frame->Tcw_[0] = prev_frame->getBodyPose().inverse();
//     } else {
//         Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
//         T.block(0, 0, 3, 3) = Rotation;
//         T.block(0, 3, 3, 1) = Translation;
//         new_frame->setBodyPose(T);
//         new_frame->Tcw_[0] = T.inverse();
//     }

//     // std::cout << "1:\n" << new_frame->getBodyPose() << std::endl;
//     // std::cout << "2:\n" << prev_frame->Twb_prior_.inverse() *
//     new_frame->Twb_prior_ << std::endl;

//     // if (!success) {
//     //     std::cout << "solve relative RT failed" << std::endl;
//     //     std::exit(0);
//     // } else {
//     //     std::cout << "solve relative RT success" << std::endl;
//     //     std::exit(0);
//     // }
// }