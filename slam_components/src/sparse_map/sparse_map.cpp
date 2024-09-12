#include "sparse_map/sparse_map.h"
#include "sparse_map/ceres_func.h"

void SparseMap::addKeyFrame(
    const FrameIDType &id, const std::vector<cv::Mat> &imgs,
    const std::vector<std::vector<Eigen::Vector2d>> &keypoints,
    const std::vector<cv::Mat> &descriptors) {
  assert(imgs.size() == calibration_->camNum());
  Frame::Ptr cur_frame(new Frame(id));

  std::vector<std::vector<Eigen::Vector3d>> bearings;
  for (int cam_id = 0; cam_id < calibration_->camNum(); ++cam_id) {
    const auto &cam = calibration_->getCamera(cam_id);
    std::vector<Eigen::Vector3d> bearings_cam;
    for (size_t pt_id = 0; pt_id < keypoints[cam_id].size(); ++pt_id) {
      Eigen::Vector2d pt = keypoints[cam_id][pt_id];
      Eigen::Vector3d bearing;
      cam.getCameraModel().planeToSpace(pt, &bearing);
      bearings_cam.push_back(bearing);
    }
    bearings.push_back(bearings_cam);
  }

  cur_frame->addData(imgs, keypoints, bearings, descriptors);
  frame_map_[cur_frame->id()] = cur_frame;
  last_frame_ = cur_frame;
}

void SparseMap::addKeyFrame(const Frame::Ptr &frame) {
  if (!frame) {
    std::cerr << "Error: frame is nullptr" << std::endl;
    return;
  }

  if (frame->camNum() != calibration_->camNum()) {
    std::cerr << "Error: camNum() != calibration_->camNum()" << std::endl;
    return;
  }

  frame_map_[frame->id()] = frame;
  last_frame_ = frame;
}

void SparseMap::removeInvalidFeature() {
  for (auto it = feature_map_.begin(); it != feature_map_.end();) {
    auto feature = it->second;
    if (!feature->isValid()) {
      it = feature_map_.erase(it);
    } else {
      it++;
    }
  }
}

void SparseMap::removeKeyFrame(const FrameIDType &id) {
  if (frame_map_.find(id) == frame_map_.end()) {
    std::cerr << "Error: frame id not found" << std::endl;
    return;
  }

  Frame::Ptr cur_frame = frame_map_[id];
  for (int cam_id = 0; cam_id < cur_frame->feature_ids().size(); ++cam_id) {
    for (int pt_id = 0; pt_id < cur_frame->feature_ids()[cam_id].size();
         ++pt_id) {
      FeatureIDType ft_id = cur_frame->feature_ids()[cam_id][pt_id];
      if (ft_id < 0)
        continue;
      Feature::Ptr feature = feature_map_[ft_id];
      feature->removeObservationByFrameId(id);
    }
  }

  removeInvalidFeature();

  frame_map_.erase(id);
}

void SparseMap::removeFeature(const FeatureIDType &id) {
  if (feature_map_.find(id) == feature_map_.end()) {
    std::cerr << "Error: feature id not found" << std::endl;
    return;
  }

  Feature::Ptr feature = feature_map_[id];
  for (const auto &obs : feature->observations()) {
    FrameIDType frame_id = obs.first;
    if (frame_map_.find(frame_id) == frame_map_.end()) {
      std::cerr << "Error: frame id not found" << std::endl;
      continue;
    }

    Frame::Ptr frame = frame_map_[frame_id];
    for (const auto &cam_obs : obs.second) {
      int cam_id = cam_obs.first;
      int pt_id = cam_obs.second;
      frame->setFeatureID(cam_id, pt_id, -1);
    }
  }

  feature_map_.erase(id);
}

void SparseMap::addInterMatches(
    const FrameIDType &cur_frame_id,
    const std::vector<std::pair<int, int>> &stereo_ids,
    std::vector<std::vector<std::pair<int, int>>> inter_matches) {
  assert(stereo_ids.size() == inter_matches.size());
  Frame::Ptr cur_frame = frame_map_[cur_frame_id];

  for (size_t pair_id = 0; pair_id < inter_matches.size(); ++pair_id) {
    int cam_id0 = stereo_ids[pair_id].first;
    int cam_id1 = stereo_ids[pair_id].second;
    for (size_t match_id = 0; match_id < inter_matches[pair_id].size();
         ++match_id) {
      int left_pt_id = inter_matches[pair_id][match_id].first;
      int right_pt_id = inter_matches[pair_id][match_id].second;
    }
  }

  // ransac
  if (use_ransac_) {
    for (size_t pair_id = 0; pair_id < stereo_ids.size(); ++pair_id) {
      int cam_id0 = stereo_ids[pair_id].first;
      int cam_id1 = stereo_ids[pair_id].second;
      ransacWithF(cur_frame_id, cam_id0, cur_frame_id, cam_id1,
                  inter_matches[pair_id]);
    }
  }

  // add observations (inter-frame)
  for (size_t pair_id = 0; pair_id < inter_matches.size(); ++pair_id) {
    int cam_id0 = stereo_ids[pair_id].first;
    int cam_id1 = stereo_ids[pair_id].second;
    for (size_t match_id = 0; match_id < inter_matches[pair_id].size();
         ++match_id) {
      int left_pt_id = inter_matches[pair_id][match_id].first;
      int right_pt_id = inter_matches[pair_id][match_id].second;

      Feature::Ptr left_feature, right_feature;
      FeatureIDType left_ft_id = cur_frame->feature_ids()[cam_id0][left_pt_id];
      FeatureIDType right_ft_id =
          cur_frame->feature_ids()[cam_id1][right_pt_id];

      if (left_ft_id < 0 && right_ft_id < 0) {
        Feature::Ptr new_feature(new Feature(feature_next_id++));
        feature_map_[new_feature->id()] = new_feature;
        left_feature = new_feature;
        right_feature = new_feature;
      } else if (left_ft_id >= 0 && right_ft_id < 0) {
        left_feature = feature_map_[left_ft_id];
        right_feature = left_feature;
      } else if (left_ft_id < 0 && right_ft_id >= 0) {
        right_feature = feature_map_[right_ft_id];
        left_feature = right_feature;
      } else {
        left_feature = feature_map_[left_ft_id];
        right_feature = feature_map_[right_ft_id];
      }

      cur_frame->setFeatureID(cam_id0, left_pt_id, left_feature->id());
      cur_frame->setFeatureID(cam_id1, right_pt_id, right_feature->id());
      left_feature->addObservation(cur_frame_id, cam_id0, left_pt_id);
      right_feature->addObservation(cur_frame_id, cam_id1, right_pt_id);
      if (left_feature->id() != right_feature->id()) {
        left_feature->addObservation(cur_frame_id, cam_id1, right_pt_id);
        right_feature->addObservation(cur_frame_id, cam_id0, left_pt_id);
      }
    }
  }
}

void SparseMap::addIntraMatches(
    const FrameIDType &pre_frame_id, const FrameIDType &cur_frame_id,
    std::vector<std::vector<std::pair<int, int>>> intra_matches) {
  Frame::Ptr cur_frame = frame_map_[cur_frame_id];
  Frame::Ptr pre_frame = frame_map_[pre_frame_id];

  // ransac
  if (use_ransac_) {
    for (size_t cam_id = 0; cam_id < intra_matches.size(); ++cam_id) {
      ransacWithF(pre_frame_id, cam_id, cur_frame_id, cam_id,
                  intra_matches[cam_id]);
    }
  }

  // add observations (intra-frame)
  for (size_t cam_id = 0; cam_id < intra_matches.size(); ++cam_id) {
    for (size_t match_id = 0; match_id < intra_matches[cam_id].size();
         ++match_id) {
      int pre_pt_id = intra_matches[cam_id][match_id].first;
      int cur_pt_id = intra_matches[cam_id][match_id].second;

      Feature::Ptr feature;

      FeatureIDType pre_ft_id = pre_frame->feature_ids()[cam_id][pre_pt_id];
      if (pre_ft_id < 0) {
        Feature::Ptr new_feature(new Feature(feature_next_id++));
        feature_map_[new_feature->id()] = new_feature;
        feature = new_feature;
      } else {
        feature = feature_map_[pre_ft_id];
      }

      pre_frame->setFeatureID(cam_id, pre_pt_id, feature->id());
      cur_frame->setFeatureID(cam_id, cur_pt_id, feature->id());
      feature->addObservation(pre_frame_id, cam_id, pre_pt_id);
      feature->addObservation(cur_frame_id, cam_id, cur_pt_id);
    }
  }
}

void SparseMap::updateKeyFramePose(const FrameIDType &id,
                                   const Eigen::Matrix4d &pose) {
  frame_map_[id]->setBodyPose(pose);
}

void triangulatePoint(const std::vector<Eigen::Matrix4d> &posecws,
                      const std::vector<Eigen::Vector3d> &bearings,
                      Eigen::Vector3d *pw) {
  assert(posecws.size() == bearings.size());
  int num = posecws.size();
  Eigen::MatrixXd A(2 * num, 4);
  for (int i = 0; i < num; ++i) {
    A.row(2 * i) =
        bearings[i](0) * posecws[i].row(2) - bearings[i](2) * posecws[i].row(0);
    A.row(2 * i + 1) =
        bearings[i](1) * posecws[i].row(2) - bearings[i](2) * posecws[i].row(1);
  }
  Eigen::JacobiSVD<Eigen::MatrixXd> mySVD(A, Eigen::ComputeFullV);
  (*pw)(0) = mySVD.matrixV()(0, 3);
  (*pw)(1) = mySVD.matrixV()(1, 3);
  (*pw)(2) = mySVD.matrixV()(2, 3);
  (*pw) = (*pw) / mySVD.matrixV()(3, 3);
}

void SparseMap::triangulate() {
  for (auto it = frame_map_.begin(); it != frame_map_.end(); ++it) {
    Frame::Ptr frame = it->second;
    Eigen::Matrix4d pose = frame->getBodyPose();
    frame->Tcw_.resize(frame->camNum());
    for (int cam_id = 0; cam_id < frame->camNum(); ++cam_id) {
      frame->Tcw_[cam_id] =
          (pose * calibration_->getCamera(cam_id).getExtrinsic()).inverse();
    }
  }

  for (auto it = feature_map_.begin(); it != feature_map_.end(); ++it) {
    Feature::Ptr feature = it->second;
    if (feature->isTriangulated())
      continue;

    if (feature->observationSize() < 2)
      continue;

    std::vector<Eigen::Vector3d> bearings;
    std::vector<Eigen::Matrix4d> Tcws;

    for (const auto &obs : feature->observations()) {
      FrameIDType frame_id = obs.first;
      Frame::Ptr frame = frame_map_[frame_id];
      for (const auto &cam_obs : obs.second) {
        int cam_id = cam_obs.first;
        int pt_id = cam_obs.second;
        bearings.push_back(frame->bearings()[cam_id][pt_id].normalized());
        Tcws.push_back(frame->Tcw_[cam_id]);
      }
    }

    Eigen::Vector3d world_point;
    triangulatePoint(Tcws, bearings, &world_point);

    feature->setWorldPoint(world_point);
    Frame::Ptr ref_frame = frame_map_[feature->refFrameId()];
    Eigen::Matrix4d Tcw = ref_frame->Tcw_[feature->refCamId()];
    Eigen::Vector3d camera_point =
        Tcw.block<3, 3>(0, 0) * feature->getWorldPoint() +
        Tcw.block<3, 1>(0, 3);

    if (camera_point.z() < 0) {
      feature->setInvDepth(1.0 / DEFAULT_DEPTH);
    } else {
      feature->setInvDepth(1.0 / camera_point.z());
    }
  }
}

void SparseMap::triangulate2() {
  for (auto it = frame_map_.begin(); it != frame_map_.end(); ++it) {
    Frame::Ptr frame = it->second;
    Eigen::Matrix4d pose = frame->getBodyPose();
    frame->Tcw_.resize(frame->camNum());
    for (int cam_id = 0; cam_id < frame->camNum(); ++cam_id) {
      frame->Tcw_[cam_id] =
          (pose * calibration_->getCamera(cam_id).getExtrinsic()).inverse();
    }
  }

  for (auto it = feature_map_.begin(); it != feature_map_.end(); ++it) {
    Feature::Ptr feature = it->second;
    if (feature->isTriangulated())
      continue;

    if (feature->observationSize() < 2)
      continue;

    std::vector<Eigen::Vector3d> bearings;
    std::vector<Eigen::Matrix4d> Tcws;

    for (auto obs : feature->observations()) {
      FrameIDType frame_id = obs.first;
      Frame::Ptr frame = frame_map_[frame_id];
      for (auto cam_obs : obs.second) {
        int cam_id = cam_obs.first;
        int pt_id = cam_obs.second;
        bearings.push_back(frame->bearings()[cam_id][pt_id].normalized());
        Tcws.push_back(frame->Tcw_[cam_id]);
      }
    }

    Eigen::Matrix4d Twc = Tcws[0].inverse();
    Eigen::Vector3d twc = Twc.block<3, 1>(0, 3);
    Eigen::Vector3d bearing0 = Twc.block<3, 3>(0, 0) * bearings[0];
    double fi = -twc(2) / bearing0(2);
    double x = twc(0) + fi * bearing0(0);
    double y = twc(1) + fi * bearing0(1);
    Eigen::Vector3d world_point(x, y, 0);

    feature->setWorldPoint(world_point);
    Frame::Ptr ref_frame = frame_map_[feature->refFrameId()];
    Eigen::Matrix4d Tcw = ref_frame->Tcw_[feature->refCamId()];
    Eigen::Vector3d camera_point =
        Tcw.block<3, 3>(0, 0) * feature->getWorldPoint() +
        Tcw.block<3, 1>(0, 3);

    if (camera_point.z() < 0) {
      feature->setInvDepth(1.0 / DEFAULT_DEPTH);
    } else {
      feature->setInvDepth(1.0 / camera_point.z());
    }
  }
}

std::vector<Eigen::Vector3d> SparseMap::getWorldPoints() {
  std::vector<Eigen::Vector3d> world_points;
  for (auto it = feature_map_.begin(); it != feature_map_.end(); ++it) {
    Feature::Ptr feature = it->second;
    if (!feature->isTriangulated())
      continue;

    world_points.push_back(feature->getWorldPoint());
  }
  return world_points;
}

void SparseMap::ransacWithF(const FrameIDType &left_frame_id,
                            const int &left_cam_id,
                            const FrameIDType &right_frame_id,
                            const int &right_cam_id,
                            std::vector<std::pair<int, int>> &good_matches) {
  if (good_matches.size() < 8) {
    good_matches.clear();
    return;
  }

  Frame::Ptr left_frame = frame_map_[left_frame_id];
  Frame::Ptr right_frame = frame_map_[right_frame_id];

  double FOCAL_LENGTH = 460.0;
  double CENTER_P = 500;

  std::vector<cv::Point2f> un_left_pts, un_right_pts;
  for (size_t i = 0; i < good_matches.size(); ++i) {
    int left_pt_id = good_matches[i].first;
    int right_pt_id = good_matches[i].second;

    Eigen::Vector3d tmp_p;

    tmp_p = left_frame->bearings()[left_cam_id][left_pt_id];
    tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + CENTER_P;
    tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + CENTER_P;
    un_left_pts.push_back(cv::Point2f(tmp_p.x(), tmp_p.y()));

    tmp_p = right_frame->bearings()[right_cam_id][right_pt_id];
    tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + CENTER_P;
    tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + CENTER_P;
    un_right_pts.push_back(cv::Point2f(tmp_p.x(), tmp_p.y()));
  }

  std::vector<uchar> status_F;
  cv::findFundamentalMat(un_left_pts, un_right_pts, cv::FM_RANSAC, 1.0, 0.99,
                         status_F);

  std::vector<uchar> status_H;
  cv::findHomography(un_left_pts, un_right_pts, cv::FM_RANSAC, 3, status_H);

  int inlier_num = 0;
  for (int i = 0; i < good_matches.size(); ++i)
    if (status_F[i] && status_H[i])
      good_matches[inlier_num++] = good_matches[i];

  good_matches.resize(inlier_num);
}

size_t SparseMap::getKeypointSize(const FrameIDType &f_id1, const int &c_id1) {
  return frame_map_[f_id1]->feature_ids()[c_id1].size();
}

bool SparseMap::bundleAdjustment(bool use_prior) {
  ceres::Problem::Options problem_options;
  ceres::Problem problem(problem_options);
  ceres::LocalParameterization *q_para =
      new ceres::EigenQuaternionParameterization();
  ceres::LossFunction *loss_function = new ceres::HuberLoss(1);

  std::cout << "pppp:" << frame_map_.size() << std::endl;

  for (auto it = frame_map_.begin(); it != frame_map_.end(); ++it) {
    Frame::Ptr frame = it->second;
    problem.AddParameterBlock(frame->getRotaionParams(), 4, q_para);
    problem.AddParameterBlock(frame->getTranslationParams(), 3);
  }

  std::cout << "pppp1" << std::endl;

  if (use_prior) {
    for (auto it = frame_map_.begin(); it != frame_map_.end(); ++it) {
      Frame::Ptr frame = it->second;
      ceres::CostFunction *cost_function =
          PoseError::Create(frame->Twb_prior_, 0, 1);
      problem.AddResidualBlock(cost_function, NULL,
                               frame->getTranslationParams(),
                               frame->getRotaionParams());
    }
  }
  // else {
  //   Frame::Ptr first_frame = frame_map_.begin()->second;
  //   problem.SetParameterBlockConstant(first_frame->pose_q);
  //   problem.SetParameterBlockConstant(first_frame->twb_.data());
  // }

  std::cout << "pppp2" << std::endl;

  for (auto it = frame_map_.begin(); it != frame_map_.end(); ++it) {
    Frame::Ptr frame = it->second;
    for (int cam_id = 0; cam_id < frame->camNum(); ++cam_id) {
      for (int pt_id = 0; pt_id < frame->feature_ids()[cam_id].size();
           ++pt_id) {
        FeatureIDType ft_id = frame->feature_ids()[cam_id][pt_id];
        if (ft_id < 0)
          continue;

        Feature::Ptr feature = feature_map_[ft_id];
        if (!feature->isTriangulated())
          continue;

        problem.AddParameterBlock(feature->getWorldPointParams(), 2);

        ceres::CostFunction *cost_function = PlaneReprojError::Create(
            frame->keypoints()[cam_id][pt_id],
            calibration_->getCamera(cam_id).getCameraModel());
        problem.AddResidualBlock(
            cost_function, loss_function, frame->getTranslationParams(),
            frame->getRotaionParams(), feature->getWorldPointParams());
      }
    }
  }

  std::cout << "pppp3" << std::endl;

  ceres::Solver::Options solver_options;
  solver_options.minimizer_progress_to_stdout = true;
  solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
  solver_options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
  solver_options.num_threads = 8;
  solver_options.max_num_iterations = 20;

  ceres::Solver::Summary summary;
  ceres::Solve(solver_options, &problem, &summary);

  return true;
}

std::vector<std::pair<size_t, size_t>>
SparseMap::getMatches(const FrameIDType &f_id1, const int &c_id1,
                      const FrameIDType &f_id2, const int &c_id2) {
  Frame::Ptr frame1 = frame_map_[f_id1];
  Frame::Ptr frame2 = frame_map_[f_id2];

  std::vector<std::pair<size_t, size_t>> matches;

  for (int pt_id1 = 0; pt_id1 < frame1->keypoints()[c_id1].size(); pt_id1++) {
    FeatureIDType ft_id1 = frame1->feature_ids()[c_id1][pt_id1];
    if (ft_id1 < 0)
      continue;

    Feature::Ptr feature1 = feature_map_[ft_id1];
    if (feature1->observations().find(f_id2) == feature1->observations().end())
      continue;

    const std::map<int, int> &obs2 = feature1->observations().at(f_id2);
    if (obs2.find(c_id2) == obs2.end())
      continue;

    int pt_id2 = obs2.at(c_id2);
    FeatureIDType ft_id2 = frame2->feature_ids()[c_id2][pt_id2];
    if (ft_id2 < 0) {
      std::cerr << "Error: ft_id2 < 0" << std::endl;
      continue;
    }

    assert(ft_id1 == ft_id2);

    matches.push_back(std::make_pair(pt_id1, pt_id2));
  }

  return matches;
}

std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>
SparseMap::getCorrespondences2D2D(const FrameIDType &f_id1, const int &c_id1,
                                  const FrameIDType &f_id2, const int &c_id2) {
  Frame::Ptr frame1 = frame_map_[f_id1];
  Frame::Ptr frame2 = frame_map_[f_id2];

  std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> corres;

  for (int pt_id1 = 0; pt_id1 < frame1->keypoints()[c_id1].size(); pt_id1++) {
    FeatureIDType ft_id1 = frame1->feature_ids()[c_id1][pt_id1];
    if (ft_id1 < 0)
      continue;

    Feature::Ptr feature1 = feature_map_[ft_id1];
    if (feature1->observations().find(f_id2) == feature1->observations().end())
      continue;

    const std::map<int, int> &obs2 = feature1->observations().at(f_id2);
    if (obs2.find(c_id2) == obs2.end())
      continue;

    int pt_id2 = obs2.at(c_id2);
    FeatureIDType ft_id2 = frame2->feature_ids()[c_id2][pt_id2];
    if (ft_id2 < 0) {
      std::cerr << "Error: ft_id2 < 0" << std::endl;
      continue;
    }

    assert(ft_id1 == ft_id2);

    corres.push_back(std::make_pair(frame1->bearings()[c_id1][pt_id1],
                                    frame2->bearings()[c_id2][pt_id2]));
  }

  return corres;
}

std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>
SparseMap::getCorrespondences2D3D(const FrameIDType &f_id1, const int &c_id1) {
  Frame::Ptr frame1 = frame_map_[f_id1];

  std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> corres;

  for (int pt_id1 = 0; pt_id1 < frame1->keypoints()[c_id1].size(); pt_id1++) {
    FeatureIDType ft_id1 = frame1->feature_ids()[c_id1][pt_id1];
    if (ft_id1 < 0)
      continue;

    Feature::Ptr feature1 = feature_map_[ft_id1];
    if (!feature1->isTriangulated())
      continue;

    corres.push_back(std::make_pair(frame1->bearings()[c_id1][pt_id1],
                                    feature1->getWorldPoint()));
  }

  return corres;
}

void SparseMap::printReprojError(const FrameIDType &f_id1, const int &c_id1) {
  Frame::Ptr frame1 = frame_map_[f_id1];
  const auto &camera = calibration_->getCamera(c_id1);
  Eigen::Matrix4d Tcw =
      (frame1->getBodyPose() * camera.getExtrinsic()).inverse();

  for (int pt_id1 = 0; pt_id1 < frame1->keypoints()[c_id1].size(); pt_id1++) {
    FeatureIDType ft_id1 = frame1->feature_ids()[c_id1][pt_id1];
    if (ft_id1 < 0)
      continue;

    Feature::Ptr feature1 = feature_map_[ft_id1];
    if (!feature1->isTriangulated())
      continue;

    Eigen::Vector3d pt_c = Tcw.block<3, 3>(0, 0) * feature1->getWorldPoint() +
                           Tcw.block<3, 1>(0, 3);
    Eigen::Vector2d pt_2d2;
    camera.getCameraModel().spaceToPlane(pt_c, &pt_2d2);
    Eigen::Vector2d pt_2d_obs = frame1->keypoints()[c_id1][pt_id1];
    Eigen::Vector2d error = pt_2d2 - pt_2d_obs;

    std::cout << "error: " << error.norm() << std::endl;
  }
}

cv::Mat SparseMap::drawKeypoint(FrameIDType frame_id, int cam_id) {
  if (frame_map_.find(frame_id) == frame_map_.end()) {
    std::cerr << "Error: frame_id not found" << std::endl;
    return cv::Mat();
  }

  cv::Mat result = frame_map_[frame_id]->drawKeyPoint(cam_id);
  cv::putText(result, std::to_string(frame_id) + "-" + std::to_string(cam_id),
              cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1,
              cv::Scalar(0, 255, 0), 2);
  return result;
}

cv::Mat SparseMap::drawMatchedKeypoint(FrameIDType frame_id, int cam_id) {
  if (frame_map_.find(frame_id) == frame_map_.end()) {
    std::cerr << "Error: frame_id not found" << std::endl;
    return cv::Mat();
  }

  cv::Mat result = frame_map_[frame_id]->drawMatchedKeyPoint(cam_id);
  cv::putText(result, std::to_string(frame_id) + "-" + std::to_string(cam_id),
              cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1,
              cv::Scalar(0, 255, 0), 2);
  return result;
}

cv::Mat SparseMap::drawFlow(FrameIDType frame_id, int cam_id,
                            FrameIDType last_frame_id) {
  FrameIDType pre_frame_id;
  if (last_frame_id < 0) {
    pre_frame_id = frame_id - 1;
  } else {
    pre_frame_id = last_frame_id;
  }

  if (frame_map_.find(pre_frame_id) == frame_map_.end()) {
    std::cerr << "Error: pre_frame_id not found" << std::endl;
    return cv::Mat();
  }

  if (frame_map_.find(frame_id) == frame_map_.end()) {
    std::cerr << "Error: frame_id not found" << std::endl;
    return cv::Mat();
  }

  Frame::Ptr frame = frame_map_[frame_id];
  cv::Mat result = frame->imgs()[cam_id].clone();

  if (result.channels() == 1) {
    cv::cvtColor(result, result, cv::COLOR_GRAY2BGR);
  }

  Frame::Ptr pre_frame = frame_map_[pre_frame_id];

  int num = 0;
  for (int pt_id = 0; pt_id < frame->feature_ids()[cam_id].size(); ++pt_id) {
    FeatureIDType ft_id = frame->feature_ids()[cam_id][pt_id];
    if (ft_id < 0)
      continue;

    Feature::Ptr feature = feature_map_[ft_id];
    int pt_id1 = feature->observation(pre_frame_id, cam_id);
    if (pt_id1 < 0)
      continue;

    FeatureIDType ft_id1 = pre_frame->feature_ids()[cam_id][pt_id1];
    if (ft_id1 < 0) {
      std::cerr << "Error: ft_id1 < 0" << std::endl;
      continue;
    }

    assert(ft_id == ft_id1);

    cv::Point pt0 = cv::Point(frame->keypoints()[cam_id][pt_id](0),
                              frame->keypoints()[cam_id][pt_id](1));
    cv::Point pt1 = cv::Point(pre_frame->keypoints()[cam_id][pt_id1](0),
                              pre_frame->keypoints()[cam_id][pt_id1](1));

    cv::circle(result, pt0, 2, cv::Scalar(255, 0, 0), 2);
    cv::line(result, pt0, pt1, cv::Scalar(255, 0, 0), 2);
    cv::putText(result, std::to_string(ft_id), pt0, cv::FONT_HERSHEY_SIMPLEX,
                0.3, cv::Scalar(0, 255, 0), 1);
    num++;
  }

  cv::putText(result, std::to_string(frame_id) + "-" + std::to_string(cam_id),
              cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1,
              cv::Scalar(0, 255, 0), 2);

  cv::putText(result, "num: " + std::to_string(num), cv::Point(10, 60),
              cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);

  return result;
}

cv::Mat SparseMap::drawMatches(FrameIDType frame_id0, int cam_id0,
                               FrameIDType frame_id1, int cam_id1) {
  if (frame_map_.find(frame_id0) == frame_map_.end()) {
    std::cerr << "Error: frame_id0 not found" << std::endl;
    return cv::Mat();
  }

  if (frame_map_.find(frame_id1) == frame_map_.end()) {
    std::cerr << "Error: frame_id1 not found" << std::endl;
    return cv::Mat();
  }

  Frame::Ptr frame0 = frame_map_[frame_id0];
  Frame::Ptr frame1 = frame_map_[frame_id1];
  cv::Mat img0 = frame0->imgs()[cam_id0].clone();
  cv::Mat img1 = frame1->imgs()[cam_id1].clone();
  cv::Mat merge_img = cv::Mat(std::max(img0.rows, img1.rows),
                              img0.cols + img1.cols, img0.type());
  img0.copyTo(merge_img(cv::Rect(0, 0, img0.cols, img0.rows)));
  img1.copyTo(merge_img(cv::Rect(img0.cols, 0, img1.cols, img1.rows)));
  cv::Point offset(img0.cols, 0);

  for (int pt_id0 = 0; pt_id0 < frame0->feature_ids()[cam_id0].size();
       pt_id0++) {
    FeatureIDType ft_id0 = frame0->feature_ids()[cam_id0][pt_id0];
    if (ft_id0 < 0)
      continue;

    Feature::Ptr feature0 = feature_map_[ft_id0];
    int pt_id1 = feature0->observation(frame_id1, cam_id1);
    if (pt_id1 < 0)
      continue;

    FeatureIDType ft_id1 = frame1->feature_ids()[cam_id1][pt_id1];
    if (ft_id1 < 0) {
      std::cerr << "Error: ft_id1 < 0" << std::endl;
      continue;
    }

    assert(ft_id0 == ft_id1);

    cv::Point pt0 = cv::Point(frame0->keypoints()[cam_id0][pt_id0](0),
                              frame0->keypoints()[cam_id0][pt_id0](1));
    cv::Point pt1 =
        cv::Point(frame1->keypoints()[cam_id1][pt_id1](0) + offset.x,
                  frame1->keypoints()[cam_id1][pt_id1](1));

    cv::circle(merge_img, pt0, 2, cv::Scalar(0, 255, 0), 2);
    cv::circle(merge_img, pt1, 2, cv::Scalar(0, 255, 0), 2);
    cv::line(merge_img, pt0, pt1, cv::Scalar(0, 255, 0), 2);
    cv::putText(merge_img, std::to_string(ft_id0), pt0,
                cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
    cv::putText(merge_img, std::to_string(ft_id0), pt1,
                cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
  }

  return merge_img;
}

cv::Mat SparseMap::drawStereoKeyPoint(FrameIDType frame_id) {
  if (frame_map_.find(frame_id) == frame_map_.end()) {
    std::cerr << "Error: frame_id not found" << std::endl;
    return cv::Mat();
  }

  Frame::Ptr frame = frame_map_[frame_id];
  std::vector<cv::Mat> imgs(frame->camNum());
  for (size_t cam_id = 0; cam_id < frame->camNum(); ++cam_id) {
    imgs[cam_id] = (frame->imgs()[cam_id].clone());
    if (imgs[cam_id].channels() == 1) {
      cv::cvtColor(imgs[cam_id], imgs[cam_id], cv::COLOR_GRAY2BGR);
    }

    cv::putText(imgs[cam_id],
                std::to_string(frame_id) + "-" + std::to_string(cam_id),
                cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1,
                cv::Scalar(0, 255, 0), 2);
  }

  for (size_t cam_id = 0; cam_id < frame->camNum(); ++cam_id) {
    for (size_t j = 0; j < frame->feature_ids()[cam_id].size(); ++j) {
      FeatureIDType ft_id = frame->feature_ids()[cam_id][j];
      if (ft_id < 0)
        continue;

      Feature::Ptr feature = feature_map_[ft_id];
      if (feature->observations().at(frame_id).size() < 2)
        continue;

      cv::circle(imgs[cam_id],
                 cv::Point(frame->keypoints()[cam_id][j](0),
                           frame->keypoints()[cam_id][j](1)),
                 2, cv::Scalar(0, 255, 0), 2);
      cv::putText(imgs[cam_id], std::to_string(ft_id),
                  cv::Point(frame->keypoints()[cam_id][j](0),
                            frame->keypoints()[cam_id][j](1)),
                  cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 255, 0), 1);
    }
  }

  cv::Mat merge_img;
  cv::hconcat(imgs, merge_img);
  return merge_img;
}

cv::Mat SparseMap::drawReprojKeyPoint(FrameIDType frame_id, int cam_id) {
  if (frame_map_.find(frame_id) == frame_map_.end()) {
    std::cerr << "Error: frame_id not found" << std::endl;
    return cv::Mat();
  }

  Frame::Ptr frame = frame_map_[frame_id];
  const auto &camera = calibration_->getCamera(cam_id);
  Eigen::Matrix4d Tcw =
      (frame->getBodyPose() * camera.getExtrinsic()).inverse();

  cv::Mat result = frame->imgs()[cam_id].clone();
  if (result.channels() == 1) {
    cv::cvtColor(result, result, cv::COLOR_GRAY2BGR);
  }

  int num = 0;
  for (int pt_id = 0; pt_id < frame->feature_ids()[cam_id].size(); ++pt_id) {
    FeatureIDType ft_id = frame->feature_ids()[cam_id][pt_id];
    if (ft_id < 0)
      continue;

    Feature::Ptr feature = feature_map_[ft_id];

    Eigen::Vector3d pt_c = Tcw.block<3, 3>(0, 0) * feature->getWorldPoint() +
                           Tcw.block<3, 1>(0, 3);
    Eigen::Vector2d pt_2d2;
    camera.getCameraModel().spaceToPlane(pt_c, &pt_2d2);

    cv::Point pt0 = cv::Point(pt_2d2(0), pt_2d2(1));

    cv::circle(result, pt0, 2, cv::Scalar(255, 0, 0), 2);
    cv::putText(result, std::to_string(ft_id), pt0, cv::FONT_HERSHEY_SIMPLEX,
                0.3, cv::Scalar(0, 255, 0), 1);
    num++;
  }

  cv::putText(result, std::to_string(frame_id) + "-" + std::to_string(cam_id),
              cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1,
              cv::Scalar(0, 255, 0), 2);

  cv::putText(result, "num: " + std::to_string(num), cv::Point(10, 60),
              cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);

  return result;
}
