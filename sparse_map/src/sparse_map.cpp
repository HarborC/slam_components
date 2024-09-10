#include "sparse_map/sparse_map.h"
#include "sparse_map/ceres_func.h"

void SparseMap::addKeyFrame(
    const FrameIDType &id, const std::vector<cv::Mat> &imgs,
    const std::vector<std::vector<Eigen::Vector2d>> &keypoints,
    const std::vector<std::vector<Eigen::Vector3d>> &bearings,
    const std::vector<cv::Mat> &descriptors) {
  assert(imgs.size() == cam_num_);
  Frame::Ptr cur_frame(new Frame(id));
  cur_frame->addData(imgs, keypoints, bearings, descriptors);
  frame_map_[cur_frame->id_] = cur_frame;
  last_frame_ = cur_frame;
}

void SparseMap::addKeyFrame(const Frame::Ptr &frame) {
  if (!frame) {
    std::cerr << "Error: frame is nullptr" << std::endl;
    return;
  }

  if (frame->cam_num_ != cam_num_) {
    std::cerr << "Error: cam_num_ != cam_num" << std::endl;
    return;
  }

  frame_map_[frame->id_] = frame;
  last_frame_ = frame;
}

void SparseMap::updateFeatureMap() {
  for (auto it = feature_map_.begin(); it != feature_map_.end();) {
    if (it->second->frameSize() == 0) {
      it = feature_map_.erase(it);
    } else {
      it->second->update();
      it++;
    }
  }
}

void SparseMap::removeKeyFrame(const FrameIDType &id) {
  Frame::Ptr cur_frame = frame_map_[id];
  for (int cam_id = 0; cam_id < cur_frame->feature_ids_.size(); ++cam_id) {
    for (int pt_id = 0; pt_id < cur_frame->feature_ids_[cam_id].size();
         ++pt_id) {
      FeatureIDType ft_id = cur_frame->feature_ids_[cam_id][pt_id];
      if (ft_id < 0)
        continue;
      Feature::Ptr feature = feature_map_[ft_id];
      feature->observations_[id].erase(cam_id);
      if (feature->observations_[id].empty()) {
        feature->removeObservationByFrameId(id);
      }
    }
  }

  updateFeatureMap();

  frame_map_.erase(id);
}

void SparseMap::addInterMatches(
    const FrameIDType &cur_frame_id,
    const std::vector<Eigen::Vector2i> &stereo_ids,
    std::vector<std::vector<Eigen::Vector2i>> inter_matches) {
  assert(stereo_ids.size() == inter_matches.size());
  Frame::Ptr cur_frame = frame_map_[cur_frame_id];

  for (size_t pair_id = 0; pair_id < inter_matches.size(); ++pair_id) {
    int cam_id0 = stereo_ids[pair_id](0);
    int cam_id1 = stereo_ids[pair_id](1);
    for (size_t match_id = 0; match_id < inter_matches[pair_id].size();
         ++match_id) {
      int left_pt_id = inter_matches[pair_id][match_id](0);
      int right_pt_id = inter_matches[pair_id][match_id](1);
    }
  }

  // ransac
  if (use_ransac_) {
    for (size_t pair_id = 0; pair_id < stereo_ids.size(); ++pair_id) {
      int cam_id0 = stereo_ids[pair_id](0);
      int cam_id1 = stereo_ids[pair_id](1);
      ransacWithF(cur_frame_id, cam_id0, cur_frame_id, cam_id1,
                  inter_matches[pair_id]);
    }
  }

  // add observations (inter-frame)
  for (size_t pair_id = 0; pair_id < inter_matches.size(); ++pair_id) {
    int cam_id0 = stereo_ids[pair_id](0);
    int cam_id1 = stereo_ids[pair_id](1);
    for (size_t match_id = 0; match_id < inter_matches[pair_id].size();
         ++match_id) {
      int left_pt_id = inter_matches[pair_id][match_id](0);
      int right_pt_id = inter_matches[pair_id][match_id](1);

      Feature::Ptr left_feature, right_feature;
      FeatureIDType left_ft_id = cur_frame->feature_ids_[cam_id0][left_pt_id];
      FeatureIDType right_ft_id = cur_frame->feature_ids_[cam_id1][right_pt_id];

      if (left_ft_id < 0 && right_ft_id < 0) {
        Feature::Ptr new_feature(new Feature(feature_next_id++));
        feature_map_[new_feature->id_] = new_feature;
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

      cur_frame->feature_ids_[cam_id0][left_pt_id] = left_feature->id_;
      cur_frame->feature_ids_[cam_id1][right_pt_id] = right_feature->id_;
      left_feature->addObservation(cur_frame_id, cam_id0, left_pt_id);
      right_feature->addObservation(cur_frame_id, cam_id1, right_pt_id);
      if (left_feature->id_ != right_feature->id_) {
        left_feature->addObservation(cur_frame_id, cam_id1, right_pt_id);
        right_feature->addObservation(cur_frame_id, cam_id0, left_pt_id);
      }
    }
  }
}

void SparseMap::addIntraMatches(
    const FrameIDType &pre_frame_id, const FrameIDType &cur_frame_id,
    std::vector<std::vector<Eigen::Vector2i>> intra_matches) {
  Frame::Ptr cur_frame = frame_map_[cur_frame_id];
  Frame::Ptr pre_frame = frame_map_[pre_frame_id];

  std::cout << "intra_matches: " << intra_matches[0].size() << std::endl;

  // ransac
  if (use_ransac_) {
    for (size_t cam_id = 0; cam_id < intra_matches.size(); ++cam_id) {
      ransacWithF(pre_frame_id, cam_id, cur_frame_id, cam_id,
                  intra_matches[cam_id]);
    }
  }

  std::cout << "intra_matches: " << intra_matches[0].size() << std::endl;

  // add observations (intra-frame)
  for (size_t cam_id = 0; cam_id < intra_matches.size(); ++cam_id) {
    for (size_t match_id = 0; match_id < intra_matches[cam_id].size();
         ++match_id) {
      int pre_pt_id = intra_matches[cam_id][match_id](0);
      int cur_pt_id = intra_matches[cam_id][match_id](1);

      Feature::Ptr feature;

      FeatureIDType pre_ft_id = pre_frame->feature_ids_[cam_id][pre_pt_id];
      if (pre_ft_id < 0) {
        Feature::Ptr new_feature(new Feature(feature_next_id++));
        feature_map_[new_feature->id_] = new_feature;
        feature = new_feature;
      } else {
        feature = feature_map_[pre_ft_id];
      }

      pre_frame->feature_ids_[cam_id][pre_pt_id] = feature->id_;
      cur_frame->feature_ids_[cam_id][cur_pt_id] = feature->id_;
      feature->addObservation(pre_frame_id, cam_id, pre_pt_id);
      feature->addObservation(cur_frame_id, cam_id, cur_pt_id);
    }
  }
}

void SparseMap::updateKeyFramePose(const FrameIDType &id,
                                   const Eigen::Matrix4d &pose) {
  frame_map_[id]->setBodyPose(pose);
  for (int cam_id = 0; cam_id < frame_map_[id]->cam_num_; ++cam_id) {
    frame_map_[id]->Tcw_[cam_id] = (pose * calibrations_[cam_id]).inverse();
  }
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
  for (auto it = feature_map_.begin(); it != feature_map_.end(); ++it) {
    Feature::Ptr feature = it->second;
    if (feature->inv_depth_ > 0)
      continue;

    if (feature->observationSize() < 2)
      continue;

    std::vector<Eigen::Vector3d> bearings;
    std::vector<Eigen::Matrix4d> Tcws;

    for (auto obs : feature->observations_) {
      FrameIDType frame_id = obs.first;
      Frame::Ptr frame = frame_map_[frame_id];
      for (auto cam_obs : obs.second) {
        int cam_id = cam_obs.first;
        int pt_id = cam_obs.second;
        bearings.push_back(frame->bearings_[cam_id][pt_id].normalized());
        Tcws.push_back(frame->Tcw_[cam_id]);
      }
    }

    Eigen::Vector3d world_point;
    triangulatePoint(Tcws, bearings, &world_point);

    feature->world_point_ = world_point;
    Frame::Ptr ref_frame = frame_map_[feature->ref_frame_id_];
    int ref_cam_id = feature->ref_cam_id_;
    Eigen::Matrix4d Tcw = ref_frame->Tcw_[ref_cam_id];
    Eigen::Vector3d camera_point =
        Tcw.block<3, 3>(0, 0) * feature->world_point_ + Tcw.block<3, 1>(0, 3);

    if (camera_point.z() < 0) {
      feature->inv_depth_ = 1.0 / DEFAULT_DEPTH;
    } else {
      feature->inv_depth_ = 1.0 / camera_point.z();
    }
  }
}

std::vector<Eigen::Vector3d> SparseMap::getWorldPoints() {
  std::vector<Eigen::Vector3d> world_points;
  for (auto it = feature_map_.begin(); it != feature_map_.end(); ++it) {
    Feature::Ptr feature = it->second;
    if (feature->inv_depth_ < 0)
      continue;

    world_points.push_back(feature->world_point_);
  }
  return world_points;
}

void SparseMap::ransacWithF(const FrameIDType &left_frame_id,
                            const int &left_cam_id,
                            const FrameIDType &right_frame_id,
                            const int &right_cam_id,
                            std::vector<Eigen::Vector2i> &good_matches) {
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
    int left_pt_id = good_matches[i](0);
    int right_pt_id = good_matches[i](1);

    Eigen::Vector3d tmp_p;

    tmp_p = left_frame->bearings_[left_cam_id][left_pt_id];
    tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + CENTER_P;
    tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + CENTER_P;
    un_left_pts.push_back(cv::Point2f(tmp_p.x(), tmp_p.y()));

    tmp_p = right_frame->bearings_[right_cam_id][right_pt_id];
    tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + CENTER_P;
    tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + CENTER_P;
    un_right_pts.push_back(cv::Point2f(tmp_p.x(), tmp_p.y()));
  }

  std::vector<uchar> status;
  cv::findFundamentalMat(un_left_pts, un_right_pts, cv::FM_RANSAC, 1.0, 0.99,
                         status);
  int inlier_num = 0;
  for (int i = 0; i < status.size(); ++i)
    if (status[i])
      good_matches[inlier_num++] = good_matches[i];

  good_matches.resize(inlier_num);
}

cv::Mat SparseMap::drawKeypoint(FrameIDType frame_id, int cam_id) {
  cv::Mat result = frame_map_[frame_id]->drawKeyPoint(cam_id);
  cv::putText(result, std::to_string(frame_id) + "-" + std::to_string(cam_id),
              cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1,
              cv::Scalar(0, 255, 0), 2);
  return result;
}

cv::Mat SparseMap::drawMatchedKeypoint(FrameIDType frame_id, int cam_id) {
  cv::Mat result = frame_map_[frame_id]->drawMatchedKeyPoint(cam_id);
  cv::putText(result, std::to_string(frame_id) + "-" + std::to_string(cam_id),
              cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1,
              cv::Scalar(0, 255, 0), 2);
  return result;
}

cv::Mat SparseMap::drawFlow(FrameIDType frame_id, int cam_id, FrameIDType last_frame_id) {
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
  cv::Mat result = frame->imgs_[cam_id].clone();

  if (result.channels() == 1) {
    cv::cvtColor(result, result, cv::COLOR_GRAY2BGR);
  }

  
  
  Frame::Ptr pre_frame = frame_map_[pre_frame_id];

  int num = 0;
  for (int pt_id = 0; pt_id < frame->feature_ids_[cam_id].size(); ++pt_id) {
    FeatureIDType ft_id = frame->feature_ids_[cam_id][pt_id];
    if (ft_id < 0)
      continue;

    Feature::Ptr feature = feature_map_[ft_id];
    if (feature->observations_.find(pre_frame_id) ==
        feature->observations_.end())
      continue;

    std::map<int, int> &obs1 = feature->observations_[pre_frame_id];
    if (obs1.find(cam_id) == obs1.end())
      continue;

    int pt_id1 = obs1[cam_id];
    FeatureIDType ft_id1 = pre_frame->feature_ids_[cam_id][pt_id1];
    if (ft_id1 < 0) {
      std::cerr << "Error: ft_id1 < 0" << std::endl;
      continue;
    }

    assert(ft_id == ft_id1);

    cv::Point pt0 = cv::Point(frame->keypoints_[cam_id][pt_id](0),
                              frame->keypoints_[cam_id][pt_id](1));
    cv::Point pt1 = cv::Point(pre_frame->keypoints_[cam_id][pt_id1](0),
                              pre_frame->keypoints_[cam_id][pt_id1](1));

    cv::circle(result, pt0, 2, cv::Scalar(255, 0, 0), 2);
    cv::line(result, pt0, pt1, cv::Scalar(255, 0, 0), 2);
    cv::putText(result, std::to_string(ft_id), pt0, cv::FONT_HERSHEY_SIMPLEX,
                0.3, cv::Scalar(0, 255, 0), 1);
    num++;
  }

  cv::putText(result, std::to_string(frame_id) + "-" + std::to_string(cam_id),
              cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1,
              cv::Scalar(0, 255, 0), 2);

  // 右上角显示特征点数量
  cv::putText(result, "num: " + std::to_string(num),
              cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 1,
              cv::Scalar(0, 255, 0), 2);

  return result;
}

cv::Mat SparseMap::drawMatches(FrameIDType frame_id0, int cam_id0,
                               FrameIDType frame_id1, int cam_id1) {
  Frame::Ptr frame0 = frame_map_[frame_id0];
  Frame::Ptr frame1 = frame_map_[frame_id1];
  cv::Mat img0 = frame0->imgs_[cam_id0].clone();
  cv::Mat img1 = frame1->imgs_[cam_id1].clone();
  cv::Mat merge_img = cv::Mat(img0.rows, img0.cols + img1.cols, img0.type());
  img0.copyTo(merge_img(cv::Rect(0, 0, img0.cols, img0.rows)));
  img1.copyTo(merge_img(cv::Rect(img0.cols, 0, img1.cols, img1.rows)));
  cv::Point offset(img0.cols, 0);

  for (int pt_id0 = 0; pt_id0 < frame0->feature_ids_[cam_id0].size();
       pt_id0++) {
    FeatureIDType ft_id0 = frame0->feature_ids_[cam_id0][pt_id0];
    if (ft_id0 < 0)
      continue;

    Feature::Ptr feature0 = feature_map_[ft_id0];
    if (feature0->observations_.find(frame_id1) ==
        feature0->observations_.end())
      continue;

    std::map<int, int> &obs1 = feature0->observations_[frame_id1];
    if (obs1.find(cam_id1) == obs1.end())
      continue;

    int pt_id1 = obs1[cam_id1];
    FeatureIDType ft_id1 = frame1->feature_ids_[cam_id1][pt_id1];
    if (ft_id1 < 0) {
      std::cerr << "Error: ft_id1 < 0" << std::endl;
      continue;
    }

    assert(ft_id0 == ft_id1);

    cv::Point pt0 = cv::Point(frame0->keypoints_[cam_id0][pt_id0](0),
                              frame0->keypoints_[cam_id0][pt_id0](1));
    cv::Point pt1 =
        cv::Point(frame1->keypoints_[cam_id1][pt_id1](0)+ offset.x,
                  frame1->keypoints_[cam_id1][pt_id1](1));

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
  Frame::Ptr frame = frame_map_[frame_id];
  std::vector<cv::Mat> imgs(frame->cam_num_);
  for (size_t cam_id = 0; cam_id < frame->cam_num_; ++cam_id) {
    imgs[cam_id] = (frame->imgs_[cam_id].clone());
    if (imgs[cam_id].channels() == 1) {
      cv::cvtColor(imgs[cam_id], imgs[cam_id], cv::COLOR_GRAY2BGR);
    }

    cv::putText(imgs[cam_id],
                std::to_string(frame_id) + "-" + std::to_string(cam_id),
                cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1,
                cv::Scalar(0, 255, 0), 2);
  }

  for (size_t cam_id = 0; cam_id < frame->cam_num_; ++cam_id) {
    for (size_t j = 0; j < frame->feature_ids_[cam_id].size(); ++j) {
      FeatureIDType ft_id = frame->feature_ids_[cam_id][j];
      if (ft_id < 0)
        continue;

      Feature::Ptr feature = feature_map_[ft_id];
      if (feature->observations_[frame_id].size() < 2)
        continue;

      cv::circle(imgs[cam_id],
                 cv::Point(frame->keypoints_[cam_id][j](0),
                           frame->keypoints_[cam_id][j](1)),
                 2, cv::Scalar(0, 255, 0), 2);
      cv::putText(imgs[cam_id], std::to_string(ft_id),
                  cv::Point(frame->keypoints_[cam_id][j](0),
                            frame->keypoints_[cam_id][j](1)),
                  cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 255, 0), 1);
    }
  }

  cv::Mat merge_img;
  cv::hconcat(imgs, merge_img);
  return merge_img;
}

std::vector<std::pair<size_t, size_t>>
SparseMap::getMatches(const FrameIDType &f_id1, const int &c_id1,
                      const FrameIDType &f_id2, const int &c_id2) {
  Frame::Ptr frame1 = frame_map_[f_id1];
  Frame::Ptr frame2 = frame_map_[f_id2];

  std::vector<std::pair<size_t, size_t>> matches;

  for (int pt_id1 = 0; pt_id1 < frame1->keypoints_[c_id1].size(); pt_id1++) {
    FeatureIDType ft_id1 = frame1->feature_ids_[c_id1][pt_id1];
    if (ft_id1 < 0)
      continue;

    Feature::Ptr feature1 = feature_map_[ft_id1];
    if (feature1->observations_.find(f_id2) == feature1->observations_.end())
      continue;

    std::map<int, int> &obs2 = feature1->observations_[f_id2];
    if (obs2.find(c_id2) == obs2.end())
      continue;

    int pt_id2 = obs2[c_id2];
    FeatureIDType ft_id2 = frame2->feature_ids_[c_id2][pt_id2];
    if (ft_id2 < 0) {
      std::cerr << "Error: ft_id2 < 0" << std::endl;
      continue;
    }

    assert(ft_id1 == ft_id2);

    matches.push_back(std::make_pair(pt_id1, pt_id2));
  }

  return matches;
}

size_t SparseMap::getKeypointSize(const FrameIDType &f_id1, const int &c_id1) { 
  return frame_map_[f_id1]->feature_ids_[c_id1].size();
}

bool SparseMap::bundleAdjustment(const double& fx, const double& fy, const double& cx, const double& cy) {
  ceres::Problem::Options problem_options;
  ceres::Problem problem(problem_options);
  ceres::LocalParameterization *q_para = new ceres::EigenQuaternionParameterization();
  ceres::LossFunction* loss_function = new ceres::HuberLoss(1);

  std::cout << "pppp:" << frame_map_.size() << std::endl;

  for (auto it = frame_map_.begin(); it != frame_map_.end(); ++it) {
    Frame::Ptr frame = it->second;
    
    std::cout << "frame-id:" << frame->id_ << std::endl;
    problem.AddParameterBlock(frame->pose_q, 4, q_para);
    std::cout << "pose_q:" << frame->pose_q[0] << " " << frame->pose_q[1] << " " << frame->pose_q[2] << " " << frame->pose_q[3] << std::endl;
    problem.AddParameterBlock(frame->twb_.data(), 3);
    std::cout << "twb:" << frame->twb_[0] << " " << frame->twb_[1] << " " << frame->twb_[2] << std::endl;
  }

  std::cout << "pppp1" << std::endl;

  for (auto it = frame_map_.begin(); it != frame_map_.end(); ++it) {
    Frame::Ptr frame = it->second;
    ceres::CostFunction *cost_function = PoseError::Create(frame->Twb_prior_, 0, 1);
    problem.AddResidualBlock(cost_function, NULL, frame->twb_.data(), frame->pose_q);
  }

  std::cout << "pppp2" << std::endl;

  for (auto it = frame_map_.begin(); it != frame_map_.end(); ++it) {
    Frame::Ptr frame = it->second;
    for (int cam_id = 0; cam_id < frame->cam_num_; ++cam_id) {
      for (int pt_id = 0; pt_id < frame->feature_ids_[cam_id].size(); ++pt_id) {
        FeatureIDType ft_id = frame->feature_ids_[cam_id][pt_id];
        if (ft_id < 0)
          continue;

        Feature::Ptr feature = feature_map_[ft_id];
        if (feature->inv_depth_ < 0)
          continue;

        problem.AddParameterBlock(feature->world_point_.data(), 3);

        ceres::CostFunction *cost_function =
            PinholeReprojError::Create(frame->keypoints_[cam_id][pt_id], fx, fy, cx, cy);
        problem.AddResidualBlock(cost_function, loss_function, frame->twb_.data(), 
                                 frame->pose_q, feature->world_point_.data());
      }
    }
  }

  std::cout << "pppp3" << std::endl;
  
  ceres::Solver::Options solver_options;
  solver_options.minimizer_progress_to_stdout = false;
  solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
  solver_options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
  solver_options.num_threads = 8;
  solver_options.max_num_iterations = 20;

  ceres::Solver::Summary summary;
  ceres::Solve(solver_options, &problem, &summary);

  std::cout << "pppp4" << std::endl;

  for (auto it = frame_map_.begin(); it != frame_map_.end(); ++it) {
    Frame::Ptr frame = it->second;
    
    Eigen::Matrix4d Twb = Eigen::Matrix4d::Identity();
    Twb.block<3, 3>(0, 0) = Eigen::Quaterniond(frame->pose_q[3], frame->pose_q[0], frame->pose_q[1], frame->pose_q[2]).toRotationMatrix();
    Twb.block<3, 1>(0, 3) = frame->twb_;
    frame->setBodyPose(Twb);
  }

  std::cout << "pppp5" << std::endl;

  return true;
}