#pragma once

#include "sparse_map/common.h"

class Frame {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Ptr = std::shared_ptr<Frame>;
  Ptr makeShared() { return std::make_shared<Frame>(*this); }

public:
  Frame() {}
  Frame(FrameIDType _id, int _cam_num);
  ~Frame() {}

  FrameIDType id();

  void setId(const FrameIDType &id) { id_ = id; }

  int camNum();

  const std::vector<cv::Mat> &imgs() { return imgs_; }

  const std::vector<std::vector<Eigen::Vector2d>> &keypoints() const {
    return keypoints_;
  }

  const std::vector<std::vector<Eigen::Vector3d>> &bearings() const {
    return bearings_;
  }

  const std::vector<cv::Mat> &descriptors() const { return descriptors_; }

  const std::vector<std::vector<FeatureIDType>> &featureIDs() {
    return feature_ids_;
  }

  void setFeatureID(const int &cam_id, const int &pt_id,
                    const FeatureIDType &ft_id);

  Eigen::Matrix4d getBodyPose();

  void setBodyPose(const Eigen::Matrix4d &Twb);

  void addData(const std::vector<cv::Mat> &_imgs = {},
               const std::vector<std::vector<Eigen::Vector2d>> &_keypoints = {},
               const std::vector<std::vector<Eigen::Vector3d>> &_bearings = {},
               const std::vector<cv::Mat> &_descriptors = {});

  void extractFeature(const std::vector<cv::Mat> &_imgs,
                      std::string detector_type = "ORB",
                      const std::vector<cv::Mat> &_masks = {});

  cv::Mat drawKeyPoint(const int &cam_id);

  cv::Mat drawMatchedKeyPoint(const int &cam_id);

  cv::Mat drawReprojKeyPoint(const int &cam_id);

  double *getRotaionParams();

  double *getTranslationParams();

public:
  Eigen::Matrix4d Twb_prior_;
  std::vector<Eigen::Matrix4d> Tcw_;
  std::vector<std::set<std::pair<FrameIDType, int>>> matched_frames_;

private:
  FrameIDType id_;
  std::vector<cv::Mat> imgs_;
  std::vector<std::vector<Eigen::Vector2d>> keypoints_;
  std::vector<std::vector<Eigen::Vector3d>> bearings_;
  std::vector<cv::Mat> descriptors_;
  std::vector<std::vector<FeatureIDType>> feature_ids_;

  // Pose
  double pose_q[4] = {0, 0, 0, 1}; // x, y, z, w (Rotation)
  double pose_t[3] = {0, 0, 0}; // x, y, z (Translation)

private:
  friend class cereal::access;

  template <class Archive> void save(Archive &ar) const {
    std::vector<double> pose_list;
    for (int i = 0; i < 4; ++i)
      pose_list.push_back(pose_q[i]);

    for (int i = 0; i < 3; ++i)
      pose_list.push_back(pose_t[i]);

    ar(cereal::make_nvp("id", id_), cereal::make_nvp("keypoints", keypoints_),
       cereal::make_nvp("bearings", bearings_),
       cereal::make_nvp("descriptors", descriptors_),
       cereal::make_nvp("feature_ids", feature_ids_),
       cereal::make_nvp("pose", pose_list),
       cereal::make_nvp("Twb_prior", Twb_prior_),
       cereal::make_nvp("imgs", imgs_),
       cereal::make_nvp("matched_frames", matched_frames_));
  }

  template <class Archive> void load(Archive &ar) {
    std::vector<double> pose_list;

    ar(cereal::make_nvp("id", id_), cereal::make_nvp("keypoints", keypoints_),
       cereal::make_nvp("bearings", bearings_),
       cereal::make_nvp("descriptors", descriptors_),
       cereal::make_nvp("feature_ids", feature_ids_),
       cereal::make_nvp("pose", pose_list),
       cereal::make_nvp("Twb_prior", Twb_prior_),
       cereal::make_nvp("imgs", imgs_),
       cereal::make_nvp("matched_frames", matched_frames_));

    for (int i = 0; i < 4; ++i)
      pose_q[i] = pose_list[i];

    for (int i = 0; i < 3; ++i)
      pose_t[i] = pose_list[i + 4];

    int cam_num = keypoints_.size();
    Tcw_.resize(cam_num);
  }
};
typedef std::map<FrameIDType, Frame::Ptr> FrameMap;