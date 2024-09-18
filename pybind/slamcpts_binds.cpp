#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "calibration/calibration.h"
#include "ndarray_converter.h"
#include "sparse_map/sparse_map.h"

namespace py = pybind11;

PYBIND11_MODULE(pyslamcpts, m) {
  NDArrayConverter::init_numpy();

  using namespace pybind11::literals;
  m.doc() = "python interface for pyslamcpts";

  py::class_<Sensor, Sensor::Ptr>(m, "Sensor")
      .def(py::init<>())
      .def("type", &Sensor::type, "get sensor type")
      .def("get_extrinsic", &Sensor::getExtrinsic, "get extrinsic")
      .def("set_extrinsic", &Sensor::setExtrinsic, "set extrinsic",
           py::arg("_extrinsic"));

  py::class_<Camera, Sensor, Camera::Ptr>(m, "Camera")
      .def(py::init<>())
      .def("set_camera_model", &Camera::setCameraModel, "set camera model",
           py::arg("_camera_model"))
      .def("get_camera_model", &Camera::getCameraModel, "get camera model");

  py::class_<Calibration, Calibration::Ptr>(m, "Calibration")
      .def(py::init<>())
      .def("cam_num", &Calibration::camNum, "get camera number")
      .def("add_camera", &Calibration::addCamera,
           "add camera to the calibration", py::arg("camera"))
      .def("get_camera", &Calibration::getCamera,
           "get camera from the calibration", py::arg("cam_id"));

  py::class_<Frame, Frame::Ptr>(m, "Frame")
      .def(py::init<>())
      .def(py::init<long long int, int>())
      .def("id", &Frame::id, "get frame id")
      .def("cam_num", &Frame::camNum, "get camera number")
      .def("imgs", &Frame::imgs, "get images")
      .def("keypoints", &Frame::keypoints, "get keypoints")
      .def("bearings", &Frame::bearings, "get bearings")
      .def("feature_ids", &Frame::featureIDs, "get feature ids")
      .def("descriptors", &Frame::descriptors, "get descriptors")
      .def("set_feature_id", &Frame::setFeatureID, "set feature id",
           py::arg("cam_id"), py::arg("pt_id"), py::arg("ft_id"))
      .def("get_body_pose", &Frame::getBodyPose, "get body pose")
      .def("set_body_pose", &Frame::setBodyPose, "set body pose",
           py::arg("Twb"))
      .def("add_data", &Frame::addData, "add data to the frame",
           py::arg("_imgs") = std::vector<cv::Mat>(),
           py::arg("_keypoints") = std::vector<std::vector<Eigen::Vector2d>>(),
           py::arg("_bearings") = std::vector<std::vector<Eigen::Vector3d>>(),
           py::arg("_descriptors") = std::vector<cv::Mat>())
      .def("extract_feature", &Frame::extractFeature, "extract feature",
           py::arg("_imgs"), py::arg("detector_type") = "ORB",
           py::arg("_masks") = std::vector<cv::Mat>())
      .def("draw_keypoint", &Frame::drawKeyPoint, "draw keypoint",
           py::arg("cam_id"))
      .def("draw_matched_keypoint", &Frame::drawMatchedKeyPoint,
           "draw matched keypoint", py::arg("cam_id"))
      .def("draw_reproj_keypoint", &Frame::drawReprojKeyPoint,
           "draw reprojected keypoint", py::arg("cam_id"));

  py::class_<Feature, Feature::Ptr>(m, "Feature")
      .def(py::init<>())
      .def(py::init<const long long int &>(), py::arg("id"))
      .def("id", &Feature::id, "get feature id")
      .def("ref_frame_id", &Feature::refFrameId, "get reference frame id")
      .def("ref_cam_id", &Feature::refCamId, "get reference camera id")
      .def("observations", &Feature::observations, "get observations")
      .def("observation", &Feature::observation, "get observation",
           py::arg("frame_id"), py::arg("cam_id"))
      .def("get_world_point", &Feature::getWorldPoint, "get world point")
      .def("observations", &Feature::observations, "get observations")
      .def("add_observation", &Feature::addObservation, "add observation",
           py::arg("frame_id"), py::arg("cam_id"), py::arg("pt_id"))
      .def("has_observation", &Feature::hasObservation,
           "check if feature has observation", py::arg("frame_id"),
           py::arg("cam_id"))
      .def("remove_observation", &Feature::removeObservation,
           "remove observation", py::arg("frame_id"), py::arg("cam_id"))
      .def("remove_observation_by_frame_id",
           &Feature::removeObservationByFrameId,
           "remove observations by frame id", py::arg("frame_id"))
      .def("is_valid", &Feature::isValid, "check if feature is valid")
      .def("covis_frame_size", &Feature::coVisFrameSize, "get covis frame size")
      .def("observation_size", &Feature::observationSize,
           "get observation size")
      .def("ref_info_update", &Feature::refInfoUpdate, "update reference info")
      .def("is_triangulated", &Feature::isTriangulated,
           "check if feature is triangulated")
      .def("get_inv_depth", &Feature::getInvDepth, "get inverse depth")
      .def("set_inv_depth", &Feature::setInvDepth, "set inverse depth",
           py::arg("inv_depth"))
      .def("get_world_point", &Feature::getWorldPoint, "get world point")
      .def("set_world_point", &Feature::setWorldPoint, "set world point",
           py::arg("world_point"));

  py::class_<SparseMap, SparseMap::Ptr>(m, "SparseMap")
      .def(py::init<>())
      .def(py::init<const Calibration::Ptr &, bool>(), py::arg("calibration"),
           py::arg("use_ransac") = true)
      .def("save", &SparseMap::save, "save the sparse map", py::arg("path"))
      .def("load", &SparseMap::load, "load the sparse map", py::arg("path"))
      .def("clear", &SparseMap::clear, "clear the sparse map")
      .def("add_keyframe",
           static_cast<void (SparseMap::*)(
               const FrameIDType &, const std::vector<cv::Mat> &,
               const std::vector<std::vector<Eigen::Vector2d>> &,
               const std::vector<cv::Mat> &)>(&SparseMap::addKeyFrame),
           "add key frame to the sparse map", py::arg("id"), py::arg("imgs"),
           py::arg("keypoints"),
           py::arg("descriptors") = std::vector<cv::Mat>())
      .def("add_keyframe",
           static_cast<void (SparseMap::*)(const Frame::Ptr &)>(
               &SparseMap::addKeyFrame),
           "add key frame to the sparse map", py::arg("frame"))
      .def("remove_invalid_feature", &SparseMap::removeInvalidFeature,
           "remove invalid feature from the sparse map")
      .def("remove_keyframe", &SparseMap::removeKeyFrame,
           "remove key frame from the sparse map", py::arg("id"))
      .def("add_intra_matches", &SparseMap::addIntraMatches,
           "add intra matches to the sparse map", py::arg("pre_frame_id"),
           py::arg("cur_frame_id"), py::arg("intra_matches"))
      .def("add_inter_matches", &SparseMap::addInterMatches,
           "add inter matches to the sparse map", py::arg("cur_frame_id"),
           py::arg("stereo_ids"), py::arg("inter_matches"))
      .def("update_keyframe_pose", &SparseMap::updateKeyFramePose,
           "update key frame pose in the sparse map", py::arg("id"),
           py::arg("pose"))
      .def("triangulate", &SparseMap::triangulate, "triangulate the sparse map")
      .def("draw_keypoint", &SparseMap::drawKeyPoint,
           "draw keypoint of the frame", py::arg("frame_id"), py::arg("cam_id"))
      .def("draw_matches", &SparseMap::drawMatches,
           "draw matches between two frames", py::arg("frame_id0"),
           py::arg("cam_id0"), py::arg("frame_id1"), py::arg("cam_id1"))
      .def("draw_flow", &SparseMap::drawFlow, "draw flow between two frames",
           py::arg("frame_id"), py::arg("cam_id"),
           py::arg("last_frame_id") = -1)
      .def("draw_stereo_keypoint", &SparseMap::drawStereoKeyPoint,
           "draw stereo keypoint of the frame", py::arg("frame_id"))
      .def("draw_reproj_keypoint", &SparseMap::drawReprojKeyPoint,
           "draw reprojected keypoint of the frame", py::arg("frame_id"),
           py::arg("cam_id"))
      .def("get_matches", &SparseMap::getMatches,
           "get matches between two frames", py::arg("f_id1"), py::arg("c_id1"),
           py::arg("f_id2"), py::arg("c_id2"))
      .def("get_correspondences_2d2d", &SparseMap::getCorrespondences2D2D,
           "get correspondences between two frames", py::arg("f_id1"),
           py::arg("c_id1"), py::arg("f_id2"), py::arg("c_id2"))
      .def("get_correspondences_2d3d", &SparseMap::getCorrespondences2D3D,
           "get correspondences between 2d and 3d", py::arg("f_id1"),
           py::arg("c_id1"))
      .def("get_keypoint_size", &SparseMap::getKeypointSize,
           "get keypoint size of the frame", py::arg("f_id1"), py::arg("c_id1"))
      .def("get_frame", &SparseMap::getFrame, "get frame from the sparse map",
           py::arg("id"))
      .def("get_world_points", &SparseMap::getWorldPoints,
           "get world points of the sparse map");
}