#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "sparse_map/sparse_map.h"
#include "ndarray_converter.h"

namespace py = pybind11;

PYBIND11_MODULE(pyslamcpts, m) {
  NDArrayConverter::init_numpy();
  
  using namespace pybind11::literals;
  m.doc() = "python interface for pyslamcpts";

  py::class_<Frame>(m, "Frame").def(py::init<long long int>());

  py::class_<SparseMap>(m, "SparseMap")
      .def(py::init<bool>(), py::arg("use_ransac") = true)
      .def("clear", &SparseMap::clear, "clear the sparse map")
      .def("set_calibration", &SparseMap::setCalibration,
           "set calibration of the sparse map")
      .def("add_keyframe", &SparseMap::addKeyFrame,
           "add key frame to the sparse map")
      .def("remove_keyframe", &SparseMap::removeKeyFrame,
           "remove key frame from the sparse map")
      .def("add_intra_matches", &SparseMap::addIntraMatches,
           "add intra matches to the sparse map")
      .def("add_inter_matches", &SparseMap::addInterMatches,
           "add inter matches to the sparse map")
      .def("update_keyframe_pose", &SparseMap::updateKeyFramePose,
           "update key frame pose in the sparse map")
      .def("triangulate", &SparseMap::triangulate,
           "triangulate the sparse map")
      .def("draw_keypoint", &SparseMap::drawKeypoint,
           "draw keypoint of the frame")
      .def("draw_matches", &SparseMap::drawMatches,
           "draw matches between two frames")
      .def("draw_flow", &SparseMap::drawFlow,
           "draw flow between two frames")
      .def("draw_stereo_keypoint", &SparseMap::drawStereoKeyPoint,
           "draw stereo keypoint of the frame")
      .def("get_world_points", &SparseMap::getWorldPoints,
           "get world points of the sparse map");
          
}