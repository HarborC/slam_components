#include <opencv2/opencv.hpp>
#include <torch/torch.h>

#include <Eigen/Core>

namespace slam_components {

template <typename V>
using MatrixXrm =
    typename Eigen::Matrix<V, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template <typename V>
using MatrixX = typename Eigen::Matrix<V, Eigen::Dynamic, Eigen::Dynamic>;

template <typename V> inline torch::Tensor eigen2libtorch(MatrixX<V> &M) {
  Eigen::Matrix<V, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> E(M);
  std::vector<int64_t> dims = {E.rows(), E.cols()};
  auto T = torch::from_blob(E.data(), dims).clone(); //.to(torch::kCPU);
  return T;
}

template <typename V>
inline torch::Tensor eigen2libtorch(MatrixXrm<V> &E, bool copydata = true) {
  std::vector<int64_t> dims = {E.rows(), E.cols()};
  auto T = torch::from_blob(E.data(), dims);
  if (copydata)
    return T.clone();
  else
    return T;
}

template <typename V> inline MatrixX<V> cv2eigen(cv::Mat &C) {
  Eigen::Map<MatrixXrm<V>> E(C.ptr<V>(), C.rows, C.cols);
  return E;
}

inline torch::Tensor cv2libtorch(cv::Mat &C, bool copydata = true,
                                 bool is_cv_image = false) {
  int kCHANNELS = C.channels();
  if (is_cv_image) {
    if (kCHANNELS == 3) {
      cv::cvtColor(C, C, cv::COLOR_BGR2RGB);
      C.convertTo(C, CV_32FC3, 1.0f / 255.0f);
    } else // considering channels = 1
      C.convertTo(C, CV_32FC1, 1.0f / 255.0f);
    auto T = torch::from_blob(C.data, {1, C.rows, C.cols, kCHANNELS});
    T = T.permute({0, 3, 1, 2});
    return T;
  } else {
    std::vector<int64_t> dims;
    at::TensorOptions options(at::kFloat);
    if (kCHANNELS == 1) {
      C.convertTo(C, CV_32FC1);
      dims = {C.rows, C.cols};
    } else { // considering channels = 3
      C.convertTo(C, CV_32FC3);
      dims = {C.rows, C.cols, kCHANNELS};
    }
    // auto T = torch::from_blob(C.ptr<float>(), dims, options).clone();
    // auto T = torch::from_blob(C.data, at::IntList(dims), options);
    auto T = torch::from_blob(C.data, dims, options);
    if (copydata)
      return T.clone();
    else
      return T;
  }
}

template <typename V>
inline Eigen::Matrix<V, Eigen::Dynamic, Eigen::Dynamic>
libtorch2eigen(torch::Tensor &Tin) {
  /*
   LibTorch is Row-major order and Eigen is Column-major order.
   MatrixXrm uses Eigen::RowMajor for compatibility.
   */
  auto T = Tin.to(torch::kCPU);
  Eigen::Map<MatrixXrm<V>> E(T.data_ptr<V>(), T.size(0), T.size(1));
  return E;
}

// Consider torch::Tensor as a float matrix
inline cv::Mat libtorch2cv(torch::Tensor &Tin, bool copy = true) {
  auto T = Tin.to(torch::kCPU);
  cv::Mat C;
  if (copy) {
    cv::Mat M(T.size(0), T.size(1), CV_32FC1, T.data<float>());
    M.copyTo(C);
  } else
    C = cv::Mat(T.size(0), T.size(1), CV_32FC1, T.data<float>());
  return C;
}

} // namespace slam_components