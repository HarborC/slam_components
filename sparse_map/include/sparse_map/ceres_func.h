#pragma once

#include <ceres/ceres.h>

struct PinholeReprojError {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  PinholeReprojError(const Eigen::Vector2d &_pt, const double &_fx,
                     const double &_fy, const double &_cx, const double &_cy)
      : pt(_pt), fx(_fx), fy(_fy), cx(_cx), cy(_cy) {}

  template <typename T>
  bool operator()(const T *const twc, const T *const qwc, const T *const pw,
                  T *residuals) const {
    using QuatT = Eigen::Quaternion<T>;
    using Vector3T = Eigen::Matrix<T, 3, 1>;
    using Vector2T = Eigen::Matrix<T, 2, 1>;
    using Matrix3T = Eigen::Matrix<T, 3, 3>;

    Vector3T t_wc(twc[0], twc[1], twc[2]);
    QuatT q_wc(qwc[3], qwc[0], qwc[1], qwc[2]);

    Vector3T p_w(pw[0], pw[1], T(0));

    Vector3T p_c = q_wc.inverse() * (p_w - t_wc);

    Eigen::Map<Vector2T> residual(residuals);
    Vector2T pt_rep;

    pt_rep[0] = T(fx) * p_c[0] / p_c[2] + T(cx);
    pt_rep[1] = T(fy) * p_c[1] / p_c[2] + T(cy);

    residual = pt.template cast<T>() - pt_rep;

    return true;
  }

  static ceres::CostFunction *Create(const Eigen::Vector2d &_pt,
                                     const double &_fx, const double &_fy,
                                     const double &_cx, const double &_cy) {
    return (new ceres::AutoDiffCostFunction<PinholeReprojError, 2, 3, 4, 2>(
        new PinholeReprojError(_pt, _fx, _fy, _cx, _cy)));
  }

  Eigen::Vector2d pt;
  double fx, fy, cx, cy;
};

struct PoseError {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  PoseError(Eigen::Matrix4d pose, double rot_weight, double pos_weight)
      : rot_weight_(rot_weight), pos_weight_(pos_weight) {
    init_q_ = Eigen::Quaterniond(pose.block<3, 3>(0, 0));
    init_p_ = pose.block<3, 1>(0, 3);
  }

  template <typename T>
  bool operator()(const T *const twi, const T *const qwi, T *residuals) const {
    using QuatT = Eigen::Quaternion<T>;
    using Vector3T = Eigen::Matrix<T, 3, 1>;
    using Matrix3T = Eigen::Matrix<T, 3, 3>;
    using Vector6T = Eigen::Matrix<T, 6, 1>;
    QuatT q_wi(qwi[3], qwi[0], qwi[1], qwi[2]);
    Vector3T t_wi(twi[0], twi[1], twi[2]);

    Eigen::Map<Vector6T> residual(residuals);
    residual.template block<3, 1>(0, 0) =
        T(pos_weight_) * (t_wi - init_p_.template cast<T>());
    residual.template block<3, 1>(3, 0) =
        T(rot_weight_) * T(2) *
        (init_q_.template cast<T>().inverse() * q_wi).vec();
    return true;
  }

  static ceres::CostFunction *Create(Eigen::Matrix4d pose, double rot_weight,
                                     double pos_weight) {
    return (new ceres::AutoDiffCostFunction<PoseError, 6, 3, 4>(
        new PoseError(pose, rot_weight, pos_weight)));
  }

  Eigen::Quaterniond init_q_;
  Eigen::Vector3d init_p_;
  double rot_weight_, pos_weight_;
};
