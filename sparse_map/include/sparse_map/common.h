#pragma once

#include <atomic>
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#define DEFAULT_DEPTH 1.0

typedef long long int FeatureIDType;
typedef long long int FrameIDType;

// id for next feature
extern std::atomic<FeatureIDType> feature_next_id;

// id for next frame
extern std::atomic<FrameIDType> frame_next_id;