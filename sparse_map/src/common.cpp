#include "sparse_map/common.h"

std::atomic<FeatureIDType> feature_next_id = 0; // id for next feature
std::atomic<FrameIDType> frame_next_id = 0; // id for next frame