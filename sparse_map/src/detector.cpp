#include "sparse_map/detector.h"

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
// #include <opencv2/xfeatures2d.hpp>

void Detector::detectORB(const cv::Mat &img, std::vector<cv::KeyPoint> &keypoints,
                          cv::Mat &descriptors, const cv::Mat &mask) {
  cv::Ptr<cv::ORB> orb = cv::ORB::create();
  orb->detectAndCompute(img, mask, keypoints, descriptors);
}