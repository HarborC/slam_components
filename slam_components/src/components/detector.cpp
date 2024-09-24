#include "components/detector.h"

#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
// #include <opencv2/xfeatures2d.hpp>

namespace slam_components {

void Detector::detectORB(const cv::Mat &img,
                         std::vector<cv::KeyPoint> &keypoints,
                         cv::Mat &descriptors, const cv::Mat &mask) {
  cv::Mat img_gray;
  if (img.channels() == 3) {
    cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
  } else {
    img_gray = img.clone();
  }
  cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
  cv::Mat clahe_img;
  clahe->apply(img_gray, clahe_img);
  cv::Ptr<cv::ORB> orb =
      cv::ORB::create(2000, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20);
  orb->detectAndCompute(clahe_img, mask, keypoints, descriptors);
  // std::cout << "ORB keypoints: " << keypoints.size() << std::endl;
}

void Detector::detectSIFT(const cv::Mat &img,
                          std::vector<cv::KeyPoint> &keypoints,
                          cv::Mat &descriptors, const cv::Mat &mask) {
  return;
}

} // namespace slam_components