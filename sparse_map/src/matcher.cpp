#include "sparse_map/matcher.h"

#include <opencv2/features2d.hpp>

void Matcher::matchORB(const cv::Mat &descriptors1, const cv::Mat &descriptors2,
                       std::vector<cv::DMatch> &matches) {
    double min_ratio = 0.8;
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);
    for (size_t i = 0; i < knn_matches.size(); i++) {
        if (knn_matches[i][0].distance < min_ratio * knn_matches[i][1].distance) {
            matches.push_back(knn_matches[i][0]);
        }
    }

}