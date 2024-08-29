#include <string>
#include <iostream>
#include <vector>
#include <Eigen/Core>

#include "utils/io_utils.h"
#include "utils/log_utils.h"
#include "sparse_map/sparse_map.h"
#include "sparse_map/matcher.h"

#include "foxglove/visualizer.h"

using namespace Eigen;

// Step 1: read images sequence from the directory
// Step 2: extract features from the images
// Step 3: match features between images

using namespace foxglove_viz;

#include <iostream>
#include <Eigen/Dense>

Eigen::Vector3d llhToECEF(double lat, double lon, double alt) {
    // WGS84楠球参数
    const double a = 6378137.0;
    const double b = 6356752.3142;
    const double f = (a - b) / a;
    double e2 = f * (2 - f);

    double N = a / std::sqrt(1 - e2 * std::sin(lat) * std::sin(lat));

    Eigen::Vector3d xyz;
    xyz(0) = (N + alt) * std::cos(lat) * std::cos(lon);
    xyz(1) = (N + alt) * std::cos(lat) * std::sin(lon);
    xyz(2) = (N * (1 - e2) + alt) * std::sin(lat);

    return xyz;
}

// llh to NUE(北天东)坐标系
Eigen::Vector3d llhToNUE(double lat, double lon, double hei, double lat0, double lon0, double hei0 = 0) {

    Eigen::Vector3d xyz0 = llhToECEF(lat0, lon0, hei0);

    Eigen::Vector3d xyz = llhToECEF(lat, lon, hei);

    Eigen::Vector3d dxyz = xyz - xyz0;

    Eigen::Matrix3d R;
    R << -std::sin(lat0) * std::cos(lon0), -std::sin(lat0) * std::sin(lon0), std::cos(lat0),
         std::cos(lat0) * std::cos(lon0), std::cos(lat0) * std::sin(lon0), std::sin(lat0),
         -std::sin(lon0), std::cos(lon0), 0;

    Eigen::Vector3d nue = R * dxyz;

    return nue;
}

Eigen::Matrix3d nueToBody(double yaw, double pitch, double roll) {
    Eigen::Matrix3d R;

    double cos_yaw = std::cos(yaw);
    double sin_yaw = std::sin(yaw);
    double cos_pitch = std::cos(pitch);
    double sin_pitch = std::sin(pitch);
    double cos_roll = std::cos(roll);
    double sin_roll = std::sin(roll);

    R(0, 0) = cos_pitch * cos_yaw;
    R(0, 1) = sin_pitch;
    R(0, 2) = -cos_pitch * sin_yaw;

    R(1, 0) = sin_yaw * sin_roll - cos_yaw * sin_pitch * cos_roll;
    R(1, 1) = cos_pitch * cos_roll;
    R(1, 2) = sin_yaw * cos_roll + sin_yaw * sin_roll * cos_yaw;

    R(2, 0) = sin_yaw * cos_roll + cos_yaw * sin_pitch * sin_roll;
    R(2, 1) = -cos_pitch * sin_roll;
    R(2, 2) = cos_yaw * cos_roll - sin_yaw * sin_pitch * sin_roll;

    return R;
}

Eigen::Matrix3d bodyToCamera(double qy, double qz) {
    Eigen::Matrix3d R;

    double cos_qy = std::cos(qy);
    double sin_qy = std::sin(qy);
    double cos_qz = std::cos(qz);
    double sin_qz = std::sin(qz);

    R(0, 0) = cos_qy * cos_qz;
    R(0, 1) = sin_qy;
    R(0, 2) = -cos_qy * sin_qz;

    R(1, 0) = -sin_qy * cos_qz;
    R(1, 1) = cos_qy;
    R(1, 2) = sin_qy * sin_qz;

    R(2, 0) = sin_qz;
    R(2, 1) = 0;
    R(2, 2) = cos_qz;

    return R;
}

std::map<std::string, std::vector<double>> readCSV(const std::string& filename) {
    std::map<std::string, std::vector<double>> ins_data;

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open the file " << filename << std::endl;
        return ins_data;
    }

    std::string line;
    std::getline(file, line);
    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }

        std::vector<std::string> items = Utils::StringSplit(line, ",");
        std::string name = items[0];
        std::vector<double> data;
        for (int i = 1; i < items.size() - 1; i++) {
            data.push_back(std::stod(items[i]));
        }
        ins_data[name] = data;
    }

    file.close();

    // print
    for (auto it = ins_data.begin(); it != ins_data.end(); it++) {
        std::cout << it->first << ": ";
        for (int i = 0; i < it->second.size(); i++) {
            std::cout << it->second[i] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "ins_data size: " << ins_data.size() << std::endl;

    return ins_data;
}

double distance(const cv::Point2f& p1, const cv::Point2f& p2) {
    return std::sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
}

cv::Mat getVaildMask(const std::vector<cv::Point2f> &pts, const cv::Mat &mask) {
  int min_window_size = 5;

  cv::Mat mask_updated = mask.clone();

  int width = mask_updated.cols;
  int height = mask_updated.rows;

  for (size_t i = 0; i < pts.size(); i++) {
    // Get current left keypoint, check that it is in bounds
    const cv::Point2f &pt = pts[i];
    int x = (int)pt.x;
    int y = (int)pt.y;
    if (x < min_window_size || x >= width - min_window_size ||
        y < min_window_size || y >= height - min_window_size) {
      continue;
    }

    // Now check if it is in a mask area or not
    // NOTE: mask has max value of 255 (white) if it should be
    if (mask.at<uint8_t>(y, x) < 127) {
      continue;
    }

    // Append this to the local mask of the image
    cv::Point pt1(x - min_window_size, y - min_window_size);
    cv::Point pt2(x + min_window_size, y + min_window_size);
    cv::rectangle(mask_updated, pt1, pt2, cv::Scalar(0));
  }

  return mask_updated.clone();
}

std::vector<Eigen::Vector2i> opticalFlow(const cv::Mat &prev_img, const cv::Mat& curr_img, const std::vector<cv::Point2f> &prev_pts, std::vector<cv::Point2f> &curr_pts, bool double_check = false) {
    int need_pts_num = 500;
    std::vector<Eigen::Vector2i> matches;
    
    if (prev_pts.size()) {
        std::vector<uchar> status;
        std::vector<float> err;
        cv::calcOpticalFlowPyrLK(prev_img, curr_img, prev_pts, curr_pts, status, err, cv::Size(21, 21), 3);

        if (double_check) {
            std::vector<uchar> reverse_status;
            std::vector<cv::Point2f> reverse_pts = prev_pts;
            cv::calcOpticalFlowPyrLK(curr_img, prev_img, curr_pts, reverse_pts, reverse_status, err,
                cv::Size(21, 21), 1, cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
                cv::OPTFLOW_USE_INITIAL_FLOW);

            for (size_t i = 0; i < status.size(); i++) {
                if (status[i] && reverse_status[i] && distance(prev_pts[i], reverse_pts[i]) <= 0.5f)
                    status[i] = 1;
                else
                    status[i] = 0;
            }
        }
        
        std::vector<cv::Point2f> curr_pts2;
        for (int i = 0; i < status.size(); i++) {
            if (status[i]) {
                curr_pts2.push_back(curr_pts[i]);
                matches.push_back(Eigen::Vector2i(i, curr_pts2.size() - 1));
            }
        }
        curr_pts = curr_pts2;
    }

    cv::Mat mask = getVaildMask(curr_pts, cv::Mat::ones(curr_img.size(), CV_8UC1) * 255);
    
    if (curr_pts.size() + 20 < need_pts_num) {
        std::vector<cv::Point2f> add_pts;
        cv::goodFeaturesToTrack(curr_img, add_pts, need_pts_num - curr_pts.size(), 0.01, 10, mask);
        for (int i = 0; i < add_pts.size(); i++) {
            curr_pts.push_back(add_pts[i]);
        }
    }

    return matches;
}

std::vector<Eigen::Vector2d> toEigen(const std::vector<cv::Point2f> &pts) {
    std::vector<Eigen::Vector2d> eigen_pts(pts.size());
    for (int i = 0; i < pts.size(); i++) {
        eigen_pts[i] << pts[i].x, pts[i].y;
    }
    return eigen_pts;
}

cv::Mat preprocessImage(const cv::Mat &img) {
    cv::Mat img_gray;
    if (img.channels() == 3) {
        cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
    } else {
        img_gray = img.clone();
    }
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    cv::Mat clahe_img;
    clahe->apply(img_gray, clahe_img);

    return clahe_img;
}

SparseMap::Ptr flow_sparse_map;

int main (int argc, char** argv) {
    Visualizer server(8088);

    std::string img_dir = "../../../datasets/TXPJ/test1/sub";
    std::string img_format = "jpg";

    auto ins_data = readCSV("../../../datasets/TXPJ/test1/ins1.csv");

    int scale = 4;

    double f = 33400 / scale;
    double cx = 4096 / 2 / scale;
    double cy = 3072 / 2 / scale;

    std::vector<std::string> img_files = Utils::GetFileList(img_dir);
    std::sort(img_files.begin(), img_files.end());

    std::vector<Eigen::Matrix4f> poses;

    flow_sparse_map.reset(new SparseMap(true));
    SparseMap::Ptr sparse_map(new SparseMap(true));

    double lat0, lon0, alt0;
    Frame::Ptr prev_flow_frame;
    cv::Mat prev_flow_img, curr_flow_img;
    std::vector<cv::Point2f> prev_flow_pts, curr_flow_pts;
    for (int i = 0; i < img_files.size(); i++) {
        std::cout << img_files[i] << std::endl;

        // read ins data
        std::string basename = Utils::GetPathBaseName(img_files[i]);
        std::vector<double> ins = ins_data[basename];
        double qy = ins[0] * M_PI / 180;
        double qz = ins[1] * M_PI / 180;
        double pitch = ins[2] * M_PI / 180;
        double yaw = ins[3] * M_PI / 180;
        double roll = ins[4] * M_PI / 180;
        double lat = ins[5] * M_PI / 180;
        double lon = ins[6] * M_PI / 180;
        double alt = ins[7];

        Eigen::Matrix3d R_nue2body = nueToBody(yaw, pitch, roll);
        Eigen::Matrix3d R_body2camera = bodyToCamera(qy, qz);
        Eigen::Vector3d t_nue = llhToNUE(lat, lon, alt, lat0, lon0, alt0);
        Eigen::Matrix3d R_nue = (R_body2camera * R_nue2body).transpose();
        Eigen::Matrix4d T_wc = Eigen::Matrix4d::Identity();
        T_wc.block(0, 0, 3, 3) = R_nue;
        T_wc.block(0, 3, 3, 1) = t_nue;
        poses.push_back(T_wc.cast<float>());

        // print
        for (int i = 0; i < ins.size(); i++) {
            std::cout << ins[i] << " ";
        }
        std::cout << std::endl;

        // init
        if (i == 0) {
            lat0 = lat;
            lon0 = lon;
            alt0 = alt;
        }

        // optical flow
        Frame::Ptr frame(new Frame(frame_next_id++));
        frame->Twb_ = T_wc;
        curr_flow_img = cv::imread(img_files[i], cv::IMREAD_COLOR);
        cv::resize(curr_flow_img, curr_flow_img, cv::Size(curr_flow_img.cols / scale, curr_flow_img.rows / scale));

        cv::Mat curr_flow_img_processed = preprocessImage(curr_flow_img);

        std::vector<Eigen::Vector2i> flow_matches = opticalFlow(prev_flow_img, curr_flow_img_processed, prev_flow_pts, curr_flow_pts);
        
        // 计算光流的中位数
        std::vector<double> flow_distances;
        for (int j = 0; j < flow_matches.size(); j++) {
            cv::Point2f p1 = prev_flow_pts[flow_matches[j].x];
            cv::Point2f p2 = curr_flow_pts[flow_matches[j].y];
            flow_distances.push_back(distance(p1, p2));
        }
        std::sort(flow_distances.begin(), flow_distances.end());
        double median_flow_distance = flow_distances[flow_distances.size() / 2];
        std::cout << "median flow distance: " << median_flow_distance << std::endl;

        // bool is_max_flow = median_flow_distance > 10;
        
        std::vector<Eigen::Vector3d> flow_bearings;
        for (int j = 0; j < curr_flow_pts.size(); j++) {
            Eigen::Vector3d bearing;
            bearing << (curr_flow_pts[j].x - cx) / f, (curr_flow_pts[j].y - cy) / f, 1;
            flow_bearings.push_back(bearing);
        }

        frame->addData({curr_flow_img}, {toEigen(curr_flow_pts)}, {flow_bearings});
        flow_sparse_map->addKeyFrame(frame);
        if (flow_matches.size() > 0)
            flow_sparse_map->addIntraMatches(prev_flow_frame->id_, frame->id_, {flow_matches});

        {
            prev_flow_frame = frame;
            prev_flow_img = curr_flow_img_processed.clone();
            prev_flow_pts = curr_flow_pts;
        }



        // pick keyframe


        // BA

        cv::Mat timg = frame->drawKeyPoint(0);
        server.showImage("image", frame->id_, timg);

        // cv::Mat timg2 = sparse_map->drawMatchedKeypoint(frame->id_, 0);
        // server.showImage("match_image", frame->id_, timg2);

        cv::Mat timg3 = flow_sparse_map->drawFlow(frame->id_, 0);
        server.showImage("flow_image", frame->id_, timg3);

        server.showPath("ins_path", frame->id_, poses, "NUE");

        std::cout << "frame id: " << frame->id_ << std::endl;

    }
  
    return 0;
}

// frame->extractFeature(imgs, "ORB");
// frame->bearings_[0].resize(frame->keypoints_[0].size());
// for (int j = 0; j < frame->keypoints_[0].size(); j++) {
//     Eigen::Vector2d kp = frame->keypoints_[0][j];
//     Eigen::Vector3d bearing;
//     bearing << (kp.x() - cx) / f, (kp.y() - cy) / f, 1;
//     frame->bearings_[0][j] = bearing;
// }

// 判断关键帧


// if (last_frame) {
//     const cv::Mat &descriptors1 = last_frame->descriptors_[0];
//     const cv::Mat &descriptors2 = frame->descriptors_[0];
//     std::vector<cv::DMatch> matches;
//     Matcher matcher;
//     matcher.matchORB(descriptors1, descriptors2, matches);

//     std::vector<Eigen::Vector2i> intra_matches_0;
//     for (int i = 0; i < matches.size(); i++) {
//         Eigen::Vector2i match(matches[i].queryIdx, matches[i].trainIdx);
//         intra_matches_0.push_back(match);
//     }
    
//     std::vector<std::vector<Eigen::Vector2i>> intra_matches;
//     intra_matches.push_back(intra_matches_0);
//     sparse_map->addIntraMatches(last_frame->id_, frame->id_, intra_matches);
// }

// last_frame = frame;