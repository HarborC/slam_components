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

// WGS84楠球参数
const double a = 6378137.0;
const double b = 6356752.3142;
const double f = (a - b) / a;

Eigen::Vector3d llhToECEF(double lat, double lon, double alt) {
    double e2 = f * (2 - f);

    double N = a / std::sqrt(1 - e2 * std::sin(lat) * std::sin(lat));

    Eigen::Vector3d xyz;
    xyz(0) = (N + alt) * std::cos(lat) * std::cos(lon);
    xyz(1) = (N + alt) * std::cos(lat) * std::sin(lon);
    xyz(2) = (N * (1 - e2) + alt) * std::sin(lat);

    return xyz;
}

// llh to NUE(北天东)坐标系
Eigen::Vector3d ecefToNUE(double x, double y, double z, double lat0, double lon0) {
    Eigen::Vector3d nue;


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

int main (int argc, char** argv) {

    Visualizer server(8088);

    
    std::string img_dir = "../../../datasets/TXPJ/test1/sub";
    std::string img_format = "jpg";

    auto ins_data = readCSV("../../../datasets/TXPJ/test1/ins1.csv");

    // return 0;

    int scale = 4;

    double f = 33400 / scale;
    double cx = 4096 / 2 / scale;
    double cy = 3072 / 2 / scale;

    std::vector<std::string> img_files = Utils::GetFileList(img_dir);
    std::sort(img_files.begin(), img_files.end());

    SparseMap::Ptr sparse_map(new SparseMap(true));

    Frame::Ptr last_frame;
    for (int i = 0; i < img_files.size(); i++) {
        std::cout << img_files[i] << std::endl;

        Frame::Ptr frame(new Frame(frame_next_id++));
        std::vector<cv::Mat> imgs;
        cv::Mat img = cv::imread(img_files[i], cv::IMREAD_COLOR);

        // 降采样4倍
        cv::resize(img, img, cv::Size(img.cols / scale, img.rows / scale));

        imgs.push_back(img);
        
        frame->extractFeature(imgs, "ORB");

        frame->bearings_[0].resize(frame->keypoints_[0].size());
        for (int j = 0; j < frame->keypoints_[0].size(); j++) {
            Eigen::Vector2d kp = frame->keypoints_[0][j];
            Eigen::Vector3d bearing;
            bearing << (kp.x() - cx) / f, (kp.y() - cy) / f, 1;
            frame->bearings_[0][j] = bearing;
        }

        // keshihua
        cv::Mat timg = frame->drawKeyPoint(0);
        server.showImage("image", frame->id_, timg);
        
        sparse_map->addKeyFrame(frame);

        if (last_frame) {
            const cv::Mat &descriptors1 = last_frame->descriptors_[0];
            std::cout << "descriptors1 size: " << descriptors1.rows << std::endl;
            const cv::Mat &descriptors2 = frame->descriptors_[0];
            std::cout << "descriptors2 size: " << descriptors2.rows << std::endl;
            std::vector<cv::DMatch> matches;
            Matcher matcher;
            matcher.matchORB(descriptors1, descriptors2, matches);
            std::cout << "matches size: " << matches.size() << std::endl;

            std::vector<Eigen::Vector2i> intra_matches_0;
            for (int i = 0; i < matches.size(); i++) {
                Eigen::Vector2i match(matches[i].queryIdx, matches[i].trainIdx);
                intra_matches_0.push_back(match);
            }
            
            std::vector<std::vector<Eigen::Vector2i>> intra_matches;
            intra_matches.push_back(intra_matches_0);
            std::cout << "addIntraMatches done1" << std::endl;
            sparse_map->addIntraMatches(last_frame->id_, frame->id_, intra_matches);
            std::cout << "addIntraMatches done" << std::endl;
        }

        // cv::Mat timg2 = sparse_map->drawMatchedKeypoint(frame->id_, 0);
        // server.showImage("match_image", frame->id_, timg2);

        cv::Mat timg3 = sparse_map->drawFlow(frame->id_, 0);
        server.showImage("flow_image", frame->id_, timg3);

        std::cout << "frame id: " << frame->id_ << std::endl;

        last_frame = frame;
    }
  
    return 0;
}