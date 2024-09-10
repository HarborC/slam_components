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
Eigen::Vector3d llhToENU(double lat, double lon, double hei, double lat0, double lon0, double hei0 = 0) {

    Eigen::Vector3d xyz0 = llhToECEF(lat0, lon0, hei0);

    Eigen::Vector3d xyz = llhToECEF(lat, lon, hei);

    Eigen::Vector3d dxyz = xyz - xyz0;

    Eigen::Matrix3d R;
    R << -std::sin(lon0), std::cos(lon0), 0,
         -std::sin(lat0) * std::cos(lon0), -std::sin(lat0) * std::sin(lon0), std::cos(lat0),
         std::cos(lat0) * std::cos(lon0), std::cos(lat0) * std::sin(lon0), std::sin(lat0);
         

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
        for (int i = 1; i < items.size(); i++) {
            data.push_back(std::stod(items[i]));
        }
        ins_data[name] = data;
    }

    file.close();

    // // print
    // for (auto it = ins_data.begin(); it != ins_data.end(); it++) {
    //     std::cout << it->first << ": ";
    //     for (int i = 0; i < it->second.size(); i++) {
    //         std::cout << it->second[i] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << "ins_data size: " << ins_data.size() << std::endl;

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

std::vector<Eigen::Vector2i> opticalFlow(const cv::Mat &prev_img, const cv::Mat& curr_img, const std::vector<cv::Point2f> &prev_pts, std::vector<cv::Point2f> &curr_pts, const cv::Mat& curr_mask, bool double_check = false) {
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

    cv::Mat mask = getVaildMask(curr_pts, curr_mask);
    
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

Eigen::Vector3d R2Omega(const Eigen::Matrix3d &R) {
    Eigen::Vector3d euler;
    euler(0) = -std::atan2(R(0, 2), R(2, 2));
    euler(1) = std::asin(-R(1, 2));
    euler(2) = std::atan2(R(1, 0), R(1, 1));
    return euler;
}

Eigen::Matrix3d Omega2R(const Eigen::Vector3d &r) {
    Eigen::Matrix3d R1;
    R1 << cos(r(0)), 0, -sin(r(0)),
          0, 1, 0,
          sin(r(0)), 0, cos(r(0));

    Eigen::Matrix3d R2;
    R2 << 1, 0, 0,
          0, cos(r(1)), -sin(r(1)),
          0, sin(r(1)), cos(r(1));

    Eigen::Matrix3d R3;
    R3 << cos(r(2)), -sin(r(2)), 0,
          sin(r(2)), cos(r(2)), 0,
          0, 0, 1;

    return R1 * R2 * R3;
}

Eigen::Matrix3d RNormalized(const Eigen::Matrix3d &R) {
    Eigen::Quaterniond q(R);
    Eigen::Matrix3d R_normalized = q.normalized().toRotationMatrix();
    return R_normalized;
}

void preprocessImage(const cv::Mat &img, cv::Mat &img_processed, cv::Mat &mask) {
    cv::Mat img_gray;
    if (img.channels() == 3) {
        cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
    } else {
        img_gray = img.clone();
    }
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();

    clahe->apply(img_gray, img_processed);

    mask = cv::Mat::ones(img.size(), CV_8UC1) * 255;

    for (int row = 0; row < img.rows; ++row) {
        for (int col = 0; col < img.cols; ++col) {
            cv::Vec3b pixelColor = img.at<cv::Vec3b>(row, col);
            if (pixelColor(0) != pixelColor(1) || pixelColor(0) != pixelColor(2)) {
                mask.at<uchar>(row, col) = 0;
            }
        }
    }

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(10, 10));
    cv::erode(mask, mask, kernel);
}

int main (int argc, char** argv) {
    Visualizer server(8088);
    std::string img_dir = "../../../datasets/TXPJ/test2/raw_data/img2_raw/";
    std::string img_format = "jpg";
    
    auto ins_data = readCSV("../../../datasets/TXPJ/test2/raw_data/ins2.csv");

    int scale = 4;

    double f = 33400 / scale;
    double cx = 4096 / 2 / scale;
    double cy = 3072 / 2 / scale;

    std::vector<std::string> img_files = Utils::GetFileList(img_dir);
    std::sort(img_files.begin(), img_files.end());

    std::vector<Eigen::Matrix4f> poses;

    SparseMap::Ptr flow_sparse_map(new SparseMap(true));
    SparseMap::Ptr sparse_map(new SparseMap(true));

    double lat0, lon0, alt0;
    lon0 = 110.365814 * M_PI / 180;
    lat0 = 35.304042 * M_PI / 180;
    alt0 = 0;
    Frame::Ptr prev_flow_frame;
    cv::Mat prev_flow_img, curr_flow_img;
    std::vector<cv::Point2f> prev_flow_pts, curr_flow_pts;
    FrameIDType last_kf_id;
    std::ofstream out_file("../../../datasets/TXPJ/test2/raw_data/poses.txt");
    for (int i = 0; i < img_files.size(); i++) {
        std::cout << sparse_map->frame_map_.size() << std::endl;
        std::cout << img_files[i] << std::endl;

        // read ins data
        std::string basename = Utils::GetPathBaseName(img_files[i]);
        std::vector<double> ins = ins_data[basename];
        double qy = ins[0] * M_PI / 180;
        double qz = ins[1] * M_PI / 180;
        double pitch = ins[2] * M_PI / 180;
        double yaw = ins[3] * M_PI / 180;
        double roll = ins[4] * M_PI / 180;
        double lat = ins[6] * M_PI / 180;
        double lon = ins[5] * M_PI / 180;
        double alt = ins[7];

        Eigen::Matrix3d R_nue2body = nueToBody(yaw, pitch, roll);
        Eigen::Matrix3d R_body2camera = bodyToCamera(qy, qz);
        Eigen::Vector3d t_enu = llhToENU(lat, lon, alt, lat0, lon0, alt0);
        Eigen::Matrix3d R_nue =R_nue2body * R_body2camera;
        Eigen::Matrix4d T_wc = Eigen::Matrix4d::Identity();

        Eigen::Matrix3d R_enu_nue; 
        R_enu_nue << 0, 0, 1,
                     1, 0, 0,
                     0, 1, 0;

        Eigen::Matrix3d R_2;
        R_2 << -1, 0, 0,
               0, 1, 0,
               0, 0, -1; 

        T_wc.block(0, 0, 3, 3) = RNormalized(R_enu_nue * R_nue * R_2);
        T_wc.block(0, 3, 3, 1) = t_enu;

        // print
        std::cout << "ins_data: ";
        for (int i = 0; i < ins.size(); i++) {
            std::cout << ins[i] << " ";
        }
        std::cout << std::endl;

        out_file << basename << " ";
        Eigen::Vector3d euler = R2Omega(T_wc.block(0, 0, 3, 3));

        out_file << std::fixed << std::setprecision(6) << euler(0) << " " << euler(1) << " " << euler(2) << " ";
        out_file << std::fixed << std::setprecision(6) << T_wc(0, 3) << " " << T_wc(1, 3) << " " << T_wc(2, 3) << " ";
        out_file << std::endl;

        // optical flow
        Frame::Ptr frame(new Frame(flow_sparse_map->frame_next_id++));
        frame->setBodyPose(T_wc);
        frame->Twb_prior_ = T_wc;
        curr_flow_img = cv::imread(img_files[i], cv::IMREAD_COLOR);
        cv::resize(curr_flow_img, curr_flow_img, cv::Size(curr_flow_img.cols / scale, curr_flow_img.rows / scale));

        cv::Mat curr_flow_img_processed, curr_mask;
        preprocessImage(curr_flow_img, curr_flow_img_processed, curr_mask);

        std::vector<Eigen::Vector2i> flow_matches = opticalFlow(prev_flow_img, curr_flow_img_processed, prev_flow_pts, curr_flow_pts, curr_mask);
        
        // 计算光流的中位数
        if (flow_matches.size()) {
            std::vector<double> flow_distances;
            for (int j = 0; j < flow_matches.size(); j++) {
                cv::Point2f p1 = prev_flow_pts[flow_matches[j].x()];
                cv::Point2f p2 = curr_flow_pts[flow_matches[j].y()];
                flow_distances.push_back(distance(p1, p2));
            }
            std::sort(flow_distances.begin(), flow_distances.end());
            double median_flow_distance = flow_distances[flow_distances.size() / 2];
            std::cout << "median flow distance: " << median_flow_distance << std::endl;

            bool is_max_flow = median_flow_distance > 10;
        }
        
        std::vector<Eigen::Vector3d> flow_bearings;
        for (int j = 0; j < curr_flow_pts.size(); j++) {
            Eigen::Vector3d bearing;
            bearing << (curr_flow_pts[j].x - cx) / f, (curr_flow_pts[j].y - cy) / f, 1;
            flow_bearings.push_back(bearing);
        }

        std::cout << sparse_map->frame_map_.size() << std::endl;

        frame->addData({curr_flow_img}, {toEigen(curr_flow_pts)}, {flow_bearings});
        flow_sparse_map->addKeyFrame(frame);
        if (flow_matches.size() > 0)
            flow_sparse_map->addIntraMatches(prev_flow_frame->id_, frame->id_, {flow_matches});

        std::cout << sparse_map->frame_map_.size() << std::endl;

        {
            prev_flow_frame = frame;
            prev_flow_img = curr_flow_img_processed.clone();
            prev_flow_pts = curr_flow_pts;
        }

        // pick keyframe
        if (i > 0) {
            std::vector<std::pair<size_t, size_t>> matches = flow_sparse_map->getMatches(last_kf_id, 0, frame->id_, 0);
            double kf_num1 = flow_sparse_map->getKeypointSize(last_kf_id, 0);
            double kf_num2 = flow_sparse_map->getKeypointSize(frame->id_, 0);
            double ratio1 = matches.size() / kf_num1;
            double ratio2 = matches.size() / kf_num2;

            double TH_ratio = 0.5;
            if (ratio1 < TH_ratio || ratio2 < TH_ratio) {
                std::cout << std::fixed << "keyframe: " << frame->id_ << " ratio1: " << ratio1 << " ratio2: " << ratio2 << std::endl;              
            } else {
                continue;
            }
        }

        last_kf_id = frame->id_;

        poses.push_back(T_wc.cast<float>());

        Frame::Ptr new_frame(new Frame(sparse_map->frame_next_id++));
        new_frame->setBodyPose(T_wc);
        new_frame->Twb_prior_ = T_wc;
        new_frame->extractFeature(frame->imgs_, "ORB");
        new_frame->bearings_[0].resize(new_frame->keypoints_[0].size());
        for (int j = 0; j < new_frame->keypoints_[0].size(); j++) {
            Eigen::Vector2d kp = new_frame->keypoints_[0][j];
            Eigen::Vector3d bearing;
            bearing << (kp.x() - cx) / f, (kp.y() - cy) / f, 1;
            new_frame->bearings_[0][j] = bearing;
        }

        if (sparse_map->last_frame_) {
            const cv::Mat &descriptors1 = sparse_map->last_frame_->descriptors_[0];
            const cv::Mat &descriptors2 = new_frame->descriptors_[0];
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
            auto prev_kf_id = sparse_map->last_frame_->id_;
            sparse_map->addKeyFrame(new_frame);

            std::cout << sparse_map->frame_map_.size() << std::endl;

            sparse_map->addIntraMatches(prev_kf_id, new_frame->id_, intra_matches);

            // cv::Mat timg5 = sparse_map->drawMatches(new_frame->id_, 0, new_frame->id_-1, 0);
            // cv::imwrite("/mnt/g/projects/slam/tmp/test/matches_" + std::to_string(frame->id_) + "_" + std::to_string(frame->id_-1) + ".png", timg5);

            sparse_map->triangulate();
            sparse_map->bundleAdjustment(f, f, cx, cy);
        } else {
            std::cout << sparse_map->frame_map_.size() << std::endl;
            sparse_map->addKeyFrame(new_frame);
            std::cout << sparse_map->frame_map_.size() << std::endl;

        }

        std::cout << "ppppppppp" << sparse_map->frame_map_.size() << std::endl;

        cv::Mat timg = frame->drawKeyPoint(0);
        server.showImage("image", frame->id_, timg);

        cv::Mat timg3 = flow_sparse_map->drawFlow(frame->id_, 0);
        server.showImage("flow_image", frame->id_, timg3);

        cv::Mat timg4 = sparse_map->drawFlow(new_frame->id_, 0, new_frame->id_-1);
        server.showImage("flow_image2", new_frame->id_, timg4);

        std::cout << sparse_map->frame_map_.size() << std::endl;
        std::vector<Eigen::Vector3d> world_points = sparse_map->getWorldPoints();
        std::cout << sparse_map->frame_map_.size() << std::endl;

        std::vector<std::vector<float>> world_points_f;
        for (int i = 0; i < world_points.size(); i++) {
            std::vector<float> pt = {world_points[i].x(), world_points[i].y(), world_points[i].z()};
            world_points_f.push_back(pt);
        }

        server.showPointCloud("world_points", new_frame->id_, world_points_f, {}, "ENU");

        server.showPath("ins_path", frame->id_, poses, "ENU");
    }

    return 0;
}
