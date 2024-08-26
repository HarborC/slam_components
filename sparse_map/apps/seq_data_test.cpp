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

int main (int argc, char** argv) {

    Visualizer server(8088);

    std::string img_dir = "/mnt/g/projects/slam/datasets/TXPJ/test1/sub";
    std::string img_format = "jpg";

    std::vector<std::string> img_files = Utils::GetFileList(img_dir);
    std::sort(img_files.begin(), img_files.end());

    SparseMap::Ptr sparse_map(new SparseMap());

    Frame::Ptr last_frame;
    for (int i = 0; i < img_files.size(); i++) {
        std::cout << img_files[i] << std::endl;

        Frame::Ptr frame(new Frame(frame_next_id++));
        std::vector<cv::Mat> imgs;
        cv::Mat img = cv::imread(img_files[i], cv::IMREAD_COLOR);

        // 降采样4倍
        cv::resize(img, img, cv::Size(img.cols / 4, img.rows / 4));

        imgs.push_back(img);
        
        frame->extractFeature(imgs, "ORB");
        // keshihua
        cv::Mat timg = frame->drawKeyPoint(0);
        server.showImage("image", frame->id_, timg);
        
        sparse_map->addKeyFrame(frame);

        if (last_frame) {
            const cv::Mat &descriptors1 = last_frame->descriptors_[0];
            const cv::Mat &descriptors2 = frame->descriptors_[0];
            std::vector<cv::DMatch> matches;
            Matcher matcher;
            matcher.matchORB(descriptors1, descriptors2, matches);
        }

        last_frame = frame;
    }
  
    return 0;
}