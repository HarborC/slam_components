#include <string>
#include <iostream>
#include <vector>
#include <Eigen/Core>

#include "utils/io_utils.h"
#include "utils/log_utils.h"
#include "sparse_map/sparse_map.h"

using namespace Eigen;

// Step 1: read images sequence from the directory
// Step 2: extract features from the images
// Step 3: match features between images

int main (int argc, char** argv) {

    std::string img_dir = "/mnt/h/TMP/TXPJ/test1/sub";
    std::string img_format = "jpg";

    std::vector<std::string> img_files = Utils::GetFileList(img_dir);
    std::sort(img_files.begin(), img_files.end());

    SparseMap::Ptr sparse_map(new SparseMap());

    for (int i = 0; i < img_files.size(); i++) {
        std::cout << img_files[i] << std::endl;

        Frame::Ptr frame(new Frame(frame_next_id++));
        std::vector<cv::Mat> imgs;
        cv::Mat img = cv::imread(img_files[i], cv::IMREAD_COLOR);
        imgs.push_back(img);
        
        frame->extractFeature(imgs, "ORB");
        // keshihua

        
        sparse_map->addKeyFrame(frame);
    }
  
    return 0;
}