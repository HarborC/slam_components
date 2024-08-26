#include <string>
#include <iostream>
#include <vector>
#include <Eigen/Core>

#include "utils/io_utils.h"
#include "utils/log_utils.h"
#include "sparse_map/sparse_map.h"

#include "foxglove/foxglove_server.h"
#include "foxglove/proto/FrameTransform.pb.h"
#include "foxglove/proto/PointCloud.pb.h"
#include "foxglove/proto/PoseInFrame.pb.h"
#include "foxglove/proto/PosesInFrame.pb.h"
#include "foxglove/utility.h"

using namespace Eigen;

// Step 1: read images sequence from the directory
// Step 2: extract features from the images
// Step 3: match features between images

namespace fg_msg = ::foxglove;
namespace fg = ::foxglove_viz::foxglove;
using namespace foxglove_viz::foxglove;

int main (int argc, char** argv) {

    FoxgloveServer server(8088);
    server.Run();
    std::this_thread::sleep_for(std::chrono::seconds(5));

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