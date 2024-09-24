#include "components/system.h"

int main(int argc, char **argv) {
  std::string config_file =
      "/mnt/i/project/slam/thirdparty/slam_components/configs/eruco/mh02.yaml";

  slam_components::System::Ptr system(new slam_components::System());
  if (!system->initialize(config_file)) {
    std::cerr << "Error: Failed to initialize System\n";
    return -1;
  }

  return 0;
}