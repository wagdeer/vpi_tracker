#include "vpi_tracker/utils.h"
#include <string>

template <typename T>
T readParam(ros::NodeHandle &n, std::string name) {
  T ans;
  if (n.getParam(name, ans)) {
    ROS_INFO_STREAM("Loaded " << name << ": " << ans);
  } else {
    ROS_ERROR_STREAM("Failed to load " << name);
    n.shutdown();
  }
  return ans;
}

void readParameters(ros::NodeHandle &n, Params &params) {
  std::string config_file;
  config_file = readParam<std::string>(n, "config_file");
  cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
  if (!fsSettings.isOpened()) {
    std::cerr << "ERROR: Wrong path to settings" << std::endl;
  }

  params.config_file = config_file;
  fsSettings["vpi_backend"] >> params.vpi_backend_;
  fsSettings["image_topic"] >> params.image_topic_;
  params.show_track = fsSettings["show_track"];
  params.row = fsSettings["image_height"];
  params.col = fsSettings["image_width"];
  params.pyrlevel = fsSettings["pyr_level"];
  params.equalize = fsSettings["equalize"];
  params.freq = fsSettings["freq"];
  params.min_dist = fsSettings["min_dist"];
  params.process_mode = fsSettings["process_mode"];
  params.f_threshold = fsSettings["F_threshold"];
  if (params.freq <= 0) {
    params.freq = 100;
  }
  fsSettings.release();
}

void showParameters(const Params &params) {
  printf("Parameters:\n");
  printf("\timage topic: %s\n", params.image_topic_.c_str());
  printf("\timage_height: %d\n", params.row);
  printf("\timage_width: %d\n\n", params.col);
}