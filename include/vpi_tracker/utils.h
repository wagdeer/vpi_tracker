#pragma once

#include <ros/ros.h>
#include <opencv2/highgui/highgui.hpp>
#include <string>

struct Params {
  std::string config_file;
  std::string image_topic_;
  std::string vpi_backend_;
  int row;
  int col;
  int pyrlevel;
  int freq;
  int equalize;
  int min_dist;
  int process_mode;
  int show_track;
  bool fisheye;
  double f_threshold;
};

void readParameters(ros::NodeHandle &n, Params &params);
void showParameters(const Params &params);