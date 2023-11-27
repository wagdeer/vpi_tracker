#pragma once

#include <ros/ros.h>
#include <vpi/Array.h>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include <vector>

struct Params
{
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

void readParameters(ros::NodeHandle& n, Params& params);
void showParameters(const Params& params);

// gpu
void download(const cv::cuda::GpuMat& src, std::vector<cv::Point2f>& dst);
void download(const cv::cuda::GpuMat& src, std::vector<uchar>& dst);
void download(const VPIArray src, std::vector<cv::Point2f>& dst);
void download(const VPIArray src, std::vector<uchar>& dst);
void upload(cv::cuda::GpuMat& src, std::vector<cv::Point2f>& dst);
void upload(const VPIArray src, std::vector<cv::Point2f>& dst);