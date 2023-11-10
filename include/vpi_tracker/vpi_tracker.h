#pragma once

#include <cv_bridge/cv_bridge.h>

#include <bitset>
#include <map>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>

#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaoptflow.hpp>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include "vpi_tracker/utils.h"

// vpi third party
#include <vpi/Array.h>
#include <vpi/Context.h>
#include <vpi/Event.h>
#include <vpi/Image.h>
#include <vpi/Pyramid.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/EqualizeHist.h>
#include <vpi/algo/FASTCorners.h>
#include <vpi/algo/GaussianPyramid.h>
#include <vpi/algo/HarrisCorners.h>
#include <vpi/algo/ORB.h>
#include <vpi/algo/OpticalFlowPyrLK.h>
#include <vpi/OpenCVInterop.hpp>

const double FOCAL_LENGTH = 460.0;

enum class ProcessMode : int {
  VpiOptiFlowMode = 1,
  VpiFastMode = 2,
  OpencvMode = 3
};

class VpiTracker {
 public:
  VpiTracker();
  void setMask(cv::Mat &mask);
  void setParams(Params &params);
  void setCameraIntrinsic(const std::string &config);
  void initVpiOpticalFlow();

  void initVpiFast();
  void initVpiEqualize();
  void initOpencvCuda();
  bool updateID(unsigned int i);
  void readImage(cv_bridge::CvImageConstPtr &ptr);
  void processByVpiOptiflow(cv_bridge::CvImageConstPtr &ptr);
  void processByVpiFast(cv_bridge::CvImageConstPtr &ptr);
  void processByOpencv(cv_bridge::CvImageConstPtr &ptr);

  bool pub_this_frame_;

 private:
  bool inBorder(const cv::Point2f &pt);
  void addPoints();
  template <typename T1, typename T2>
  void reduceVector(std::vector<T1> &v, const std::vector<T2> &s);
  void rejectWithF();
  void selectPtsByMask(std::vector<cv::Point2f> &spts,
                       std::vector<cv::Point2f> &dpts, cv::Mat &mask, int max);
  void selectPtsByMask(std::vector<cv::Point2f> &spts, VPIArray keypoints,
                       int max, cv::Mat &mask);
  void undistortedPoints();

  Params *params_;

  // vpi harris conner detect
  VPIContext ctx_;
  VPIBackend backend_;
  VPIEvent barrier_harris_;
  VPIStream fast_stream_;
  VPIStream lk_stream_;
  VPIStream harris_stream_;
  VPIStream orb_stream_;
  VPIStream equalize_stream_;
  VPIArray fast_corners_;
  VPIArray orb_corners_;
  VPIArray orb_descriptors;
  VPIArray scores_;
  VPIArray status_;
  VPIArray prevFeatures_;
  VPIArray curFeatures_;
  VPIPyramid pyrPrevFrame_;
  VPIPyramid pyrCurFrame_;
  VPIPayload harris_;
  VPIPayload optflow_;
  VPIPayload orb_;
  VPIPayload equalize_;
  VPIImage imgFrame_;
  VPIImage imgInput_;
  VPIImage imgInput2_;
  VPIImage imgGrayscale_;
  VPIImage imgGrayscale2_;
  VPIImage imgEqualize_;
  VPIHarrisCornerDetectorParams harrisParams_;
  VPIOpticalFlowPyrLKParams lkParams_;
  VPIFASTCornerDetectorParams fastParams_;
  VPIORBParams orb_params_;

  // opencv cuda
  cv::Ptr<cv::cuda::CornersDetector> detector_;
  cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> pyrLK_gpu_sparse_;
  cv::cuda::GpuMat prev_gpu_img_, cur_gpu_img_, forw_gpu_img_;
  cv::cuda::GpuMat gpu_status_, n_gpu_pts;
  cv::cuda::GpuMat cur_gpu_pts_, forw_gpu_pts_;

  camodocal::CameraPtr camera_;

 public:
  std::map<int, cv::Point2f> cur_un_pts_map, prev_un_pts_map;
  std::vector<cv::Point2f> n_pts;
  std::vector<cv::Point2f> pts_velocity;
  std::vector<cv::Point2f> prev_pts, cur_pts, forw_pts;
  std::vector<cv::Point2f> prev_un_pts, cur_un_pts;
  std::vector<int> ids;
  std::vector<int> track_cnt;
  double cur_time;
  double prev_time;
  static int n_id;

  cv::Mat prev_img_, cur_img_, forw_img_;
};