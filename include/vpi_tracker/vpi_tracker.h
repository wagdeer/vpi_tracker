#pragma once

#include <cv_bridge/cv_bridge.h>

#include <bitset>
#include <map>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>

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

const double FOCAL_LENGTH = 600.0;

enum class ProcessMode : int
{
    VpiOptiFlowMode = 1,
    VpiFastMode = 2,
    OpencvMode = 3
};

class VpiTracker
{
public:
    VpiTracker();

    void setParams(Params &params);

    bool updateID(unsigned int i);
    void readImage(cv_bridge::CvImageConstPtr &ptr);

    bool pub_this_frame_;

private:
    void setMask(cv::Mat &mask);
    bool inBorder(const cv::Point2f &pt);
    void addPoints();
    void setCameraIntrinsic(const std::string &config);

    template <typename T1, typename T2>
    void reduceVector(std::vector<T1> &v, const std::vector<T2> &s);
    void rejectWithF();
    void selectPtsByMask(std::vector<cv::Point2f> &spts,
                         std::vector<cv::Point2f> &dpts, cv::Mat &mask, int max);
    void selectPtsByMask(std::vector<cv::Point2f> &spts, VPIArray keypoints,
                         int max, cv::Mat &mask);
    void undistortedPoints();

    void initVpiOpticalFlow();
    void initVpiFast();
    void initVpiEqualize();
    void initOpencvCuda();
    void processByVpiOptiflow(cv_bridge::CvImageConstPtr &ptr);
    void processByVpiFast(cv_bridge::CvImageConstPtr &ptr);
    void processByOpencv(cv_bridge::CvImageConstPtr &ptr);

    Params *params_;

    // vpi harris conner detect
    VPIBackend backend_;
    VPIStream fast_stream_, lk_stream_, harris_stream_, equalize_stream_;
    VPIArray fast_corners_, status_, prevFeatures_, curFeatures_;
    VPIPyramid pyrPrevFrame_, pyrCurFrame_;
    VPIPayload harris_, optflow_, equalize_;
    VPIImage imgFrame_, imgInput_, imgGrayscale_;
    VPIHarrisCornerDetectorParams harrisParams_;
    VPIOpticalFlowPyrLKParams lkParams_;
    VPIFASTCornerDetectorParams fastParams_;

    // opencv cuda
    cv::Ptr<cv::cuda::CLAHE> clahe_;
    cv::Ptr<cv::cuda::CornersDetector> detector_;
    cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> pyrLK_gpu_sparse_;
    cv::cuda::GpuMat cur_gpu_img_, forw_gpu_img_;

    camodocal::CameraPtr camera_;

    std::map<int, cv::Point2f> cur_un_pts_map, prev_un_pts_map;
    cv::Mat prev_img_, cur_img_, forw_img_;
    double cur_time, prev_time;

public:
    std::vector<cv::Point2f> n_pts;
    std::vector<cv::Point2f> pts_velocity;
    std::vector<cv::Point2f> prev_pts, cur_pts, forw_pts;
    std::vector<cv::Point2f> prev_un_pts, cur_un_pts;
    std::vector<int> ids;
    std::vector<int> track_cnt;

    static int n_id;
};