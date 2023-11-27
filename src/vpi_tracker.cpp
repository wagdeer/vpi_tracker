#include "vpi_tracker/vpi_tracker.h"

using namespace std;

#define CHECK_STATUS(STMT)                              \
  do {                                                  \
    VPIStatus status = (STMT);                          \
    if (status != VPI_SUCCESS) {                        \
      char buffer[VPI_MAX_STATUS_MESSAGE_LENGTH];       \
      vpiGetLastStatusMessage(buffer, sizeof(buffer));  \
      std::ostringstream ss;                            \
      ss << vpiStatusGetName(status) << ": " << buffer; \
      throw std::runtime_error(ss.str());               \
    }                                                   \
  } while (0);

// Max number of corners detected by harris corner algo
constexpr int MAX_HARRIS_CORNERS = 8192;

// Max number of keypoints to be tracked
constexpr int MAX_KEYPOINTS = 180;

int VpiTracker::n_id = 0;

void VpiTracker::selectPtsByMask(vector<cv::Point2f>& spts,
                                 vector<cv::Point2f>& dpts, cv::Mat& mask,
                                 int max) {
  dpts.clear();
  for (int i = 0; i < spts.size(); ++i) {
    if (mask.at<uchar>(spts[i]) == 0) {
      continue;
    }
    dpts.emplace_back(spts[i]);
    if (dpts.size() >= max) {
      return;
    }
  }
}

void VpiTracker::selectPtsByMask(std::vector<cv::Point2f>& spts,
                                 VPIArray keypoints, int max, cv::Mat& mask) {
  VPIArrayData ptsData;
  vpiArrayLockData(keypoints, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS,
                   &ptsData);
  VPIArrayBufferAOS& aosKeypoints = ptsData.buffer.aos;

  std::vector<cv::Point2f> kpt(*aosKeypoints.sizePointer);
  VPIKeypointF32* kptData =
      reinterpret_cast<VPIKeypointF32*>(aosKeypoints.data);
  transform(kptData, kptData + (*aosKeypoints.sizePointer), back_inserter(kpt),
            [](VPIKeypointF32 p) { return cv::Point2f(p.x, p.y); });

  vpiArrayUnlock(keypoints);

  spts.clear();
  for (int i = 0; i < kpt.size(); ++i) {
    if (mask.at<uchar>(kpt[i]) != 0) {
      spts.emplace_back(kpt[i]);
      if (spts.size() >= max) {
        break;
      }
    }
  }
}

void VpiTracker::setMask(cv::Mat& mask) {
  mask = cv::Mat(params_->row, params_->col, CV_8UC1, cv::Scalar(255));

  // prefer to keep features that are tracked for long time
  vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

  for (unsigned int i = 0; i < forw_pts.size(); i++)
    cnt_pts_id.push_back(
        make_pair(track_cnt[i], make_pair(forw_pts[i], ids[i])));

  sort(cnt_pts_id.begin(), cnt_pts_id.end(),
       [](const pair<int, pair<cv::Point2f, int>>& a,
          const pair<int, pair<cv::Point2f, int>>& b) {
         return a.first > b.first;
       });

  forw_pts.clear();
  ids.clear();
  track_cnt.clear();

  for (auto& it : cnt_pts_id) {
    if (mask.at<uchar>(it.second.first) == 255) {
      forw_pts.push_back(it.second.first);
      ids.push_back(it.second.second);
      track_cnt.push_back(it.first);
      cv::circle(mask, it.second.first, params_->min_dist, 0, -1);
    }
  }
}

void VpiTracker::addPoints() {
  for (auto& p : n_pts) {
    forw_pts.push_back(p);
    ids.push_back(-1);
    track_cnt.push_back(1);
  }
}

bool VpiTracker::inBorder(const cv::Point2f& pt) {
  const int BORDER_SIZE = 1;
  int img_x = cvRound(pt.x);
  int img_y = cvRound(pt.y);
  return BORDER_SIZE <= img_x && img_x < params_->col - BORDER_SIZE &&
         BORDER_SIZE <= img_y && img_y < params_->row - BORDER_SIZE;
}

template <typename T1, typename T2>
void VpiTracker::reduceVector(std::vector<T1>& v, const std::vector<T2>& s) {
  int j = 0;
  for (int i = 0; i < v.size(); ++i) {
    if (s[i]) {
      v[j++] = v[i];
    }
  }
  v.resize(j);
}

bool VpiTracker::updateID(unsigned int i) {
  if (i < ids.size()) {
    if (ids[i] == -1) ids[i] = n_id++;
    return true;
  }
  return false;
}

void VpiTracker::rejectWithF() {
  if (forw_pts.size() >= 8) {
    vector<cv::Point2f> un_cur_pts(cur_pts.size()),
        un_forw_pts(forw_pts.size());
    for (unsigned int i = 0; i < cur_pts.size(); i++) {
      Eigen::Vector3d tmp_p;
      camera_->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y),
                              tmp_p);
      tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + params_->col / 2.0;
      tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + params_->row / 2.0;
      un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

      camera_->liftProjective(Eigen::Vector2d(forw_pts[i].x, forw_pts[i].y),
                              tmp_p);
      tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + params_->col / 2.0;
      tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + params_->row / 2.0;
      un_forw_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
    }
    vector<uchar> status;
    cv::findFundamentalMat(un_cur_pts, un_forw_pts, cv::FM_RANSAC,
                           params_->f_threshold, 0.99, status);
    int size_a = cur_pts.size();
    reduceVector(prev_pts, status);
    reduceVector(cur_pts, status);
    reduceVector(forw_pts, status);
    reduceVector(cur_un_pts, status);
    reduceVector(ids, status);
    reduceVector(track_cnt, status);
  }
}

void VpiTracker::undistortedPoints() {
  cur_un_pts.clear();
  cur_un_pts_map.clear();
  for (unsigned int i = 0; i < cur_pts.size(); i++) {
    Eigen::Vector2d a(cur_pts[i].x, cur_pts[i].y);
    Eigen::Vector3d b;
    camera_->liftProjective(a, b);
    cur_un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
    cur_un_pts_map.insert(
        make_pair(ids[i], cv::Point2f(b.x() / b.z(), b.y() / b.z())));
  }
  // caculate points velocity
  if (!prev_un_pts_map.empty()) {
    double dt = cur_time - prev_time;
    pts_velocity.clear();
    for (unsigned int i = 0; i < cur_un_pts.size(); i++) {
      if (ids[i] != -1) {
        std::map<int, cv::Point2f>::iterator it;
        it = prev_un_pts_map.find(ids[i]);
        if (it != prev_un_pts_map.end()) {
          double v_x = (cur_un_pts[i].x - it->second.x) / dt;
          double v_y = (cur_un_pts[i].y - it->second.y) / dt;
          pts_velocity.push_back(cv::Point2f(v_x, v_y));
        } else
          pts_velocity.push_back(cv::Point2f(0, 0));
      } else {
        pts_velocity.push_back(cv::Point2f(0, 0));
      }
    }
  } else {
    for (unsigned int i = 0; i < cur_pts.size(); i++) {
      pts_velocity.push_back(cv::Point2f(0, 0));
    }
  }
  prev_un_pts_map = cur_un_pts_map;
}

VpiTracker::VpiTracker() {}

void VpiTracker::setParams(Params& params) {
  params_ = &params;
  if (params_->vpi_backend_ == "cpu") {
    backend_ = VPI_BACKEND_CPU;
  } else if (params_->vpi_backend_ == "cuda") {
    backend_ = VPI_BACKEND_CUDA;
  } else if (params_->vpi_backend_ == "pva") {
    backend_ = VPI_BACKEND_PVA;
  } else {
    throw std::runtime_error(
        "Backend '" + params_->vpi_backend_ +
        "' not recognized, it must be either cpu, cuda or pva.");
  }

  if (params_->equalize) {
    initVpiEqualize();
  }

  cout << "Process Mode: " << endl;
  ProcessMode mode = (ProcessMode)params_->process_mode;
  if (mode == ProcessMode::VpiOptiFlowMode) {
    initVpiOpticalFlow();
    cout << "VPI OpticalFlow Mode." << endl;
  } else if (mode == ProcessMode::VpiFastMode) {
    initVpiFast();
    cout << "VPI Fast Conner Mode." << endl;
  } else if (mode == ProcessMode::OpencvMode) {
    initOpencvCuda();
    cout << "Opencv Mode." << endl;
  }

  setCameraIntrinsic(params_->config_file);
}

void VpiTracker::setCameraIntrinsic(const std::string& config) {
  camera_ =
      camodocal::CameraFactory::instance()->generateCameraFromYamlFile(config);
}

void VpiTracker::initVpiEqualize() {
  vpiCreateEqualizeHist(VPI_BACKEND_CUDA, VPI_IMAGE_FORMAT_U8, &equalize_);
}

void VpiTracker::initOpencvCuda() {
  detector_ = cv::cuda::createGoodFeaturesToTrackDetector(
      CV_8UC1, MAX_KEYPOINTS, 0.005, params_->min_dist);
  pyrLK_gpu_sparse_ =
      cv::cuda::SparsePyrLKOpticalFlow::create(cv::Size(21, 21), 3);
  pyrLK_gpu_sparse_->setUseInitialFlow(false);
  clahe_ = cv::cuda::createCLAHE(60, cv::Size(3, 3));
}

void VpiTracker::initVpiOpticalFlow() {
  // Create the stream where processing will happen.
  CHECK_STATUS(vpiStreamCreate(0, &lk_stream_));
  CHECK_STATUS(vpiStreamCreate(0, &harris_stream_));
  // Create grayscale image representation of input.
  CHECK_STATUS(vpiImageCreate(params_->col, params_->row, VPI_IMAGE_FORMAT_U8,
                              0, &imgFrame_));
  CHECK_STATUS(vpiImageCreate(params_->col, params_->row, VPI_IMAGE_FORMAT_U8,
                              0, &imgGrayscale_));
  // Create the image pyramids used by the algorithm
  CHECK_STATUS(vpiPyramidCreate(params_->col, params_->row, VPI_IMAGE_FORMAT_U8,
                                params_->pyrlevel, 0.5, 0, &pyrPrevFrame_));
  CHECK_STATUS(vpiPyramidCreate(params_->col, params_->row, VPI_IMAGE_FORMAT_U8,
                                params_->pyrlevel, 0.5, 0, &pyrCurFrame_));
  // Create input and output arrays
  CHECK_STATUS(vpiArrayCreate(MAX_HARRIS_CORNERS, VPI_ARRAY_TYPE_KEYPOINT_F32,
                              0, &prevFeatures_));
  CHECK_STATUS(vpiInitOpticalFlowPyrLKParams(&lkParams_));
  lkParams_.useInitialFlow = 0;

  // Create Optical Flow payload
  CHECK_STATUS(vpiCreateOpticalFlowPyrLK(backend_, params_->col, params_->row,
                                         VPI_IMAGE_FORMAT_U8, params_->pyrlevel,
                                         0.5, &optflow_));
  CHECK_STATUS(vpiInitHarrisCornerDetectorParams(&harrisParams_));
  CHECK_STATUS(vpiCreateHarrisCornerDetector(VPI_BACKEND_CUDA, params_->col,
                                             params_->row, &harris_));
  // adjust some parameters
  harrisParams_.strengthThresh = 0;
  harrisParams_.sensitivity = 0.01;

  vpiInitFASTCornerDetectorParams(&fastParams_);
  fastParams_.circleRadius = 3;
  fastParams_.arcLength = 9;
  fastParams_.intensityThreshold = 90;
  fastParams_.nonMaxSuppression = 1;
}

void VpiTracker::initVpiFast() {
  vpiStreamCreate(0, &fast_stream_);
  vpiImageCreate(params_->col, params_->row, VPI_IMAGE_FORMAT_U8, 0,
                 &imgFrame_);
  vpiImageCreate(params_->col, params_->row, VPI_IMAGE_FORMAT_U8, 0,
                 &imgGrayscale_);
  vpiArrayCreate(MAX_HARRIS_CORNERS, VPI_ARRAY_TYPE_KEYPOINT_F32, 0,
                 &fast_corners_);
  vpiInitFASTCornerDetectorParams(&fastParams_);
  fastParams_.circleRadius = 3;
  fastParams_.arcLength = 9;
  fastParams_.intensityThreshold = 100;
  fastParams_.nonMaxSuppression = 1;
}

void VpiTracker::readImage(cv_bridge::CvImageConstPtr& ptr) {
  ProcessMode mode = (ProcessMode)params_->process_mode;
  if (mode == ProcessMode::VpiOptiFlowMode) {
    processByVpiOptiflow(ptr);
  } else if (mode == ProcessMode::VpiFastMode) {
    processByVpiFast(ptr);
  } else if (mode == ProcessMode::OpencvMode) {
    processByOpencv(ptr);
  } else {
    cout << "unknorwn process mode, please check your config file." << endl;
  }
}

void VpiTracker::processByVpiOptiflow(cv_bridge::CvImageConstPtr& ptr) {
  static bool first = true;
  cur_time = ptr->header.stamp.toSec();
  if (first) {
    vpiImageCreateWrapperOpenCVMat(ptr->image, 0, &imgInput_);
    first = false;
  } else {
    // Make the reference wrapper point to the reference frame
    vpiImageSetWrappedOpenCVMat(imgInput_, ptr->image);
  }
  // Convert input to grayscale to conform with harris corner detector
  // restrictions
  vpiSubmitConvertImageFormat(lk_stream_, VPI_BACKEND_CUDA, imgInput_,
                              imgGrayscale_, NULL);
  if (params_->equalize) {
    vpiSubmitEqualizeHist(lk_stream_, VPI_BACKEND_CUDA, equalize_,
                          imgGrayscale_, imgFrame_);
    
  } else {
    imgFrame_ = imgGrayscale_;
  }

  forw_pts.clear();
  vpiArrayCreate(MAX_HARRIS_CORNERS, VPI_ARRAY_TYPE_KEYPOINT_F32, 0,
                 &curFeatures_);
  vpiSubmitGaussianPyramidGenerator(lk_stream_, backend_, imgFrame_,
                                    pyrCurFrame_, VPI_BORDER_CLAMP);
  vpiStreamSync(lk_stream_);
  if (cur_pts.size() > 0) {
    vector<uchar> status;
    vpiArrayCreate(MAX_HARRIS_CORNERS, VPI_ARRAY_TYPE_U8, 0, &status_);
    vpiSubmitOpticalFlowPyrLK(lk_stream_, 0, optflow_, pyrPrevFrame_,
                              pyrCurFrame_, prevFeatures_, curFeatures_,
                              status_, &lkParams_);
    vpiStreamSync(lk_stream_);
    download(curFeatures_, forw_pts);
    download(status_, status);
    vpiArrayDestroy(status_);
    for (int i = 0; i < int(forw_pts.size()); i++)
      if (status[i] && !inBorder(forw_pts[i])) status[i] = 0;

    reduceVector(prev_pts, status);
    reduceVector(cur_pts, status);
    reduceVector(forw_pts, status);
    reduceVector(ids, status);
    reduceVector(cur_un_pts, status);
    reduceVector(track_cnt, status);
  }

  for (auto& n : track_cnt) n++;

  cv::Mat mask;
  rejectWithF();
  setMask(mask);

  int n_max_cnt = MAX_KEYPOINTS - static_cast<int>(forw_pts.size());
  if (n_max_cnt > 0) {
    VPIArray features;
    vpiArrayCreate(MAX_HARRIS_CORNERS, VPI_ARRAY_TYPE_KEYPOINT_F32, 0,
                   &features);
    vpiSubmitFASTCornerDetector(lk_stream_, backend_, imgFrame_, features,
                                &fastParams_, VPI_BORDER_LIMITED);
    vpiStreamSync(lk_stream_);
    selectPtsByMask(n_pts, features, n_max_cnt, mask);
    vpiArrayDestroy(features);
  } else {
    n_pts.clear();
  }
  addPoints();
  upload(curFeatures_, forw_pts);

  prev_pts = cur_pts;
  cur_pts = forw_pts;
  prev_un_pts = cur_un_pts;
  undistortedPoints();
  prev_time = cur_time;
  swap(curFeatures_, prevFeatures_);
  swap(pyrCurFrame_, pyrPrevFrame_);
  if (curFeatures_) {
    vpiArrayDestroy(curFeatures_);
  }
}

void VpiTracker::processByVpiFast(cv_bridge::CvImageConstPtr& ptr) {
  static int frame_cnt = 0;
  static bool first = true;
  cur_time = ptr->header.stamp.toSec();
  cv::Mat img;
  cv::Mat debug_img = ptr->image.clone();
  cv::cvtColor(debug_img, debug_img, CV_GRAY2RGB);
  if (first) {
    // warp cv_img to vpi_img
    vpiImageCreateWrapperOpenCVMat(ptr->image, 0, &imgInput_);
    first = false;
  } else {
    // Make the reference wrapper point to the reference frame
    vpiImageSetWrappedOpenCVMat(imgInput_, ptr->image);
  }
  // Convert input to grayscale to conform with harris corner detector
  // restrictions
  vpiSubmitConvertImageFormat(fast_stream_, VPI_BACKEND_CUDA, imgInput_,
                              imgGrayscale_, NULL);
  if (params_->equalize) {
    vpiSubmitEqualizeHist(fast_stream_, VPI_BACKEND_CUDA, equalize_,
                          imgGrayscale_, imgFrame_);
  } else {
    imgFrame_ = imgGrayscale_;
  }

  // Conner detect
  vpiSubmitFASTCornerDetector(fast_stream_, VPI_BACKEND_CUDA, imgFrame_,
                              fast_corners_, &fastParams_, VPI_BORDER_LIMITED);
  vpiStreamSync(fast_stream_);

  if (forw_img_.empty()) {
    prev_img_ = cur_img_ = forw_img_ = ptr->image;
  } else {
    forw_img_ = ptr->image;
  }

  forw_pts.clear();
  if (cur_pts.size() > 0) {
    vector<uchar> status;
    vector<float> err;
    cv::calcOpticalFlowPyrLK(cur_img_, forw_img_, cur_pts, forw_pts, status,
                             err, cv::Size(21, 21), 3);

    for (int i = 0; i < int(forw_pts.size()); i++) {
      if (status[i] && !inBorder(forw_pts[i])) {
        status[i] = 0;
      }
    }

    reduceVector(prev_pts, status);
    reduceVector(cur_pts, status);
    reduceVector(forw_pts, status);
    reduceVector(ids, status);
    reduceVector(cur_un_pts, status);
    reduceVector(track_cnt, status);
  }

  for (auto& n : track_cnt) {
    n++;
  }

  if (pub_this_frame_) {
    cv::Mat mask;
    rejectWithF();
    setMask(mask);
    int n_max_cnt = MAX_KEYPOINTS - static_cast<int>(forw_pts.size());
    if (n_max_cnt > 0) {
      selectPtsByMask(n_pts, fast_corners_, n_max_cnt, mask);
    } else {
      n_pts.clear();
    }
    addPoints();
  }

  prev_img_ = cur_img_;
  prev_pts = cur_pts;
  cur_img_ = forw_img_;
  cur_pts = forw_pts;
  undistortedPoints();
  prev_time = cur_time;
  frame_cnt++;
}

void VpiTracker::processByOpencv(cv_bridge::CvImageConstPtr& ptr) {
  forw_gpu_img_.upload(ptr->image);
  if (equalize_) {
    clahe_->apply(forw_gpu_img_, forw_gpu_img_);
  }
  forw_pts.clear();
  if (cur_pts.size() > 0) {
    vector<uchar> status;
    cv::Mat tmp_pts(cur_pts);
    cv::transpose(tmp_pts, tmp_pts);
    cv::cuda::GpuMat cur_gpu_pts(tmp_pts), forw_gpu_pts, d_status;

    pyrLK_gpu_sparse_->calc(cur_gpu_img_, forw_gpu_img_, cur_gpu_pts,
                            forw_gpu_pts, d_status);
    download(d_status, status);
    download(forw_gpu_pts, forw_pts);
    for (int i = 0; i < int(forw_pts.size()); i++) {
      if (status[i] && !inBorder(forw_pts[i])) {
        status[i] = 0;
      }
    }

    reduceVector(prev_pts, status);
    reduceVector(cur_pts, status);
    reduceVector(forw_pts, status);
    reduceVector(ids, status);
    reduceVector(cur_un_pts, status);
    reduceVector(track_cnt, status);
  }

  for (auto& n : track_cnt) {
    n++;
  }

  cv::Mat mask;
  rejectWithF();
  setMask(mask);
  int n_max_cnt = MAX_KEYPOINTS - static_cast<int>(forw_pts.size());
  if (n_max_cnt > 0) {
    vector<cv::Point2f> vpts;
    cv::cuda::GpuMat pts;
    detector_->detect(forw_gpu_img_, pts);
    download(pts, vpts);
    selectPtsByMask(vpts, n_pts, mask, n_max_cnt);
  } else {
    n_pts.clear();
  }
  addPoints();

  prev_pts = move(cur_pts);
  cur_pts = move(forw_pts);
  prev_un_pts = move(cur_un_pts);
  swap(cur_gpu_img_, forw_gpu_img_);
  undistortedPoints();
  prev_time = cur_time;
}