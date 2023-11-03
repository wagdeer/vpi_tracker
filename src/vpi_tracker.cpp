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

static cv::Mat DrawKeypoints(cv::Mat img, VPIKeypointF32 *kpts,
                             VPIBriefDescriptor *descs, int numKeypoints) {
  cv::Mat out;
  img.convertTo(out, CV_8UC1);
  cvtColor(out, out, cv::COLOR_GRAY2BGR);

  if (numKeypoints == 0) {
    return out;
  }

  std::vector<int> distances(numKeypoints, 0);
  float maxDist = 0.f;

  for (int i = 0; i < numKeypoints; i++) {
    for (int j = 0; j < VPI_BRIEF_DESCRIPTOR_ARRAY_LENGTH; j++) {
      distances[i] +=
          std::bitset<8 * sizeof(uint8_t)>(descs[i].data[j] ^ descs[0].data[j])
              .count();
    }
    if (distances[i] > maxDist) {
      maxDist = distances[i];
    }
  }

  uint8_t ids[256];
  std::iota(&ids[0], &ids[0] + 256, 0);
  cv::Mat idsMat(256, 1, CV_8UC1, ids);

  cv::Mat cmap;
  applyColorMap(idsMat, cmap, cv::COLORMAP_JET);

  for (int i = 0; i < numKeypoints; i++) {
    circle(out, cv::Point(kpts[i].x, kpts[i].y), 3, cv::Scalar(255, 0, 255),
           -1);
  }

  return out;
}

static cv::Mat DrawKeypoints(cv::Mat img, VPIKeypointF32 *kpts,
                             uint32_t *scores, int numKeypoints) {
  cv::Mat out;
  img.convertTo(out, CV_8UC1);
  cvtColor(out, out, cv::COLOR_GRAY2BGR);

  if (numKeypoints == 0) {
    return out;
  }

  // prepare our colormap
  cv::Mat cmap(1, 256, CV_8UC3);
  {
    cv::Mat gray(1, 256, CV_8UC1);
    for (int i = 0; i < 256; ++i) {
      gray.at<unsigned char>(0, i) = i;
    }
    applyColorMap(gray, cmap, cv::COLORMAP_HOT);
  }

  float maxScore = *std::max_element(scores, scores + numKeypoints);

  for (int i = 0; i < numKeypoints; ++i) {
    cv::Vec3b color = cmap.at<cv::Vec3b>(scores[i] / maxScore * 255);
    circle(out, cv::Point(kpts[i].x, kpts[i].y), 3,
           cv::Scalar(color[0], color[1], color[2]), -1);
  }

  return out;
}

static int UpdateMask(cv::Mat &debug_img, VPIArray prevFeatures,
                      VPIArray curFeatures, VPIArray track_status) {
  VPIArrayData curFeaturesData, statusData;
  CHECK_STATUS(vpiArrayLockData(curFeatures, VPI_LOCK_READ,
                                VPI_ARRAY_BUFFER_HOST_AOS, &curFeaturesData));
  CHECK_STATUS(vpiArrayLockData(track_status, VPI_LOCK_READ,
                                VPI_ARRAY_BUFFER_HOST_AOS, &statusData));

  const VPIArrayBufferAOS &aosCurFeatures = curFeaturesData.buffer.aos;
  const VPIArrayBufferAOS &aosStatus = statusData.buffer.aos;

  const VPIKeypointF32 *pCurFeatures = (VPIKeypointF32 *)aosCurFeatures.data;
  const uint8_t *pStatus = (uint8_t *)aosStatus.data;

  const VPIKeypointF32 *pPrevFeatures;
  if (prevFeatures) {
    VPIArrayData prevFeaturesData;
    CHECK_STATUS(vpiArrayLockData(prevFeatures, VPI_LOCK_READ,
                                  VPI_ARRAY_BUFFER_HOST_AOS,
                                  &prevFeaturesData));
    pPrevFeatures = (VPIKeypointF32 *)prevFeaturesData.buffer.aos.data;
  } else {
    pPrevFeatures = NULL;
  }

  int numTrackedKeypoints = 0;
  int totKeypoints = *curFeaturesData.buffer.aos.sizePointer;

  for (int i = 0; i < totKeypoints; i++) {
    cv::Point curPoint{(int)round(pCurFeatures[i].x),
                       (int)round(pCurFeatures[i].y)};
    // keypoint is being tracked?
    if (pStatus[i] == 0) {
      // draw the tracks

      if (pPrevFeatures != NULL) {
        cv::Point2f prevPoint{pPrevFeatures[i].x, pPrevFeatures[i].y};
        line(debug_img, prevPoint, curPoint, cv::Scalar(255, 0, 0), 2);
      }

      circle(debug_img, curPoint, 5, cv::Scalar(0, 0, 255), -1);

      numTrackedKeypoints++;
    } else {
      circle(debug_img, curPoint, 5, cv::Scalar(0, 255, 0), -1);
    }
  }

  if (prevFeatures) {
    CHECK_STATUS(vpiArrayUnlock(prevFeatures));
  }
  CHECK_STATUS(vpiArrayUnlock(curFeatures));
  CHECK_STATUS(vpiArrayUnlock(track_status));

  return numTrackedKeypoints;
}

VpiTracker::VpiTracker() {}

void VpiTracker::setParams(Params &params) {
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
    cout << "Opencv Mode." << endl;
  }

  setCameraIntrinsic(params_->config_file);
}

void VpiTracker::setCameraIntrinsic(const std::string &config) {
  camera_ =
      camodocal::CameraFactory::instance()->generateCameraFromYamlFile(config);
}

void VpiTracker::initVpiEqualize() {
  vpiStreamCreate(0, &equalize_stream_);
  vpiImageCreate(params_->col, params_->row, VPI_IMAGE_FORMAT_U8, 0,
                 &imgEqualize_);
  vpiCreateEqualizeHist(VPI_BACKEND_CUDA, VPI_IMAGE_FORMAT_U8, &equalize_);
}

void VpiTracker::initVpiOpticalFlow() {
  vpiContextCreate(0, &ctx_);
  vpiContextSetCurrent(ctx_);
  vpiEventCreate(0, &barrier_harris_);
  // Create the stream where processing will happen.
  CHECK_STATUS(vpiStreamCreate(0, &lk_stream_));
  CHECK_STATUS(vpiStreamCreate(0, &harris_stream_));
  // Create grayscale image representation of input.
  CHECK_STATUS(vpiImageCreate(params_->col, params_->row, VPI_IMAGE_FORMAT_U8,
                              0, &imgGrayscale_));
  CHECK_STATUS(vpiImageCreate(params_->col, params_->row, VPI_IMAGE_FORMAT_U8,
                              0, &imgGrayscale2_));
  // Create the image pyramids used by the algorithm
  CHECK_STATUS(vpiPyramidCreate(params_->col, params_->row, VPI_IMAGE_FORMAT_U8,
                                params_->pyrlevel, 0.5, 0, &pyrPrevFrame_));
  CHECK_STATUS(vpiPyramidCreate(params_->col, params_->row, VPI_IMAGE_FORMAT_U8,
                                params_->pyrlevel, 0.5, 0, &pyrCurFrame_));
  // Create input and output arrays
  CHECK_STATUS(vpiArrayCreate(MAX_HARRIS_CORNERS, VPI_ARRAY_TYPE_KEYPOINT_F32,
                              0, &prevFeatures_));
  // CHECK_STATUS(vpiArrayCreate(MAX_HARRIS_CORNERS,
  // VPI_ARRAY_TYPE_KEYPOINT_F32,
  //                            0, &curFeatures_));
  // CHECK_STATUS(
  //    vpiArrayCreate(MAX_HARRIS_CORNERS, VPI_ARRAY_TYPE_U8, 0, &status_));
  // CHECK_STATUS(
  //   vpiArrayCreate(MAX_HARRIS_CORNERS, VPI_ARRAY_TYPE_U32, 0, &scores_));
  // Parameters we'll use. No need to change them on the fly, so just define
  // them here. We're using the default parameters.
  CHECK_STATUS(vpiInitOpticalFlowPyrLKParams(&lkParams_));
  lkParams_.useInitialFlow = 0;
  lkParams_.numIterations = 32;
  lkParams_.windowDimension = 32;
  // Create Optical Flow payload
  CHECK_STATUS(vpiCreateOpticalFlowPyrLK(backend_, params_->col, params_->row,
                                         VPI_IMAGE_FORMAT_U8, params_->pyrlevel,
                                         0.5, &optflow_));
  CHECK_STATUS(vpiInitHarrisCornerDetectorParams(&harrisParams_));
  CHECK_STATUS(vpiCreateHarrisCornerDetector(VPI_BACKEND_CUDA, params_->col,
                                             params_->row, &harris_));
  // adjust some parameters
  harrisParams_.strengthThresh = 0;
  harrisParams_.sensitivity = 0.05;
  // harrisParams_.
}

void VpiTracker::initVpiOrb() {
  vpiInitORBParams(&orb_params_);
  orb_params_.fastParams.circleRadius = 3;
  orb_params_.fastParams.arcLength = 9;
  orb_params_.fastParams.intensityThreshold = 142;
  orb_params_.fastParams.nonMaxSuppression = 1;
  orb_params_.maxFeaturesPerLevel = 128;
  orb_params_.maxPyramidLevels = 8;
  int bufCapacity = orb_params_.maxFeaturesPerLevel * 20;
  vpiCreateORBFeatureDetector(VPI_BACKEND_CUDA, bufCapacity, &orb_);
  vpiStreamCreate(0, &orb_stream_);
  CHECK_STATUS(vpiPyramidCreate(params_->col, params_->row, VPI_IMAGE_FORMAT_U8,
                                params_->pyrlevel, 0.5, 0, &pyrCurFrame_));
  int outCapacity =
      orb_params_.maxFeaturesPerLevel * orb_params_.maxPyramidLevels;
  vpiArrayCreate(outCapacity, VPI_ARRAY_TYPE_KEYPOINT_F32, 0, &orb_corners_);
  vpiArrayCreate(outCapacity, VPI_ARRAY_TYPE_BRIEF_DESCRIPTOR, 0,
                 &orb_descriptors);
  // Create grayscale image representation of input.
  CHECK_STATUS(vpiImageCreate(params_->col, params_->row, VPI_IMAGE_FORMAT_U8,
                              0, &imgGrayscale_));
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

void VpiTracker::readImage(cv_bridge::CvImageConstPtr &ptr) {
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

void VpiTracker::processByVpiOptiflow(cv_bridge::CvImageConstPtr &ptr) {
  static bool first = true;
  cur_time = ptr->header.stamp.toSec();
  // cv::Mat debug_img = ptr->image.clone();
  // cv::cvtColor(debug_img, debug_img, CV_GRAY2RGB);
  if (first) {
    // warp cv_img to vpi_img
    vpiArrayCreate(MAX_HARRIS_CORNERS, VPI_ARRAY_TYPE_KEYPOINT_F32, 0,
                   &curFeatures_);
    vpiArrayCreate(MAX_HARRIS_CORNERS, VPI_ARRAY_TYPE_U32, 0, &scores_);
    vpiImageCreateWrapperOpenCVMat(ptr->image, 0, &imgInput_);
    vpiSubmitConvertImageFormat(lk_stream_, VPI_BACKEND_CUDA, imgInput_,
                                imgGrayscale_, NULL);
    vpiSubmitHarrisCornerDetector(lk_stream_, VPI_BACKEND_CUDA, harris_,
                                  imgGrayscale_, curFeatures_, scores_,
                                  &harrisParams_);
    vpiSubmitGaussianPyramidGenerator(lk_stream_, backend_, imgGrayscale_,
                                      pyrCurFrame_, VPI_BORDER_CLAMP);
    vpiStreamSync(lk_stream_);

    cv::Mat mask;
    n_pts = sortKeypoints(curFeatures_, scores_, mask, MAX_KEYPOINTS);

    std::swap(prevFeatures_, curFeatures_);
    std::swap(pyrPrevFrame_, pyrCurFrame_);
    vpiArrayDestroy(curFeatures_);
    vpiArrayDestroy(scores_);
    first = false;
    return;
  }
  vpiArrayCreate(MAX_HARRIS_CORNERS, VPI_ARRAY_TYPE_KEYPOINT_F32, 0,
                 &curFeatures_);
  vpiArrayCreate(MAX_HARRIS_CORNERS, VPI_ARRAY_TYPE_U8, 0, &status_);
  vpiArrayCreate(MAX_HARRIS_CORNERS, VPI_ARRAY_TYPE_U32, 0, &scores_);

  vpiImageSetWrappedOpenCVMat(imgInput_, ptr->image);
  vpiSubmitConvertImageFormat(lk_stream_, VPI_BACKEND_CUDA, imgInput_,
                              imgGrayscale_, NULL);
  vpiSubmitGaussianPyramidGenerator(lk_stream_, backend_, imgGrayscale_,
                                    pyrCurFrame_, VPI_BORDER_CLAMP);

  vpiSubmitOpticalFlowPyrLK(lk_stream_, 0, optflow_, pyrPrevFrame_,
                            pyrCurFrame_, prevFeatures_, curFeatures_, status_,
                            &lkParams_);
  vpiStreamSync(lk_stream_);

  // int cnt = UpdateMask(debug_img, prevFeatures_, curFeatures_, status_);
  // cout << "track count = " << cnt << endl;
  // cv::imshow("debug_img", debug_img);
  // cv::waitKey(1);

  forw_pts.clear();
  if (cur_pts.size() > 0) {
    vector<bool> status;
    convertData(status, forw_pts, curFeatures_);
    for (int i = 0; i < int(forw_pts.size()); i++) {
      if (status[i] && !inBorder(forw_pts[i])) {
        status[i] = 0;
      }
    }
    cout << "forw_pts 1: " << forw_pts.size() << endl;

    reduceVector(prev_pts, status);
    reduceVector(cur_pts, status);
    reduceVector(forw_pts, status);
    reduceVector(ids, status);
    reduceVector(cur_un_pts, status);
    reduceVector(track_cnt, status);
  }

  for (auto &n : track_cnt) {
    n++;
  }

  if (pub_this_frame_) {
    vpiSubmitHarrisCornerDetector(harris_stream_, VPI_BACKEND_CUDA, harris_,
                                  imgGrayscale_, curFeatures_, scores_,
                                  &harrisParams_);
    vpiStreamSync(harris_stream_);
    cv::Mat mask;
    rejectWithF();
    setMask(mask);
    int n_max_cnt = MAX_KEYPOINTS - static_cast<int>(forw_pts.size());
    if (n_max_cnt > 0) {
      n_pts = sortKeypoints(curFeatures_, scores_, mask, n_max_cnt);
      // cout << "n_pts count: " << n_pts.size() << endl;
    } else {
      n_pts.clear();
    }
    addPoints();
  }

  prev_pts = cur_pts;
  cur_pts = forw_pts;
  undistortedPoints();
  prev_time = cur_time;

  std::swap(prevFeatures_, curFeatures_);
  std::swap(pyrPrevFrame_, pyrCurFrame_);

  vpiArrayDestroy(status_);
  vpiArrayDestroy(scores_);
  vpiArrayDestroy(curFeatures_);
}

void VpiTracker::processByVpiFast(cv_bridge::CvImageConstPtr &ptr) {
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

  for (auto &n : track_cnt) {
    n++;
  }

  if (pub_this_frame_) {
    cv::Mat mask;
    rejectWithF();
    setMask(mask);
    int n_max_cnt = MAX_KEYPOINTS - static_cast<int>(forw_pts.size());
    if (n_max_cnt > 0) {
      n_pts = sortKeypoints(fast_corners_, n_max_cnt, mask, debug_img);
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

void VpiTracker::processByOpencv(cv_bridge::CvImageConstPtr &ptr) {
  cv::Mat img = ptr->image;
  cur_time = ptr->header.stamp.toSec();
  if (forw_img_.empty()) {
    prev_img_ = cur_img_ = forw_img_ = img;
  } else {
    forw_img_ = img;
  }

  forw_pts.clear();

  if (cur_pts.size() > 0) {
    vector<uchar> status;
    vector<float> err;
    cv::calcOpticalFlowPyrLK(cur_img_, forw_img_, cur_pts, forw_pts, status,
                             err, cv::Size(21, 21), 3);

    for (int i = 0; i < int(forw_pts.size()); i++)
      if (status[i] && !inBorder(forw_pts[i])) status[i] = 0;

    //cout << "forw_pts: " << forw_pts.size() << endl;
    reduceVector(prev_pts, status);
    reduceVector(cur_pts, status);
    reduceVector(forw_pts, status);
    reduceVector(ids, status);
    reduceVector(cur_un_pts, status);
    reduceVector(track_cnt, status);
  }

  for (auto &n : track_cnt) n++;

  if (pub_this_frame_) {
    cv::Mat mask;
    rejectWithF();
    setMask(mask);
    int n_max_cnt = MAX_KEYPOINTS - static_cast<int>(forw_pts.size());
    if (n_max_cnt > 0) {
      if (mask.empty()) cout << "mask is empty " << endl;
      if (mask.type() != CV_8UC1) cout << "mask type wrong " << endl;
      if (mask.size() != forw_img_.size()) cout << "wrong size " << endl;
      cv::goodFeaturesToTrack(forw_img_, n_pts, MAX_KEYPOINTS - forw_pts.size(),
                              0.01, params_->min_dist, mask);
    } else
      n_pts.clear();

    addPoints();
  }

  prev_img_ = cur_img_;
  prev_pts = cur_pts;
  prev_un_pts = cur_un_pts;
  cur_img_ = forw_img_;
  cur_pts = forw_pts;
  undistortedPoints();
  prev_time = cur_time;
}

void VpiTracker::setMask(cv::Mat &mask) {
  mask = cv::Mat(params_->row, params_->col, CV_8UC1, cv::Scalar(255));

  // prefer to keep features that are tracked for long time
  vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

  for (unsigned int i = 0; i < forw_pts.size(); i++)
    cnt_pts_id.push_back(
        make_pair(track_cnt[i], make_pair(forw_pts[i], ids[i])));

  sort(cnt_pts_id.begin(), cnt_pts_id.end(),
       [](const pair<int, pair<cv::Point2f, int>> &a,
          const pair<int, pair<cv::Point2f, int>> &b) {
         return a.first > b.first;
       });

  forw_pts.clear();
  ids.clear();
  track_cnt.clear();

  for (auto &it : cnt_pts_id) {
    if (mask.at<uchar>(it.second.first) == 255) {
      forw_pts.push_back(it.second.first);
      ids.push_back(it.second.second);
      track_cnt.push_back(it.first);
      cv::circle(mask, it.second.first, params_->min_dist, 0, -1);
    }
  }
}

void VpiTracker::addPoints() {
  for (auto &p : n_pts) {
    forw_pts.push_back(p);
    ids.push_back(-1);
    track_cnt.push_back(1);
  }
}

bool VpiTracker::inBorder(const cv::Point2f &pt) {
  const int BORDER_SIZE = 1;
  int img_x = cvRound(pt.x);
  int img_y = cvRound(pt.y);
  return BORDER_SIZE <= img_x && img_x < params_->col - BORDER_SIZE &&
         BORDER_SIZE <= img_y && img_y < params_->row - BORDER_SIZE;
}

void VpiTracker::convertData(std::vector<bool> &track_status,
                             std::vector<cv::Point2f> &forw_pts,
                             VPIArray curFeatures) {
  // Lock the input and output arrays to draw the tracks to the output mask.
  VPIArrayData curFeaturesData, statusData;
  CHECK_STATUS(vpiArrayLockData(curFeatures, VPI_LOCK_READ,
                                VPI_ARRAY_BUFFER_HOST_AOS, &curFeaturesData));
  CHECK_STATUS(vpiArrayLockData(status_, VPI_LOCK_READ,
                                VPI_ARRAY_BUFFER_HOST_AOS, &statusData));

  const VPIArrayBufferAOS &aosCurFeatures = curFeaturesData.buffer.aos;
  const VPIArrayBufferAOS &aosStatus = statusData.buffer.aos;

  const VPIKeypointF32 *pCurFeatures = (VPIKeypointF32 *)aosCurFeatures.data;
  const uint8_t *pStatus = (uint8_t *)aosStatus.data;

  int numTrackedKeypoints = 0;
  int totKeypoints = *curFeaturesData.buffer.aos.sizePointer;

  forw_pts.resize(totKeypoints);
  track_status.resize(totKeypoints);
  for (int i = 0; i < totKeypoints; ++i) {
    forw_pts[i] = cv::Point2f(pCurFeatures[i].x, pCurFeatures[i].y);
    if (pStatus[i] == 0) {
      track_status[i] = true;
      numTrackedKeypoints++;
    } else {
      track_status[i] = false;
    }
  }

  CHECK_STATUS(vpiArrayUnlock(curFeatures));
  CHECK_STATUS(vpiArrayUnlock(status_));
}

template <typename T1, typename T2>
void VpiTracker::reduceVector(std::vector<T1> &v, const std::vector<T2> &s) {
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

std::vector<cv::Point2f> VpiTracker::sortKeypoints(VPIArray keypoints, int max,
                                                   cv::Mat &mask,
                                                   cv::Mat &debug_img) {
  VPIArrayData ptsData;
  vpiArrayLockData(keypoints, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS,
                   &ptsData);
  VPIArrayBufferAOS &aosKeypoints = ptsData.buffer.aos;

  std::vector<cv::Point2f> kpt(*aosKeypoints.sizePointer);
  VPIKeypointF32 *kptData =
      reinterpret_cast<VPIKeypointF32 *>(aosKeypoints.data);
  transform(kptData, kptData + (*aosKeypoints.sizePointer), back_inserter(kpt),
            [](VPIKeypointF32 p) { return cv::Point2f(p.x, p.y); });

  vpiArrayUnlock(keypoints);

  std::vector<cv::Point2f> pts;
  for (int i = 0; i < kpt.size(); ++i) {
    if (mask.at<uchar>(kpt[i]) != 0) {
      pts.emplace_back(kpt[i]);
      if (pts.size() > max) {
        break;
      }
    }
  }

  return pts;
}

vector<cv::Point2f> VpiTracker::sortKeypoints(VPIArray keypoints,
                                              VPIArray scores, cv::Mat &mask,
                                              int max) {
  VPIArrayData ptsData, scoresData;
  CHECK_STATUS(vpiArrayLockData(keypoints, VPI_LOCK_READ_WRITE,
                                VPI_ARRAY_BUFFER_HOST_AOS, &ptsData));
  CHECK_STATUS(vpiArrayLockData(scores, VPI_LOCK_READ_WRITE,
                                VPI_ARRAY_BUFFER_HOST_AOS, &scoresData));

  VPIArrayBufferAOS &aosKeypoints = ptsData.buffer.aos;
  VPIArrayBufferAOS &aosScores = scoresData.buffer.aos;

  std::vector<int> indices(*aosKeypoints.sizePointer);
  std::iota(indices.begin(), indices.end(), 0);

  stable_sort(indices.begin(), indices.end(), [&aosScores](int a, int b) {
    uint32_t *score = reinterpret_cast<uint32_t *>(aosScores.data);
    return score[a] >= score[b];  // decreasing score order
  });

  // keep the only 'max' indexes.
  indices.resize(std::min<size_t>(indices.size(), max));

  VPIKeypointF32 *kptData =
      reinterpret_cast<VPIKeypointF32 *>(aosKeypoints.data);

  // reorder the keypoints to keep the first 'max' with highest scores.
  std::vector<VPIKeypointF32> kpt;
  std::transform(indices.begin(), indices.end(), std::back_inserter(kpt),
                 [kptData](int idx) { return kptData[idx]; });
  std::copy(kpt.begin(), kpt.end(), kptData);

  vector<cv::Point2f> pts;
  for (int i = 0; i < kpt.size() && !mask.empty(); ++i) {
    cv::Point2f p(kpt[i].x, kpt[i].y);
    if (mask.at<uchar>(p) != 0) {
      pts.emplace_back(p);
    }
  }

  // update keypoint array size.
  *aosKeypoints.sizePointer = kpt.size();
  // cout << "aosKeypoints.sizePointer: " << *aosKeypoints.sizePointer << endl;

  vpiArrayUnlock(scores);
  vpiArrayUnlock(keypoints);

  // cout << "key points size: " << kpt.size() << endl;

  return pts;
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
  // cv::undistortPoints(cur_pts, un_pts, K, cv::Mat());
  for (unsigned int i = 0; i < cur_pts.size(); i++) {
    Eigen::Vector2d a(cur_pts[i].x, cur_pts[i].y);
    Eigen::Vector3d b;
    camera_->liftProjective(a, b);
    cur_un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
    cur_un_pts_map.insert(
        make_pair(ids[i], cv::Point2f(b.x() / b.z(), b.y() / b.z())));
    // printf("cur pts id %d %f %f", ids[i], cur_un_pts[i].x, cur_un_pts[i].y);
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