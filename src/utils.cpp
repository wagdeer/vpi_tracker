#include "vpi_tracker/utils.h"

#include <string>

using namespace std;

template <typename T>
T readParam(ros::NodeHandle& n, string name)
{
    T ans;
    if (n.getParam(name, ans))
    {
        ROS_INFO_STREAM("Loaded " << name << ": " << ans);
    }
    else
    {
        ROS_ERROR_STREAM("Failed to load " << name);
        n.shutdown();
    }
    return ans;
}

void readParameters(ros::NodeHandle& n, Params& params)
{
    string config_file;
    config_file = readParam<string>(n, "config_file");
    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if (!fsSettings.isOpened())
    {
        cerr << "ERROR: Wrong path to settings" << endl;
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
    if (params.freq <= 0)
    {
        params.freq = 100;
    }
    fsSettings.release();
}

void showParameters(const Params& params)
{
    printf("Parameters:\n");
    printf("\timage topic: %s\n", params.image_topic_.c_str());
    printf("\timage_height: %d\n", params.row);
    printf("\timage_width: %d\n\n", params.col);
}

void download(const cv::cuda::GpuMat& src, vector<cv::Point2f>& dst)
{
    dst.resize(src.cols);
    cv::Mat mat(1, src.cols, CV_32FC2, (void*)&dst[0]);
    src.download(mat);
}

void download(const cv::cuda::GpuMat& src, vector<uchar>& dst)
{
    dst.resize(src.cols);
    cv::Mat mat(1, src.cols, CV_8UC1, (void*)&dst[0]);
    src.download(mat);
}

void download(const VPIArray src, vector<cv::Point2f>& dst)
{
    VPIArrayData srcData;
    vpiArrayLockData(src, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS, &srcData);
    const VPIArrayBufferAOS& aosSrcData = srcData.buffer.aos;
    const VPIKeypointF32* pKpt = (VPIKeypointF32*)aosSrcData.data;
    int len = *srcData.buffer.aos.sizePointer;

    dst.resize(len);
    for (int i = 0; i < len; ++i)
    {
        dst[i].x = pKpt[i].x;
        dst[i].y = pKpt[i].y;
    }
    vpiArrayUnlock(src);
}

void download(const VPIArray src, vector<uchar>& dst)
{
    VPIArrayData srcData;
    vpiArrayLockData(src, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS, &srcData);
    const VPIArrayBufferAOS& aosSrcData = srcData.buffer.aos;
    const uint8_t* pSts = (uint8_t*)aosSrcData.data;
    int len = *srcData.buffer.aos.sizePointer;

    dst.resize(len);
    for (int i = 0; i < len; ++i)
    {
        dst[i] = !pSts[i];
    }
    vpiArrayUnlock(src);
}

void upload(cv::cuda::GpuMat& src, vector<cv::Point2f>& dst)
{
    cv::Mat mat(dst);
    src.upload(mat);
}

void upload(const VPIArray src, vector<cv::Point2f>& dst)
{
    VPIArrayData ptsData;
    vpiArrayLockData(src, VPI_LOCK_READ_WRITE, VPI_ARRAY_BUFFER_HOST_AOS,
                     &ptsData);
    VPIArrayBufferAOS& aosKeypoints = ptsData.buffer.aos;
    VPIKeypointF32* kptData =
        reinterpret_cast<VPIKeypointF32*>(aosKeypoints.data);
    for (int i = 0; i < dst.size(); ++i)
    {
        kptData[i].x = dst[i].x;
        kptData[i].y = dst[i].y;
    }
    *aosKeypoints.sizePointer = dst.size();
    vpiArrayUnlock(src);
}