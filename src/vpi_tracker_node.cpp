#include <cv_bridge/cv_bridge.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud.h>
#include <std_msgs/Bool.h>

#include <string>

#include "vpi_tracker/utils.h"
#include "vpi_tracker/vpi_tracker.h"

using namespace std;

class VpiTrackerNode
{
public:
    VpiTrackerNode() : node_handle_("~")
    {
        readParameters(node_handle_, params_);
        tracker_.setParams(params_);
        sub_img_ = node_handle_.subscribe<sensor_msgs::Image>(
            params_.image_topic_, 1, &VpiTrackerNode::imgCallback, this);
        pub_img_ = node_handle_.advertise<sensor_msgs::PointCloud>("feature", 1000);
        pub_match_ =
            node_handle_.advertise<sensor_msgs::Image>("feature_img", 1000);
        pub_restart_ = node_handle_.advertise<std_msgs::Bool>("restart", 1000);
    }

private:
    void imgCallback(const sensor_msgs::Image::ConstPtr& img_msg)
    {
        if (first_image_flag_)
        {
            first_image_flag_ = false;
            first_image_time_ = img_msg->header.stamp.toSec();
            last_image_time_ = img_msg->header.stamp.toSec();
            return;
        }

        // check camera timestamp stable
        if (img_msg->header.stamp.toSec() - last_image_time_ > 1.0 ||
            img_msg->header.stamp.toSec() < last_image_time_)
        {
            ROS_WARN("image discontinue! reset the feature tracker!");
            first_image_flag_ = true;
            last_image_time_ = 0;
            pub_count_ = 1;
            std_msgs::Bool restart_flag;
            restart_flag.data = true;
            pub_restart_.publish(restart_flag);
            return;
        }

        last_image_time_ = img_msg->header.stamp.toSec();
        // frequency control
        if (round(1.0 * pub_count_ /
                  (img_msg->header.stamp.toSec() - first_image_time_)) <=
            params_.freq)
        {
            tracker_.pub_this_frame = true;
            // reset the frequency control
            if (abs(1.0 * pub_count_ /
                        (img_msg->header.stamp.toSec() - first_image_time_) -
                    params_.freq) < 0.01 * params_.freq)
            {
                first_image_time_ = img_msg->header.stamp.toSec();
                pub_count_ = 0;
            }
        }
        else
        {
            tracker_.pub_this_frame = false;
        }

        cv_bridge::CvImageConstPtr ptr;
        if (img_msg->encoding == "8UC1")
        {
            sensor_msgs::Image img;
            img.header = img_msg->header;
            img.height = img_msg->height;
            img.width = img_msg->width;
            img.is_bigendian = img_msg->is_bigendian;
            img.step = img_msg->step;
            img.data = img_msg->data;
            img.encoding = "mono8";
            ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
        }
        else
            ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

        tracker_.readImage(ptr);

        for (unsigned int i = 0;; i++)
        {
            if (tracker_.updateID(i) == false)
            {
                break;
            }
        }

        if (tracker_.pub_this_frame)
        {
            pub_count_++;
            sensor_msgs::PointCloudPtr feature_points(new sensor_msgs::PointCloud);
            sensor_msgs::ChannelFloat32 id_of_point;
            sensor_msgs::ChannelFloat32 u_of_point;
            sensor_msgs::ChannelFloat32 v_of_point;
            sensor_msgs::ChannelFloat32 velocity_x_of_point;
            sensor_msgs::ChannelFloat32 velocity_y_of_point;

            feature_points->header = img_msg->header;
            feature_points->header.frame_id = "world";

            vector<cv::Point2f>& pts_velocity = tracker_.pts_velocity;
            vector<cv::Point2f>& un_pts = tracker_.cur_un_pts;
            vector<cv::Point2f>& cur_pts = tracker_.cur_pts;
            vector<int>& ids = tracker_.ids;

            for (int i = 0; i < ids.size(); ++i)
            {
                if (tracker_.track_cnt[i] > 1)
                {
                    int p_id = ids[i];
                    geometry_msgs::Point32 p;
                    p.x = un_pts[i].x;
                    p.y = un_pts[i].y;
                    p.z = 1;
                    feature_points->points.push_back(p);
                    id_of_point.values.push_back(p_id);
                    u_of_point.values.push_back(cur_pts[i].x);
                    v_of_point.values.push_back(cur_pts[i].y);
                    velocity_x_of_point.values.push_back(pts_velocity[i].x);
                    velocity_y_of_point.values.push_back(pts_velocity[i].y);
                }
            }

            feature_points->channels.push_back(id_of_point);
            feature_points->channels.push_back(u_of_point);
            feature_points->channels.push_back(v_of_point);
            feature_points->channels.push_back(velocity_x_of_point);
            feature_points->channels.push_back(velocity_y_of_point);
            ROS_DEBUG("publish %f, at %f", feature_points->header.stamp.toSec(),
                      ros::Time::now().toSec());

            static int init_pub = 0;
            if (!init_pub)
            {
                init_pub = 1;
            }
            else
            {
                pub_img_.publish(feature_points);
            }

            if (params_.show_track)
            {
                ptr = cv_bridge::cvtColor(ptr, sensor_msgs::image_encodings::BGR8);
                cv::Mat tmp_img = ptr->image;
                // cv::cvtColor(tmp_img, tmp_img, CV_GRAY2RGB);
                for (int i = 0; i < tracker_.cur_pts.size(); ++i)
                {
                    double len = std::min(1.0, 1.0 * tracker_.track_cnt[i] / 10);
                    cv::circle(tmp_img, tracker_.cur_pts[i], 2,
                               cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
                }
                // cv::imshow("show track", tmp_img);
                // cv::waitKey(1);
                pub_match_.publish(ptr->toImageMsg());
            }
        }
    }

    Params params_;
    VpiTracker tracker_;
    ros::NodeHandle node_handle_;
    ros::Subscriber sub_img_;
    ros::Publisher pub_img_;
    ros::Publisher pub_match_;
    ros::Publisher pub_restart_;

    double first_image_time_;
    double last_image_time_;
    bool first_image_flag_ = true;
    int pub_count_ = 1;
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "vpi_tracker");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME,
                                   ros::console::levels::Debug);
    VpiTrackerNode node;
    while (ros::ok())
    {
        ros::spin();
    }

    return 0;
}
