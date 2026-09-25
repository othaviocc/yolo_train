#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/string.hpp>
#include <string>

// PCL Headers
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/common/common.h>

class GestureDetector : public rclcpp::Node {
public:
    GestureDetector() : Node("gesture_detector_node") {
        // ROS 2 Subscribers and Publishers
        cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/livox/lidar", 10, std::bind(&GestureDetector::cloud_callback, this, std::placeholders::_1));

        cmd_pub_ = this->create_publisher<std_msgs::msg::String>("/fbot/mission_control/gesture_events", 10);
        
        RCLCPP_INFO(this->get_logger(), "3D Gesture Detection Node Started. Waiting for human operator...");
    }

private:
    std::string last_published_cmd_ = "";
    int debounce_counter_ = 0; // Para garantir que o gesto é real e não um ruído rápido

    void cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::fromROSMsg(*msg, *cloud);

        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZI>);
        
        // Isolate X axis (Depth): Look only between 1.0m and 4.0m in front of the drone
        pcl::PassThrough<pcl::PointXYZI> pass;
        pass.setInputCloud(cloud);
        pass.setFilterFieldName("x");
        pass.setFilterLimits(1.0, 4.0); 
        pass.filter(*cloud_filtered);

        // Isolate Z axis (Height): Remove floor and ceiling
        pass.setInputCloud(cloud_filtered);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(-0.5, 2.3); 
        pass.filter(*cloud_filtered);

        if (cloud_filtered->empty()) return;

        pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>);
        tree->setInputCloud(cloud_filtered);
        
        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;
        ec.setClusterTolerance(0.15); // Points within 15cm belong to the human
        ec.setMinClusterSize(150);    // Filter out small floating noise
        ec.setMaxClusterSize(25000);  
        ec.setSearchMethod(tree);
        ec.setInputCloud(cloud_filtered);
        ec.extract(cluster_indices);

        if (cluster_indices.empty()) return;

        // Take the largest cluster (The operator)
        pcl::PointCloud<pcl::PointXYZI>::Ptr human_cluster(new pcl::PointCloud<pcl::PointXYZI>);
        for (const auto& idx : cluster_indices[0].indices) {
            human_cluster->push_back((*cloud_filtered)[idx]);
        }

        pcl::PointXYZI min_pt, max_pt;
        pcl::getMinMax3D(*human_cluster, min_pt, max_pt);

        float centroid_y = (max_pt.y + min_pt.y) / 2.0;
        
        float height = max_pt.z - min_pt.z; // Z Axis
        float depth = max_pt.x - min_pt.x;  // X Axis
        
        float left_arm_extension = max_pt.y - centroid_y;  // Y Axis (Left)
        float right_arm_extension = centroid_y - min_pt.y; // Y Axis (Right)

        std_msgs::msg::String cmd_msg;
        
        // GESTURE: LAND (Squatting - Height drops significantly)
        if (height < 1.10) { 
            cmd_msg.data = "LAND";
        } 
        // GESTURE: MOVE FORWARD (Both arms straight up - Height increases significantly)
        else if (height > 2.00) {
            cmd_msg.data = "MOVE_FORWARD";
        }
        // GESTURE: MOVE BACKWARD (Both arms pointing at the drone - Depth increases)
        else if (depth > 0.65) {
            cmd_msg.data = "MOVE_BACKWARD";
        }
        // GESTURE: MOVE LEFT (Operator's Right Arm extended sideways)
        else if (left_arm_extension > 0.50 && right_arm_extension < 0.35) { 
            cmd_msg.data = "MOVE_LEFT";
        } 
        // GESTURE: MOVE RIGHT (Operator's Left Arm extended sideways)
        else if (right_arm_extension > 0.50 && left_arm_extension < 0.35) { 
            cmd_msg.data = "MOVE_RIGHT";
        }
        // IDLE STATE (Standing normally)
        else {
            cmd_msg.data = "HOLD"; 
        }

        // Only trigger if the command changes and stabilizes (avoids jitter)
        if (cmd_msg.data != last_published_cmd_) {
            debounce_counter_++;
            if (debounce_counter_ > 5) { // Needs 5 consecutive consistent frames to confirm
                if (cmd_msg.data != "HOLD") {
                    // Terminal print for the referee
                    RCLCPP_INFO(this->get_logger(), ">>> VISUAL COMMAND RECOGNIZED: [%s] <<<", cmd_msg.data.c_str());
                    RCLCPP_INFO(this->get_logger(), "    [Metrics] Height: %.2fm | Depth: %.2fm | Y-Left: %.2fm | Y-Right: %.2fm", 
                                height, depth, left_arm_extension, right_arm_extension);
                }
                last_published_cmd_ = cmd_msg.data;
                cmd_pub_->publish(cmd_msg); // Send String to Mission Control
                debounce_counter_ = 0;
            }
        } else {
            debounce_counter_ = 0;
        }
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr cmd_pub_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<GestureDetector>());
    rclcpp::shutdown();
    return 0;
}