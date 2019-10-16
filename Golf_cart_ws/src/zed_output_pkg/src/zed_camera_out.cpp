#include <iostream>
#include <ros/ros.h>
#include <sl_zed/Camera.hpp>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <time.h>
#include <stdio.h>
#include <string>

using namespace std;
using namespace sl;
using namespace cv;

// Get current date/time, format is YYYY-MM-DD.HH:mm:ss
const std::string currentDateTime() {
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    // Visit http://en.cppreference.com/w/cpp/chrono/c/strftime
    // for more information about date/time format
    strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);

    return buf;
}


static const std::string OPENCV_WINDOW = "Image window";
//ros::Subscriber camera_sub;

class ImageConverter
{
    ros::NodeHandle n;
    image_transport::ImageTransport it_;
    image_transport::Subscriber camera_sub;

public:
  ImageConverter()
    : it_(n)
  {
    
    camera_sub = it_.subscribe("/zed_node/rgb/image_rect_color",  1, &ImageConverter::camera_out, this);

    //cv::namedWindow(OPENCV_WINDOW);
  }

  ~ImageConverter()
  {
    //cv::destroyWindow(OPENCV_WINDOW);
	}

void camera_out(const sensor_msgs::ImageConstPtr& cam_out)
{
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvCopy(cam_out, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

	string path;
	path = "./cam_data/test" + currentDateTime() + ".jpg";
	cout << path;

    //imshow(OPENCV_WINDOW, cv_ptr->image);
    imwrite(path, cv_ptr->image);
}
  
};


int main(int argc, char **argv)
{
  ROS_INFO("hello");
    ros::init(argc, argv, "camera_output");
    ImageConverter a;
    ros::spin();

  return 0;
}














/*
int main(int argc, char **argv) {

Camera zed;


cv::Mat slMat2cvMat(Mat& input);

// Create an RGBA sl::Mat object

    Mat image_zed(zed.getResolution(), MAT_TYPE_8U_C4);
// Create an OpenCV Mat that shares sl::Mat data
    cv::Mat image_ocv = slMat2cvMat(image_zed);


    if (zed.grab() == SUCCESS) {
// Retrieve the left image in sl::Mat
// The cv::Mat is automatically updated
        zed.retrieveImage(image_zed, VIEW_LEFT);
// Display the left image from the cv::Mat object
        cv::imshow("Image", image_ocv);
    }


// Create a sl::Mat with float type (32-bit)
    Mat depth_zed(zed.getResolution(), MAT_TYPE_32F_C1);
// Create an OpenCV Mat that shares sl::Mat data
    cv::Mat depth_ocv = slMat2cvMat(depth_zed);

    if (zed.grab() == SUCCESS) {
// Retrieve the depth measure (32-bit)
        zed.retrieveMeasure(depth_zed, MEASURE_DEPTH);
// Print the depth value at the center of the image


	
        std::cout << depth_ocv.at<float>(depth_ocv.rows / 2, depth_ocv.cols / 2) << std::endl;
    }


    cv::waitKey(0);
    return 0;
}

cv::Mat slMat2cvMat(Mat& input) {
    // Mapping between MAT_TYPE and CV_TYPE
    int cv_type = -1;
    switch (input.getDataType()) {
        case MAT_TYPE_32F_C1: cv_type = CV_32FC1; break;
        case MAT_TYPE_32F_C2: cv_type = CV_32FC2; break;
        case MAT_TYPE_32F_C3: cv_type = CV_32FC3; break;
        case MAT_TYPE_32F_C4: cv_type = CV_32FC4; break;
        case MAT_TYPE_8U_C1: cv_type = CV_8UC1; break;
        case MAT_TYPE_8U_C2: cv_type = CV_8UC2; break;
        case MAT_TYPE_8U_C3: cv_type = CV_8UC3; break;
        case MAT_TYPE_8U_C4: cv_type = CV_8UC4; break;
        default: break;
    }

    // Since cv::Mat data requires a uchar* pointer, we get the uchar1 pointer from sl::Mat (getPtr<T>())
    // cv::Mat and sl::Mat will share a single memory structure
    return cv::Mat(input.getHeight(), input.getWidth(), cv_type, input.getPtr<sl::uchar1>(MEM_CPU));
}
*/
