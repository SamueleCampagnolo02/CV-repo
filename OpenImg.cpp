#include <opencv2/highgui.hpp>


int main(int argc, char** argv)
{
    cv::Mat image = cv::imread(argv[1]);   
    cv::namedWindow("Display Image");
    cv::imshow("Display Image", image);
    cv::waitKey(0);
    return 0;


}