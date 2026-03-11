#include <opencv2/opencv.hpp>
using namespace cv;

int main()
{
    Mat image(200,200,CV_8U);
   for(int i=0; i<image.rows; i++)
   {
       for(int j=0; j<image.cols; j++)
       {
           image.at<unsigned char>(i,j) = 0;
    }
    namedWindow("Display Image", WINDOW_AUTOSIZE);
    imshow("Display Image", image);
    waitKey(0);
    return 0;
}