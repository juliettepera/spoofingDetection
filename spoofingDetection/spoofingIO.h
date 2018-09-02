#include <stdio.h>
#include <string.h>
#include <exception>

#include <opencv2/opencv.hpp>
namespace io
{
void displayImage(const cv::Mat& image, const std::string& imageName = "")
{
    cv::namedWindow(imageName, cv::WINDOW_AUTOSIZE);
    cv::imshow(imageName, image);

    cv::waitKey(0);
}

void readImage(const std::string& inputPath, cv::Mat& inputImage)
{
    inputImage = cv::imread( inputPath, 1 );

    if ( !inputImage.data )
    {
        throw std::runtime_error("Failed to read the input image");
    }
}

bool fileExists(const std::string& fileName)
{
    std::ifstream infile(fileName);
    return infile.good();
}
}
