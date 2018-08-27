#include <stdio.h>
#include <iostream>
#include <string.h>
#include <exception>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

void displayImage(const cv::Mat& image, const std::string& imageName = "")
{
    cv::namedWindow(imageName, cv::WINDOW_AUTOSIZE);
    cv::imshow(imageName, image);

    cv::waitKey(0);
}

void applyGabor(const cv::Mat& inputImage, cv::Mat& filteredImage)
{
    int kernel_size = 31;
    double sig = 1;
    double th = 0;
    double lm = 1.0;
    double gm = 0.02;
    double ps = 0;

    cv::Mat kernel = cv::getGaborKernel(cv::Size(kernel_size,kernel_size), sig, th, lm, gm, ps);
    cv::filter2D(inputImage, filteredImage, CV_32F, kernel);

    filteredImage.convertTo(filteredImage,CV_8U,1.0/255.0);

    displayImage(kernel, "kernel");
}

void extractGradient(const cv::Mat& inputImage, cv::Mat& scaledGradX, const int dimension, const int kvalue = 3)
{
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;

    const bool dx = (dimension == 0) ? true : false;
    const bool dy = (dimension == 0) ? false : true;

    cv::Mat gradX;
    cv::Sobel(inputImage, gradX, ddepth, dx, dy, kvalue, scale, delta, cv::BORDER_DEFAULT);

    if ( !gradX.data )
    {
        throw std::runtime_error("Failed to extract the gradient");
    }

    cv::convertScaleAbs(gradX, scaledGradX);

    if ( !scaledGradX.data )
    {
        throw std::runtime_error("Failed to sclaed the gradient image");
    }
}

void convertToGray(const cv::Mat& inputImage, cv::Mat& grayImage)
{
    cv::cvtColor(inputImage, grayImage, CV_BGR2GRAY);

    if ( !grayImage.data )
    {
        throw std::runtime_error("Failed to convert the image to gray");
    }
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

void parseArgs(int argc, char** argv, std::string& inputPath)
{
    if ( argc != 2 )
    {
        throw std::runtime_error("Missing arguments to execute the program");
    }

    inputPath = argv[1];

    if (!fileExists(inputPath))
    {
        throw std::runtime_error("Image file doesn't exists");
    }
}

int main(int argc, char** argv)
{
    try
    {
        std::string inputPath;
        parseArgs(argc, argv, inputPath);

        cv::Mat inputImage;
        readImage(inputPath, inputImage);

        displayImage(inputImage, "original image");

        cv::Mat grayImage;
        convertToGray(inputImage, grayImage);

        displayImage(grayImage, "grayscaled image");

        cv::Mat gradientX;
        extractGradient(grayImage, gradientX, 0);

        displayImage(gradientX, "gradient X image");

        cv::Mat gradientY;
        extractGradient(grayImage, gradientY, 1);

        displayImage(gradientY, "gradient Y image");

        cv::Mat gradientImage = gradientX + gradientY;

        displayImage(gradientImage, "gradient image");

        cv::Mat filteredImage;
        applyGabor(inputImage, filteredImage);

        displayImage(filteredImage, "filtered image");

        return 0;
    }
    catch (std::runtime_error err)
    {
        std::cout << err.what() << std::endl;
        return -1;
    }
}
