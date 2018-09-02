#include "spoofingIO.h"
#include "spoofingUtils.h"

#include <opencv2/opencv.hpp>

#include <iostream>
#include <string.h>
#include <exception>

void parseArgs(int argc, char** argv, std::string& inputPath)
{
    if ( argc != 2 )
    {
        throw std::runtime_error("Missing arguments to execute the program");
    }

    inputPath = argv[1];

    if (!io::fileExists(inputPath))
    {
        throw std::runtime_error("Image file doesn't exists");
    }
}

int main(int argc, char** argv)
{
    try
    {
        const bool display = true;

        std::string inputPath;
        parseArgs(argc, argv, inputPath);

        // Read the input image
        cv::Mat inputImage;
        io::readImage(inputPath, inputImage);
        if(display)
        {
            io::displayImage(inputImage, "input image");
        }

        // Convert it to gray
        cv::Mat grayImage;
        util::convertToGray(inputImage, grayImage);
        if(display)
        {
            io::displayImage(grayImage, "grayscaled image");
        }

        // Run LBP on gray image
        cv::Mat lpbGray;
        util::runLBPOnImage(grayImage, 10, lpbGray);
        if(display)
        {
            io::displayImage(lpbGray, "LBP scores on gray image");
        }

        // Threshold the image
        cv::Mat thresh(grayImage.rows, grayImage.cols, CV_8U, cv::Scalar(0));
        const size_t count = 15;
        const size_t step = 255 / count;
        for (int i = 1 ; i < count-1 ; ++i)
        {
            util::thresholdImage(grayImage, i*step, thresh, i*step);
        }
        if(display)
        {
            io::displayImage(thresh, "thresholded image");
        }

        // Run LBP on gray image
        cv::Mat lpbThresh;
        util::runLBPOnImage(thresh, 10, lpbThresh);
        if(display)
        {
            io::displayImage(lpbThresh, "LBP scores on thresholded image");
        }

        // Extract the X gradient
        cv::Mat grad;
        util::extractGradientX(grayImage, grad);
        if(display)
        {
            io::displayImage(grad, "gradient X image");
        }

        // Run LBP on gradient image
        cv::Mat lpbGrad;
        util::runLBPOnImage(grad, 10, lpbGrad);
        if(display)
        {
            io::displayImage(lpbGrad, "LBP scores gradient X image");
        }

//        // Compute the histogram
//        cv::Mat histo;
//        util::createHistogram(LBP, histo);
//        if(display)
//        {
//            io::displayImage(histo, "histogram LPB");
//        }

        return 0;
    }
    catch (std::runtime_error err)
    {
        std::cerr << err.what() << std::endl;
        return -1;
    }
}

///\todo : check matrix type on algo
///\todo : add documentation
///\todo : check covariance matrix
///\todo : check histogram comparison
