#include "spoofingIO.h"
#include "spoofingUtils.h"

#include <opencv2/opencv.hpp>

#include <iostream>
#include <string.h>
#include <exception>

// read the fake and true image, try different operation on the images and display the resuts
void testing(const std::string& inputPath)
{
    const bool display = true;
    const bool threshold = true;
    const bool ROI = true;
    const bool gradient = false;
    const bool lbp = false;

    const std::string truePath = inputPath + "true.jpg";
    const std::string fakePath = inputPath + "fake.jpg";

    // Read the input image
    cv::Mat trueImage;
    cv::Mat fakeImage;
    io::readImage(truePath, trueImage);
    io::readImage(fakePath, fakeImage);

    // Convert it to gray
    cv::Mat trueGrayImage;
    cv::Mat fakeGrayImage;
    util::convertToGray(trueImage, trueGrayImage);
    util::convertToGray(fakeImage, fakeGrayImage);

    // Threshold the image
    cv::Mat trueThresh(trueGrayImage.rows, trueGrayImage.cols, CV_8U, cv::Scalar(255));
    cv::Mat fakeThresh(fakeGrayImage.rows, fakeGrayImage.cols, CV_8U, cv::Scalar(255));
    if(threshold)
    {
        const size_t count = 10;
        const size_t step = 255 / count;
        for (int i = 1 ; i < count-1 ; ++i)
        {
            util::thresholdImage(trueGrayImage, i*step, trueThresh, i*step);
            util::thresholdImage(fakeGrayImage, i*step, fakeThresh, i*step);
        }

        io::saveImage(inputPath+"true_thresh.jpg", trueThresh);
        io::saveImage(inputPath+"fake_thresh.jpg", fakeThresh);

        if(display)
        {
            io::displayImage(trueThresh, "true thresholded image");
            io::displayImage(fakeThresh, "fake thresholded image");
        }
    }
    else
    {
        trueThresh = trueGrayImage;
        fakeThresh = fakeGrayImage;
    }

    // Get only the ROI
    cv::Mat trueROI;
    cv::Mat fakeROI;
    if(ROI)
    {
        util::maskImage(trueGrayImage, trueThresh, trueROI);
        util::maskImage(fakeGrayImage, fakeThresh, fakeROI);

        io::saveImage(inputPath+"true_ROI.jpg", trueROI);
        io::saveImage(inputPath+"fake_ROI.jpg", fakeROI);

        if(display)
        {
            io::displayImage(trueROI, "true ROI image");
            io::displayImage(fakeROI, "fake ROI image");
        }

        std::vector<float> trueROIHist;
        std::vector<float> fakeROIHist;
        util::computeHistogram(trueROI, trueROIHist);
        util::computeHistogram(fakeROI, fakeROIHist);

        std::cout <<"TRUE HISTOGRAM : ";
        for(int i = 0; i < trueROIHist.size(); ++i)
        {
            std::cout << trueROIHist[i] << ", ";
        }

        std::cout <<"FAKE HISTOGRAM : ";
        for(int i = 0; i < fakeROIHist.size(); ++i)
        {
            std::cout << fakeROIHist[i] << ", ";
        }
    }
    else
    {
        trueROI = trueThresh;
        fakeROI = fakeThresh;
    }

    // Extract the X gradient
    cv::Mat trueGrad;
    cv::Mat fakeGrad;
    if(gradient)
    {
        util::extractGradientX(trueROI, trueGrad);
        util::extractGradientX(fakeROI, fakeGrad);

        io::saveImage(inputPath+"true_gradX.jpg", trueGrad);
        io::saveImage(inputPath+"fake_gradX.jpg", fakeGrad);

        if(display)
        {
            io::displayImage(trueGrad, "true gradient X image");
            io::displayImage(fakeGrad, "fake gradient X image");
        }
    }
    else
    {
        trueGrad = trueROI;
        fakeGrad = fakeROI;
    }

    // Run LBP true
    cv::Mat trueLbpGray;
    cv::Mat fakeLbpGray;
    if (lbp)
    {
        util::runLBPOnImage(trueGrad, 10, trueLbpGray);
        util::runLBPOnImage(fakeGrad, 10, fakeLbpGray);

        io::saveImage(inputPath+"true_lbp_gray.jpg", trueLbpGray);
        io::saveImage(inputPath+"fake_lbp_gray.jpg", fakeLbpGray);

        if(display)
        {
            io::displayImage(trueLbpGray, "trueLbpGray");
            io::displayImage(fakeLbpGray, "fakeLbpGray");
        }
    }
}

/// \brief Detect if an attack is happening
/// First the image is loaded and converted to gray.
/// Then the ROI is detected and used as a mask for the image
/// Finally the histogram of the image with ROI is used determine if an attack is happening
/// \param inputPath, the path to the input image to test
/// \return true if an attack is detected, false otherwise
bool detectAttack(const std::string& inputPath)
{
    // Read the image
    cv::Mat image;
    io::readImage(inputPath, image);

    // Convert to grayscale
    cv::Mat grayImage;
    util::convertToGray(image, grayImage);

    // Create the ROI mask by thresholding the image multiple times
    cv::Mat mask(grayImage.rows, grayImage.cols, CV_8U, cv::Scalar(255));
    const size_t count = 10;
    const size_t step = 255 / count;
    for (int i = 1 ; i < count-1 ; ++i)
    {
        util::thresholdImage(grayImage, i*step, mask, i*step);
    }

    // Apply the mask to the image
    cv::Mat roiImage;
    util::maskImage(grayImage, mask, roiImage);

    // Compute the histogram of the image
    std::vector<float> histogram;
    util::computeHistogram(roiImage, histogram);

    // Normalize the histogram
    const size_t pixelCount = roiImage.rows*roiImage.cols;
    for(int i = 0; i < histogram.size(); ++i)
    {
        histogram[i] = histogram[i]*100.0/pixelCount;
    }

    // Check we have a correct picture
    if (histogram[0] > 70.0)
    {
        return true;
    }
    return false;
}

/// \brief Parse the argument
/// \param inputPath, path to the input file
/// \throw if the argument count is incorrect or if the file doesn't exists
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
        std::string inputPath;
        parseArgs(argc, argv, inputPath);
        if (detectAttack(inputPath))
        {
            std::cout << "An attack was detected !!!!" << std::endl;
        }

        std::cout << "No attack was detected" << std::endl;
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
