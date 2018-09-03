#include <stdio.h>
#include <string.h>
#include <exception>

#include <opencv2/opencv.hpp>
namespace io
{
/// \brief Display the image on a window, close the window on "enter" key
/// \param image, the image to display
/// \param imageName, the name of the windows to display
void displayImage(const cv::Mat& image, const std::string& windowName = "")
{
    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
    cv::imshow(windowName, image);

    cv::waitKey(0);
}

/// \brief Load the image using the input path and save the image data in the matrix
/// \param inputPath, the path to the image to load
/// \param inputImage, the matrix of the image
/// \throw if the image contains no data
void readImage(const std::string& inputPath, cv::Mat& inputImage)
{
    inputImage = cv::imread( inputPath, 1 );

    if ( !inputImage.data )
    {
        throw std::runtime_error("Failed to read the input image");
    }
}

/// \brief Save the image at the chosen path
/// \param outputPath, the path to save the image
/// \param outputImage, the image to save
/// \throw if the saving failed
void saveImage(const std::string& outputPath, const cv::Mat& outputImage)
{
    if (!cv::imwrite(outputPath, outputImage))
    {
        throw std::runtime_error("Failed to write image");
    }
}

/// \brief Check if the file contained at the path exists
/// \param fileName, the path to the file
/// \return true if the file exists, false otherwise
bool fileExists(const std::string& fileName)
{
    std::ifstream infile(fileName);
    return infile.good();
}
}
