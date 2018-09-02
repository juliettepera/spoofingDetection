#include <opencv2/opencv.hpp>

#include <exception>

namespace
{
const size_t NEIGHBORS = 8;
const int R_NEIGBHORS[] = {-1, -1, -1, 0, 1, 1, 1, 0};
const int C_NEIGBHORS[] = {-1, 0, 1, 1, 1, 0, -1, -1};

/// \brief Compute and return the score of the reference pixel within the cell
/// The score is computed by going through the 8 neighboring pixels of the reference pixel.
/// For each neighbors, their value is compared to the reference pixel's value.
/// If ref value < neighbor value, then the neighbhor pixel get attributed a score of 1
/// Else, the neighbhor pixel get attributed a score of 0
/// Those scores finally form a 8-bit value that is converted to decimal.
/// This decimal value represent the reference pixel's score
/// \note For the score to be consistent, we must pass through the neighbors in a similar order
/// everytime we compute the score for a pixel, and for each cell.
/// \param cell, the input cell to score
/// \param r, the row index of the pixel, it must fit within an 8 neighbors box in the cell
/// \param c, the col index of the pixel, it must fit within an 8 neighbors box in the cell
/// \return the score of the pixel
/// \throw if the reference pixel cannot fit in an 8-neighbors window within the cell
uchar scorePixel(const cv::Mat& cell, const int r, const int c)
{
    // Check our pixel is within an 8-neighbors window
    if (r < 1 || r > cell.rows - 2 || c < 1 || c > cell.cols - 2)
    {
        throw std::runtime_error("Reference pixel is out of bound");
    }

    // Get the reference pixel's value
    const double pixelValue = cell.at<double>(r,c);

    // Initialize the score
    size_t score = 0;

    // Go through the neighbors in a clockwise way
    for(size_t n = 0; n < NEIGHBORS; ++n)
    {
        // If the reference pixel's value is bellow the neighbor's value
        // The neighbor's score will be "1" in the 8-bit value
        // Thus 2^(neighbor's position) in the decimal value
        if (pixelValue < cell.at<double>(R_NEIGBHORS[n], C_NEIGBHORS[n]))
        {
            score += std::pow(2,n);
        }
    }
    return score;
}

/// \brief Fill the score matrix for a specific cell
/// \note The scores matrix will have the same size of cell,
/// however a border of 1 pixel size won't contain any sccore
/// \param cell, the cell to score
/// \param scores, the scores matrix
/// \throw if the cell is not square and it's size is above 20
void scoreCell(const cv::Mat& cell, cv::Mat& scores)
{
    // Check the cell's dimension
    const int cellSize = cell.rows;
    if(cellSize != cell.cols || cellSize > 20)
    {
        throw std::runtime_error("Wrong cell size");
    }

    // Score each pixel's of the cell
    for(size_t r = 1; r < cellSize - 1; ++r)
    {
        for(size_t c = 1; c < cellSize - 1; ++c)
        {
            scores.at<uchar>(r,c) = scorePixel(cell, r, c);
        }
    }
}
}

namespace util
{
void convertToGray(const cv::Mat& inputImage, cv::Mat& grayImage)
{
    cv::cvtColor(inputImage, grayImage, CV_BGR2GRAY);

    if ( !grayImage.data )
    {
        throw std::runtime_error("Failed to convert the image to gray");
    }
}

void createHistogram(const cv::Mat& inputImage, cv::Mat& histImage)
{
    // Establish the number of bins
    int histSize = 256;

    // Set the ranges
    float range[] = { 0, 256 } ;
    const float* histRange = { range };

    bool uniform = true;
    bool accumulate = false;

    cv::Mat hist;

    // Compute the histograms:
    cv::calcHist(&inputImage, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);

    // Draw the histograms for B, G and R
    int hist_w = 512; int hist_h = 400;
    int bin_w = cvRound( (double) hist_w/histSize );

    histImage = cv::Mat( hist_h, hist_w, CV_8UC3, cv::Scalar(0,0,0));

    // Normalize the result to [ 0, histImage.rows ]
    cv::normalize(hist, hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );

    // Draw
    for( int i = 1; i < histSize; i++ )
    {
        cv::line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(hist.at<float>(i-1)) ) ,
                  cv::Point( bin_w*(i), hist_h - cvRound(hist.at<float>(i)) ),
                  cv::Scalar( 255, 0, 0), 2, 8, 0  );
    }
}

/// \brief Threshold the image wrt a threshold value.
/// If the value of the pixel is above the threshold value, the pixel is set to "binValue" on the output image
/// \note The output image is not cleanup before and only the pixel above the threshold value will be modified.
/// This is to be able to have multiple thresholds value on the same image
/// \param inputImage, the input image to threshold
/// \param threshValue, the threshold value
/// \param threshImage, the thresholded image
/// \param binValue, the value of the pixel's above the threshold value in the thresholded image
void thresholdImage(const cv::Mat& inputImage, const size_t threshValue, cv::Mat& threshImage, const size_t binValue = 255)
{
    for(int r = 0; r < inputImage.rows ; ++r)
    {
        for(int c = 0; c < inputImage.cols ; ++c)
        {
            const size_t val = inputImage.at<uchar>(r,c);
            if (val > threshValue)
            {
                threshImage.at<uchar>(r,c) = binValue;
            }
        }
    }
}

void extractGradientX(const cv::Mat& inputImage, cv::Mat& gradImage)
{
    cv::Mat sobelx;
    cv::Sobel(inputImage, sobelx, CV_32F, 1, 0);
    double minVal, maxVal;
    cv::minMaxLoc(sobelx, &minVal, &maxVal); //find minimum and maximum intensities
    sobelx.convertTo(gradImage, CV_8U, 255.0/(maxVal - minVal), -minVal * 255.0/(maxVal - minVal));
}

void extractGradientY(const cv::Mat& inputImage, cv::Mat& gradImage)
{
    cv::Mat sobely;
    cv::Sobel(inputImage, sobely, CV_32F, 0, 1);
    double minVal, maxVal;
    cv::minMaxLoc(sobely, &minVal, &maxVal); //find minimum and maximum intensities
    sobely.convertTo(gradImage, CV_8U, 255.0/(maxVal - minVal), -minVal * 255.0/(maxVal - minVal));
}

/// \brief Run the LBP (local binary pattern) algorithm on the inputImage
/// As the the score is local, it must be run on small sized cells
/// \param inputImage, the input image to run the LBP on
/// \param cellSize, the size of the cell
/// \param lbpImage, the output of the lpb scores
void runLBPOnImage(const cv::Mat& inputImage, const int cellSize, cv::Mat& lbpImage)
{
    const int rowSize = inputImage.rows;
    const int colSize = inputImage.cols;

    // Initialize the number of cells depending on the chosen size
    const int cellCountR = std::floor(rowSize/cellSize);
    const int cellCountC = std::floor(colSize/cellSize);

    // Initialize the output image
    lbpImage = cv::Mat(rowSize, colSize, CV_8U, cv::Scalar(0));

    // For each cells in the image, compute the score of the cell and
    // add the results to the lbp output matrix
    for(int cR = 0; cR < cellCountR; ++cR)
    {
        for(int cC = 0; cC < cellCountC; ++cC)
        {
            // Get the cell
            cv::Rect roi(cC*cellSize, cR*cellSize, cellSize, cellSize);
            cv::Mat cell = inputImage(roi);

            // Score the cell
            cv::Mat scores(cellSize,cellSize,CV_8U, cv::Scalar(0));
            scoreCell(cell, scores);

            // Add the cell score to the LBP matrix
            cv::Mat aux = lbpImage.rowRange(cR*cellSize, (cR+1)*cellSize).colRange(cC*cellSize, (cC+1)*cellSize);
            scores.copyTo(aux);
        }
    }
}
}
