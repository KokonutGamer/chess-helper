#pragma once

#include <ChessHelper/utils.h>

#include <opencv2/opencv.hpp>

#include <vector>

namespace ChessHelper {

/**
 * TODO document, will include Gaussian blur step
 */
cv::Mat sample(const cv::Mat &image, int ksize = 5, double sigma = 1.02);

/**
 * TODO document
 */
Optional<std::vector<cv::Point>> centerCorners(const cv::Mat &response);

} // namespace ChessHelper
