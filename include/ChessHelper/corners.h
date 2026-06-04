#pragma once

#include <opencv2/opencv.hpp>

#include <vector>

namespace ChessHelper {

/**
 * TODO document, will include Gaussian blur step
 */
cv::Mat sample(const cv::Mat &image, int ksize = 5, double sigma = 1.02);

cv::Mat adjustGamma(const cv::Mat &image, double gamma = 4.0);

} // namespace ChessHelper