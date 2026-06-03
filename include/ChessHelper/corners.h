#pragma once

#include <opencv2/opencv.hpp>

namespace ChessHelper {

/**
 * TODO document
 */
cv::Mat detectCorners(const cv::Mat &image);
} // namespace ChessHelper