#pragma once

#include <opencv2/opencv.hpp>

#include <vector>

namespace ChessHelper {

/**
 * TODO document
 */
cv::Mat detectCorners(const cv::Mat &image);

/**
 * TODO document
 */
std::vector<cv::Point2f> collapsePoints(const cv::Mat &corners);
} // namespace ChessHelper