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

/**
 * TODO document
 */
cv::Subdiv2D delaunay(cv::Mat &image, const std::vector<cv::Point2f> &points,
                      const cv::Scalar &color = cv::Scalar(0, 255, 0));
} // namespace ChessHelper