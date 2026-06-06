#pragma once

#include <ChessHelper/utils.h>

#include <opencv2/opencv.hpp>

#include <vector>

namespace ChessHelper {

using Mapping = std::function<cv::Point2i(const cv::Mat &, int, int)>;

/**
 * Execute the sampling algorithm based on Chess-board Extraction by Subtraction
 * and Summation. Before sampling, a Gaussian blur is used to preprocess the
 * image to remove noise. After sampling, the response is normalized to
 * [0.0, 1.0] in order to apply gamma correction.
 *
 * @param image     A BGR image of type CV_8UC3
 * @param ksize     The kernel size of the Gaussian blur
 * @param sigma     The scale of the Gaussian blur
 * @return          The response image of type CV_64FC1
 */
cv::Mat sample(const cv::Mat &image, int ksize = 5, double sigma = 1.02);

/**
 * Discovers the center of the corners using the response from the sampling
 * step. A dilation is applied to keep the max responses within a neighbor.
 * Gamma correction is then used to accentuate high responses while lowering low
 * ones. Ideally, high-confidence corner candidates stay above a certain
 * threshold.
 *
 * After these first few steps, corner candidates are usually connected
 * (adjacent) to one another. Connected components are calculated along with
 * their centroids at their average position.
 *
 * To find the the four outer-points of the board (not including the edge
 * corners), the maximum quadrilateral is taken using the convex hull of the
 * remaining candidate points. This is approximated using the
 * Ramer-Douglas–Peucker algorithm (as implemented in approxPolyDP).
 *
 * Assumes the response is a double-precision floating-point grayscale image
 * normalized to [0.0, 1.0].
 *
 * @param response      A grayscale image of type CV_64FC1
 * @return              An Optional containing the four outer-most points if
 *							found.
 */
Optional<std::vector<cv::Point>> centerCorners(const cv::Mat &response);

static inline cv::Point2i topLeft(const cv::Mat &image, int r, int c) {
  cv::Point2i mapping = {r, c};
  return mapping;
}

static inline cv::Point2i topRight(const cv::Mat &image, int r, int c) {
  cv::Point2i mapping = {r, image.cols - c - 1};
  return mapping;
}

static inline cv::Point2i botLeft(const cv::Mat &image, int r, int c) {
  cv::Point2i mapping = {image.rows - r - 1, c};
  return mapping;
}

static inline cv::Point2i botRight(const cv::Mat &image, int r, int c) {
  cv::Point2i mapping = {image.rows - r - 1, image.cols - c - 1};
  return mapping;
}

/**
 * Computes the absolute value of the hyperbolic metric. Higher values score
 * better (maximizes distance in one axis while minimizes distance in the
 * other).
 */
static inline int hyperbolic(int r, int c, int targetR, int targetC) {
  int distR = std::abs(r - targetR);
  int distC = std::abs(c - targetC);
  return std::abs(distR - distC);
}

/**
 * Finds the outer-most corners using the binary mask produced by corner
 * detection.
 */
Optional<std::vector<cv::Point2i>> findCorners(cv::Mat &);
} // namespace ChessHelper