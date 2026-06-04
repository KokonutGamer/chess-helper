#pragma once

#include <cmath>
#include <functional>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>

namespace ChessHelper {

template <typename T> using Optional = std::pair<T, bool>;

using Mapping = std::function<cv::Point2i(const cv::Mat &, int, int)>;

template <typename T> static inline Optional<T> value(T &&val) {
  return {val, true};
}

template <typename T> static inline Optional<T> empty() {
  return std::make_pair<T, bool>(T(), false);
}

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

/**
 * Transforms the given image to a grayscale (CV_8U) square image
 * according to the perspective transform matrix.
 * @param image is the image to convert (not modified).
 * @param transform is the perspective matrix to transform the image with.
 * @return a newly transformed image.
 */
cv::Mat grayWarp(const cv::Mat &image, const cv::Mat &transform);

/**
 * TODO document
 */
static inline int clamp(float val, int low, int high) {
  return (val < low) ? low : (val > high) ? high : static_cast<int>(val);
}
} // namespace ChessHelper
