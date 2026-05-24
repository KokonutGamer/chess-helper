#pragma once

#include <functional>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>

namespace ChessHelper {

template <typename T> using Optional = std::pair<T, bool>;

using Mapping = std::function<cv::Vec2i(const cv::Mat &, int, int)>;

template <typename T> static inline Optional<T> value(T &&val) {
  return {val, true};
}

template <typename T> static inline Optional<T> empty() {
  return std::make_pair<T, bool>(T(), false);
}

static inline cv::Vec2i topLeft(const cv::Mat &image, int r, int c) {
  cv::Vec2i mapping = {r, c};
  return mapping;
}

static inline cv::Vec2i topRight(const cv::Mat &image, int r, int c) {
  cv::Vec2i mapping = {r, image.cols - c - 1};
  return mapping;
}

static inline cv::Vec2i botLeft(const cv::Mat &image, int r, int c) {
  cv::Vec2i mapping = {image.rows - r - 1, c};
  return mapping;
}

static inline cv::Vec2i botRight(const cv::Mat &image, int r, int c) {
  cv::Vec2i mapping = {image.rows - r - 1, image.cols - c - 1};
  return mapping;
}

/**
 * Finds the outer-most corners using the binary mask produced by corner
 * detection.
 */
Optional<std::vector<cv::Vec2i>> findCorners(cv::Mat &);
} // namespace ChessHelper
