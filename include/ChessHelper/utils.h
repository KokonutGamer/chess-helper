#pragma once

#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>

namespace ChessHelper {

template <typename T> using Optional = std::pair<T, bool>;

template <typename T> static inline Optional<T> value(T&& val) {
  return {val, true};
}

template <typename T> static inline Optional<T> empty() {
  return std::make_pair<T, bool>(T(), false);
}

/**
 * Finds the outer-most corners using the binary mask produced by corner
 * detection.
 */
Optional<std::vector<cv::Vec2i>> findCorners(cv::Mat &);
} // namespace ChessHelper
