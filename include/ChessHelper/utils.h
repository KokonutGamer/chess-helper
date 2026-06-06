#pragma once

#include <cmath>
#include <functional>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>

namespace ChessHelper {

/**
 * Defines an alias named Optional since C++14 doesn't have std::optional yet.
 */
template <typename T> using Optional = std::pair<T, bool>;

/**
 * Factory method for creating an Optional on success.
 */
template <typename T> static inline Optional<T> value(T &&val) {
  return {val, true};
}

/**
 * Factory method for creating an empty Optional.
 */
template <typename T> static inline Optional<T> empty() {
  return std::make_pair<T, bool>(T(), false);
}

/**
 * Transforms the given image to a square image according to the perspective
 * transform matrix.
 *
 * @param image		The image to convert (not modified).
 * @param transform	The perspective matrix to transform the image with.
 * @return			A newly transformed image.
 */
cv::Mat warpImage(const cv::Mat &image, const cv::Mat &transform);
} // namespace ChessHelper
