#include "ChessHelper/utils.h"

namespace ChessHelper {

/**
 * Transforms the given image to a square image
 * according to the perspective transform matrix.
 * @param image is the image to convert (not modified).
 * @param transform is the perspective matrix to transform the image with.
 * @return a newly transformed image.
 */
cv::Mat ChessHelper::warpImage(const cv::Mat &image, const cv::Mat &transform) {
  cv::Mat warped;
  cv::warpPerspective(image, warped, transform, image.size());

  // The image might not be perfectly square (off by a pixel).
  int smallAxis = std::min(warped.cols - 1, warped.rows - 1);
  warped = warped(cv::Rect(0, 0, smallAxis, smallAxis));

  return warped;
}

} // namespace ChessHelper