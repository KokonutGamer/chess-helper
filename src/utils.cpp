#include "ChessHelper/utils.h"

namespace ch = ChessHelper;

namespace ChessHelper {
static Optional<cv::Point2i> findCorner(const cv::Mat &image, int minR,
                                        int maxR, int minC, int maxC,
                                        const Mapping &mapper) {

  // Base case, traverses the smallest region to find a value
  if ((maxR - minR <= 2) && (maxC - minC <= 2)) {
    for (int r = minR; r < maxR; r++) {
      for (int c = minC; c < maxC; c++) {
        cv::Point2i real = mapper(image, r, c);
        if (image.at<unsigned char>(real.x, real.y) == 255) {
          cv::Point2i val = {real.x, real.y};
          return value<cv::Point2i>(std::move(val));
        }
      }
    }
    return empty<cv::Point2i>();
  }

  int midR = minR + (maxR - minR) / 2 + 1;
  int midC = minC + (maxC - minC) / 2 + 1;

  // Because this function maps values (so we don't have to write three other
  // separate functions to perform the corner matching), the order is what is
  // expected from a top-left-only implementation.

  // always return the top-left result
  Optional<cv::Point2i> topLeft =
      findCorner(image, minR, midR, minC, midC, mapper);
  if (topLeft.second) {
    return topLeft;
  }

  // For top-right and bot-left, we must compare them based on the hyperbolic
  // metric. This maximizes distance in either x or y while minimizes distance
  // in the other.
  Optional<cv::Point2i> topRight =
      findCorner(image, minR, midR, midC, maxC, mapper);
  Optional<cv::Point2i> botLeft =
      findCorner(image, midR, maxR, minC, midC, mapper);
  if (topRight.second && botLeft.second) {
    cv::Point2i target = mapper(image, minR, minC);

    int trScore =
        hyperbolic(topRight.first.x, topRight.first.y, target.x, target.y);
    int blScore =
        hyperbolic(botLeft.first.x, botLeft.first.y, target.x, target.y);

    return (trScore > blScore) ? topRight : botLeft;
  }

  if (topRight.second) {
    return topRight;
  }

  if (botLeft.second) {
    return botLeft;
  }

  // worst case bottom right (can be null)
  Optional<cv::Point2i> botRight =
      findCorner(image, midR, maxR, midC, maxC, mapper);
  return botRight;
}

Optional<std::vector<cv::Point2i>> findCorners(cv::Mat &image) {

  // assertions (image must be non-empty and 8-bit unsigned)
  if (image.empty()) {
    return empty<std::vector<cv::Point2i>>();
  }

  if (image.type() != CV_8U) {
    return empty<std::vector<cv::Point2i>>();
  }

  std::vector<cv::Point2i> corners;
  corners.reserve(4);

  Optional<cv::Point2i> topLeftCorner =
      findCorner(image, 0, image.rows, 0, image.cols, &topLeft);
  if (topLeftCorner.second) {
    corners.push_back(topLeftCorner.first);
  }

  Optional<cv::Point2i> topRightCorner =
      findCorner(image, 0, image.rows, 0, image.cols, &topRight);
  if (topRightCorner.second) {
    corners.push_back(topRightCorner.first);
  }

  Optional<cv::Point2i> botLeftCorner =
      findCorner(image, 0, image.rows, 0, image.cols, &botLeft);
  if (botLeftCorner.second) {
    corners.push_back(botLeftCorner.first);
  }

  Optional<cv::Point2i> botRightCorner =
      findCorner(image, 0, image.rows, 0, image.cols, &botRight);
  if (botRightCorner.second) {
    corners.push_back(botRightCorner.first);
  }

  if (corners.empty()) {
    return empty<std::vector<cv::Point2i>>();
  }

  return value(std::move(corners));
}

cv::Mat ChessHelper::warpImage(const cv::Mat &image, const cv::Mat &transform) {
  cv::Mat warped;
  cv::warpPerspective(image, warped, transform, image.size());

  // The image might not be perfectly square (off by a pixel).
  int smallAxis = std::min(warped.cols - 1, warped.rows - 1);
  warped = warped(cv::Rect(0, 0, smallAxis, smallAxis));

  return warped;
}
} // namespace ChessHelper