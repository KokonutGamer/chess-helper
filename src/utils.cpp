#include "ChessHelper/utils.h"

namespace ch = ChessHelper;

static ch::Optional<cv::Point2i> findCorner(const cv::Mat &image, int minR,
                                            int maxR, int minC, int maxC,
                                            const ch::Mapping &mapper) {

  // Base case, traverses the smallest region to find a value
  if ((maxR - minR <= 2) && (maxC - minC <= 2)) {
    for (int r = minR; r < maxR; r++) {
      for (int c = minC; c < maxC; c++) {
        cv::Point2i real = mapper(image, r, c);
        if (image.at<unsigned char>(real.x, real.y) == 255) {
          cv::Point2i val = {real.x, real.y};
          return ch::value<cv::Point2i>(std::move(val));
        }
      }
    }
    return ch::empty<cv::Point2i>();
  }

  int midR = minR + (maxR - minR) / 2 + 1;
  int midC = minC + (maxC - minC) / 2 + 1;

  // Because this function maps values (so we don't have to write three other
  // separate functions to perform the corner matching), the order is what is
  // expected from a top-left-only implementation.

  // always return the top-left result
  ch::Optional<cv::Point2i> topLeft =
      findCorner(image, minR, midR, minC, midC, mapper);
  if (topLeft.second) {
    return topLeft;
  }

  // For top-right and bot-left, we must compare them based on the hyperbolic
  // metric. This maximizes distance in either x or y while minimizes distance
  // in the other.
  ch::Optional<cv::Point2i> topRight =
      findCorner(image, minR, midR, midC, maxC, mapper);
  ch::Optional<cv::Point2i> botLeft =
      findCorner(image, midR, maxR, minC, midC, mapper);
  if (topRight.second && botLeft.second) {
    cv::Point2i target = mapper(image, minR, minC);

    int trScore =
        ch::hyperbolic(topRight.first.x, topRight.first.y, target.x, target.y);
    int blScore =
        ch::hyperbolic(botLeft.first.x, botLeft.first.y, target.x, target.y);

    return (trScore > blScore) ? topRight : botLeft;
  }

  if (topRight.second) {
    return topRight;
  }

  if (botLeft.second) {
    return botLeft;
  }

  // worst case bottom right (can be null)
  ch::Optional<cv::Point2i> botRight =
      findCorner(image, midR, maxR, midC, maxC, mapper);
  return botRight;
}

ch::Optional<std::vector<cv::Point2i>> ch::findCorners(cv::Mat &image) {

  // assertions (image must be non-empty and 8-bit unsigned)
  if (image.empty()) {
    return ch::empty<std::vector<cv::Point2i>>();
  }

  if (image.type() != CV_8U) {
    return ch::empty<std::vector<cv::Point2i>>();
  }

  std::vector<cv::Point2i> corners;
  corners.reserve(4);

  ch::Optional<cv::Point2i> topLeft =
      findCorner(image, 0, image.rows, 0, image.cols, &ch::topLeft);
  if (topLeft.second) {
    corners.push_back(topLeft.first);
  }

  ch::Optional<cv::Point2i> topRight =
      findCorner(image, 0, image.rows, 0, image.cols, &ch::topRight);
  if (topRight.second) {
    corners.push_back(topRight.first);
  }

  ch::Optional<cv::Point2i> botLeft =
      findCorner(image, 0, image.rows, 0, image.cols, &ch::botLeft);
  if (botLeft.second) {
    corners.push_back(botLeft.first);
  }

  ch::Optional<cv::Point2i> botRight =
      findCorner(image, 0, image.rows, 0, image.cols, &ch::botRight);
  if (botRight.second) {
    corners.push_back(botRight.first);
  }

  if (corners.empty()) {
    return ch::empty<std::vector<cv::Point2i>>();
  }

  return ch::value(std::move(corners));
}
