#include "ChessHelper/utils.h"

namespace ch = ChessHelper;

static ch::Optional<cv::Vec2i> findCorner(const cv::Mat &image, int minR,
                                          int maxR, int minC, int maxC,
                                          const ch::Mapping &mapper) {
  if ((maxR - minR <= 2) && (maxC - minC <= 2)) {
    for (int r = minR; r < maxR; r++) {
      for (int c = minC; c < maxC; c++) {
        cv::Vec2i real = mapper(image, r, c);
        if (image.at<unsigned char>(real[0], real[1]) == 255) {
          cv::Vec2i val = {real[0], real[1]};
          return ch::value<cv::Vec2i>(std::move(val));
        }
      }
    }
    return ch::empty<cv::Vec2i>();
  }

  int midR = minR + (maxR - minR) / 2 + 1;
  int midC = minC + (maxC - minC) / 2 + 1;

  ch::Optional<cv::Vec2i> topLeft =
      findCorner(image, minR, midR, minC, midC, mapper);
  if (topLeft.second) {
    return topLeft;
  }

  ch::Optional<cv::Vec2i> topRight =
      findCorner(image, minR, midR, midC, maxC, mapper);
  if (topRight.second) {
    return topRight;
  }

  ch::Optional<cv::Vec2i> botLeft =
      findCorner(image, midR, maxR, minC, midC, mapper);
  if (botLeft.second) {
    return botLeft;
  }

  // worst case bottom right (can be null)
  ch::Optional<cv::Vec2i> botRight =
      findCorner(image, midR, maxR, midC, maxC, mapper);
  return botRight;
}

static ch::Optional<cv::Vec2i> findTopLeft(cv::Mat &image, int minR, int maxR,
                                           int minC, int maxC) {
  if ((maxR - minR <= 2) && (maxC - minC <= 2)) {
    for (int r = minR; r < maxR; r++) {
      for (int c = minC; c < maxC; c++) {
        if (image.at<unsigned char>(r, c) == 255) {
          cv::Vec2i val = {r, c};
          return ch::value<cv::Vec2i>(std::move(val));
        }
      }
    }
    return ch::empty<cv::Vec2i>();
  }

  int midR = minR + (maxR - minR) / 2 + 1;
  int midC = minC + (maxC - minC) / 2 + 1;

  ch::Optional<cv::Vec2i> topLeft = findTopLeft(image, minR, midR, minC, midC);
  if (topLeft.second) {
    return topLeft;
  }

  ch::Optional<cv::Vec2i> topRight = findTopLeft(image, minR, midR, midC, maxC);
  if (topRight.second) {
    return topRight;
  }

  ch::Optional<cv::Vec2i> botLeft = findTopLeft(image, midR, maxR, minC, midC);
  if (botLeft.second) {
    return botLeft;
  }

  // worst case bottom right (can be null)
  ch::Optional<cv::Vec2i> botRight = findTopLeft(image, midR, maxR, midC, maxC);
  return botRight;
}

ch::Optional<std::vector<cv::Vec2i>> ch::findCorners(cv::Mat &image) {

  // assertions (image must be non-empty and 8-bit unsigned)
  if (image.empty()) {
    return ch::empty<std::vector<cv::Vec2i>>();
  }

  if (image.type() != CV_8U) {
    return ch::empty<std::vector<cv::Vec2i>>();
  }

  std::vector<cv::Vec2i> corners;
  corners.reserve(4);

  ch::Optional<cv::Vec2i> topLeft =
      findCorner(image, 0, image.rows, 0, image.cols, &ch::topLeft);
  if (topLeft.second) {
    corners.push_back(topLeft.first);
  }

  ch::Optional<cv::Vec2i> topRight =
      findCorner(image, 0, image.rows, 0, image.cols, &ch::topRight);
  if (topRight.second) {
    corners.push_back(topRight.first);
  }

  ch::Optional<cv::Vec2i> botLeft =
      findCorner(image, 0, image.rows, 0, image.cols, &ch::botLeft);
  if (botLeft.second) {
    corners.push_back(botLeft.first);
  }

  ch::Optional<cv::Vec2i> botRight =
      findCorner(image, 0, image.rows, 0, image.cols, &ch::botRight);
  if (botRight.second) {
    corners.push_back(botRight.first);
  }

  if (corners.empty()) {
    return ch::empty<std::vector<cv::Vec2i>>();
  }

  return ch::value(std::move(corners));
}
