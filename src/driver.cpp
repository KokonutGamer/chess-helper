#include <iostream>
#include <limits>
#include <vector>

#include <opencv2/opencv.hpp>

#include "ChessHelper/utils.h"

const int NUM_DOWNSAMPLES = 2;
const int MARGIN = 0; // in pixels

namespace ch = ChessHelper;

int main() {
  cv::Mat gray =
      cv::imread("images/empty-chess-board.jpg", cv::IMREAD_GRAYSCALE);

  if (gray.empty()) {
    return EXIT_FAILURE;
  }

  // found that downsampling is pretty quick and can help with speeding up
  // computation
  for (int i = 0; i < NUM_DOWNSAMPLES; i++) {
    cv::pyrDown(gray, gray);
  }

  // use float for harris corner detection; needs to be normalized to work
  // properly with cv::cornerHarris
  gray.convertTo(gray, CV_32F, 1.0 / 255.0);

  // returns CV_32FC1 (32-bit float with one color channel)
  cv::Mat corners = cv::Mat::zeros(gray.size(), gray.type());
  cv::cornerHarris(gray, corners, 3, 7, 0.05);

  float max = std::numeric_limits<float>::min();
  corners.forEach<float>([&](float &pixel, const int *position) {
    if (pixel > max) {
      max = pixel;
    }
  });

  cv::threshold(corners, corners, max * 0.05, 255, cv::THRESH_BINARY);
  corners.convertTo(corners, CV_8U);

  ch::Optional<std::vector<cv::Point2i>> outer = ch::findCorners(corners);

  if (!outer.second) {
    std::cout << "Could not find corners" << std::endl;
    return EXIT_SUCCESS;
  }

  // we need these points as a float for the perspective transform
  std::vector<cv::Point2f> points(outer.first.begin(), outer.first.end());
  std::cout << "Found corners: " << points[0];
  for (int i = 1; i < points.size(); i++) {
    std::cout << ", " << points[i];
  }
  std::cout << std::endl;

  // must match source point order (TL, TR, BL, BR)
  std::vector<cv::Point2f> destination = {
      {MARGIN, MARGIN},
      {MARGIN, static_cast<float>(gray.cols - MARGIN - 1)},
      {static_cast<float>(gray.rows - MARGIN - 1), MARGIN},
      {static_cast<float>(gray.cols - MARGIN - 1),
       static_cast<float>(gray.cols - MARGIN - 1)}};

  cv::Mat M = cv::getPerspectiveTransform(points, destination);
  cv::Mat warped;
  cv::warpPerspective(gray, warped, M, gray.size());

  cv::imshow("Warped", warped);
  cv::waitKey(0);

  return EXIT_SUCCESS;
}
