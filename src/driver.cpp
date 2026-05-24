#include <iostream>
#include <limits>
#include <vector>

#include <opencv2/opencv.hpp>

#include "ChessHelper/utils.h"

const int NUM_DOWNSAMPLES = 2;

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

  float min = std::numeric_limits<float>::max();
  float max = std::numeric_limits<float>::min();
  corners.forEach<float>([&](float &pixel, const int *position) {
    if (pixel < min) {
      min = pixel;
    }
    if (pixel > max) {
      max = pixel;
    }
  });

  std::cout << "Min: " << min << std::endl;
  std::cout << "Max: " << max << std::endl;

  cv::threshold(corners, corners, max * 0.05, 255, cv::THRESH_BINARY);
  corners.convertTo(corners, CV_8U);
  std::cout << "New corner type: " << corners.type() << std::endl;

  cv::imshow("Thresholded", corners);
  cv::waitKey(0);

  ch::Optional<std::vector<cv::Vec2i>> outer = ch::findCorners(corners);

  if (outer.second) {
    std::cout << "Found corners: " << outer.first[0];
    for (int i = 1; i < outer.first.size(); i++) {
      std::cout << ", " << outer.first[i];
    }
    std::cout << std::endl;
  }

  return EXIT_SUCCESS;
}
