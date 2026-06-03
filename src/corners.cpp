#include "ChessHelper/corners.h"

namespace ChessHelper {
/**
 * TODO document
 */
cv::Mat detectCorners(const cv::Mat &image) {
  // convert video frame to grayscale
  cv::Mat grayOriginal;
  cv::cvtColor(image, grayOriginal, cv::COLOR_BGR2GRAY);
  grayOriginal.convertTo(grayOriginal, CV_32F, 1.0 / 255.0);

  // harris corner detection
  cv::Mat corners = cv::Mat::zeros(grayOriginal.size(), grayOriginal.type());
  cv::cornerHarris(grayOriginal, corners, 3, 7, 0.05);

  // use a small 3x3 kernel for morphological open (erosion then dilation)
  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
  cv::Mat opened;
  cv::morphologyEx(corners, opened, cv::MORPH_OPEN, kernel);
  opened.convertTo(opened, CV_8U);

  // small threshold for now
  cv::threshold(opened, opened, 10, 255, cv::THRESH_BINARY);
  opened.convertTo(opened, CV_8U);

  // erode result so we have smaller clusters of pixels to work with
  cv::Mat eroded;
  cv::erode(opened, eroded, kernel);

  return eroded; // should be return-value optimized
}

std::vector<cv::Point2f> collapsePoints(const cv::Mat &corners) {
  if (corners.empty()) {
    throw std::runtime_error("Corners must not be empty.");
  }

  if (corners.type() != CV_8U) {
    throw std::runtime_error("Corners must be 8-bit grayscale.");
  }

  cv::Mat labels, stats, centroids;

  // group connected components together (eight directions)
  int numLabels =
      cv::connectedComponentsWithStats(corners, labels, stats, centroids);

  std::vector<cv::Point2f> points;

  if (numLabels <= 1) {
    return points; // no foreground found
  }

  points.reserve(numLabels - 1);

  // ignore background (label 0)
  for (int i = 1; i < numLabels; i++) {
    double x = centroids.at<double>(i, 0);
    double y = centroids.at<double>(i, 1);
    points.push_back(cv::Point2f(static_cast<float>(x), static_cast<float>(y)));
  }
  return points;
}
} // namespace ChessHelper