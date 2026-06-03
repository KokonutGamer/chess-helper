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
} // namespace ChessHelper