#include "ChessHelper/corners.h"
#include "ChessHelper/utils.h"

/**
 * Offset for response calculation
 */
static constexpr int D[16] = {-5, -5, -4, -2, 0, 2,  4,  5,
                              5,  5,  4,  2,  0, -2, -4, -5};

/**
 * Sum 4 sections, find the diff between 8 sections, and take mean across all 16
 * neighbors around the circle
 */
static constexpr int SUM_NUM_NEIGHBORS = 4;
static constexpr int DIFF_NUM_NEIGHBORS = 8;
static constexpr int NUM_NEIGHBORS = 16;

static constexpr double CORRECTION_GAMMA = 5.0;
static constexpr double RESPONSE_THRESHOLD = 0.5;

/**
 * TODO document, assumes r and c are already in image (just a helper function)
 */
static inline int getInt(const cv::Mat &image, int r, int c) {
  return static_cast<int>(image.at<unsigned char>(r, c));
}

/**
 * TODO document, assumes r and c are already in image (just a helper function)
 */
static inline double getDouble(const cv::Mat &image, int r, int c) {
  return static_cast<double>(image.at<unsigned char>(r, c));
}

/**
 * TODO document, image must be grayscale uchar; ASSUME CANDIDATE IS CORRECT
 * DIMENSIONS, CORRECT TYPE AND IMAGE IS NON-NULL AND CORRECT TYPE
 */
static double calcResponse(const cv::Mat &image, int r, int c) {
  // note that we can assume all points on the circle will be within the image;
  // we do not sample pixels on the edges and corners of the image (it's only an
  // 11 x 11 area anyways)
  double sumRes = 0.0;
  double diffRes = 0.0;
  double meanNeighbors = 0.0;
  for (int i = 0; i < NUM_NEIGHBORS; i++) {
    int curr = getInt(image, r + D[i], c + D[(i + 4) % NUM_NEIGHBORS]);
    meanNeighbors += static_cast<double>(curr);
    if (i >= DIFF_NUM_NEIGHBORS)
      continue;

    int currOpp = getInt(image, r + D[i + 8], c + D[(i + 12) % NUM_NEIGHBORS]);
    diffRes += std::abs(curr - currOpp);
    if (i >= SUM_NUM_NEIGHBORS)
      continue;

    // take a pixel, its opposing endpoint on the diameter, and the pixels on
    // the perpendicular bisector (perpendicular diameter)
    // won't access out-of-bounds since i + offset for below will always be less
    // than NUM_NEIGHBORS
    int currNext = getInt(image, r + D[i + 4], c + D[i + 8]);
    int currPrev = getInt(image, r + D[i + 12], c + D[i + 0]);
    sumRes += std::abs((curr + currOpp) - (currNext + currPrev));
  }
  meanNeighbors /= NUM_NEIGHBORS;

  double meanLocal = (getDouble(image, r, c) + getDouble(image, r - 1, c) +
                      getDouble(image, r, c + 1) + getDouble(image, r + 1, c) +
                      getDouble(image, r, c - 1)) /
                     5.0;

  double meanRes = std::abs(meanNeighbors - meanLocal);
  return sumRes - diffRes - 16 * meanRes;
}

namespace ChessHelper {

/**
 * TODO document
 */
cv::Mat sample(const cv::Mat &image, int ksize, double sigma) {
  if (image.empty()) {
    throw std::runtime_error("Image must not be empty.");
  }

  if (image.type() != CV_8UC3) {
    throw std::runtime_error("Image must be BGR.");
  }
  cv::Mat gray;
  cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

  // in-place (not truly, but handled by OpenCV)
  cv::GaussianBlur(gray, gray, cv::Size(ksize, ksize), sigma);

  cv::Mat cand = cv::Mat::zeros(image.size(), CV_64FC1);
  for (int i = ksize; i < gray.rows - ksize; i++) {
    for (int j = ksize; j < gray.cols - ksize; j++) {
      cand.at<double>(i, j) = calcResponse(gray, i, j);
    }
  }
  cv::normalize(cand, cand, 1.0, 0.0, cv::NORM_MINMAX);
  return cand;
}

Optional<std::vector<cv::Point>> centerCorners(const cv::Mat &response) {
  // dilate response image to keep maximums in 5 x 5 neighborhood
  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_DILATE, cv::Size(5, 5));
  cv::Mat pointResponse;
  cv::dilate(response, pointResponse, kernel);

  // gamma correction (pixels SHOULD be within 0.0 to 1.0 range)
  cv::pow(pointResponse, CORRECTION_GAMMA, pointResponse);

  // need to threshold to binary mask for centroid calculation
  cv::Mat inter;
  cv::threshold(pointResponse, inter, RESPONSE_THRESHOLD, 255.0,
                cv::THRESH_BINARY);
  cv::Mat mask;
  inter.convertTo(mask, CV_8UC1);

  // calculate centroids as average of connected components
  cv::Mat labels, stats, centroids;
  int num = cv::connectedComponentsWithStats(mask, labels, stats, centroids);

  if (num <= 1) {
    return empty<std::vector<cv::Point>>();
  }

  std::vector<cv::Point> points;

  // skip background (label 0)
  for (int i = 1; i < num; i++) {
    double x = centroids.at<double>(i, 0);
    double y = centroids.at<double>(i, 1);
    points.emplace_back(static_cast<int>(x), static_cast<int>(y));
  }

  std::vector<cv::Point> hull;
  cv::convexHull(points, hull);

  std::vector<cv::Point> quad;
  double epsilon = 0.02 * cv::arcLength(hull, true);

  while (true) {
    cv::approxPolyDP(hull, quad, epsilon, true);
    if (quad.size() == 4) {
      break;
    } else if (quad.size() < 4) {
      epsilon *= 0.9;
    } else {
      epsilon *= 1.1;
    }
  }

  return value<std::vector<cv::Point>>(std::move(quad));
}

} // namespace ChessHelper