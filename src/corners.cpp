#include "ChessHelper/corners.h"
#include "ChessHelper/utils.h"

/**
 * Offset for response calculation. Let r and c represent a pixel on an image
 * where adding any offset to either r or c will provide a valid coordinate
 * within the bounds of the image.
 *
 * Adding the offset at index i with r and the offset at index (i + 4) % 16 will
 * result in a pixel located approximately 5 pixels away from r and c. These
 * offsets are ordered such that adding them to r and c in the way detailed
 * previously will traverse a circle starting from the top-most pixel (r - 5, c)
 * clockwise. This is technique was inspired by the Chess-board Extraction by
 * Subtraction and Summation detector.
 */
static constexpr int D[16] = {-5, -5, -4, -2, 0, 2,  4,  5,
                              5,  5,  4,  2,  0, -2, -4, -5};

/**
 * Dividing the sixteen points of the circle into quarters results in checking
 * only four neighbors for their "sum response".
 */
static constexpr int SUM_NUM_NEIGHBORS = 4;

/**
 * Dividing the sixteen points of the circle into halves results in checking
 * only eight neighbors for their "diff response".
 */
static constexpr int DIFF_NUM_NEIGHBORS = 8;

/**
 * The total number of neighbors around the sampling circle.
 */
static constexpr int NUM_NEIGHBORS = 16;

/**
 * The gamma correction exponent to apply to a normalized floating-point
 * grayscale image.
 */
static constexpr double CORRECTION_GAMMA = 5.0;

/**
 * The response threshold used to distinguish predicted corners after gamma
 * correction.
 */
static constexpr double RESPONSE_THRESHOLD = 0.5;

/**
 * Casts the byte value from the supplied image into a signed integer. Assumes
 * the image is an unsigned char grayscale image and r and c are within the
 * bounds of the image.
 *
 * @param image     An image of type CV_8UC1
 * @param r         The row to retrieve the pixel from
 * @param c         The column to retrieve the pixel from
 * @return          The pixel value casted to an integer
 */
static inline int getInt(const cv::Mat &image, int r, int c) {
  return static_cast<int>(image.at<unsigned char>(r, c));
}

/**
 * Casts the byte value from the supplied image into a double precision
 * floating-point. Assumes the image is an unsigned char grayscale image and r
 * and c are within the bounds of the image.
 *
 * @param iamge     An image of type CV_8UC1
 * @param r         The row to retrieve the pixel form
 * @param c         The column to retrieve the pixel form
 * @return          The pixel value casted to a double
 */
static inline double getDouble(const cv::Mat &image, int r, int c) {
  return static_cast<double>(image.at<unsigned char>(r, c));
}

/**
 * Calculates the overall response of the pixel at row r and column c. Three
 * measurements are used for this computation: sum response, diff response, and
 * mean response.
 *
 * The sum response takes the sum of the absolute values of four sets of four
 * points. In each set of four points, opposite points are summed together,
 * while neighbor points (not directly adjacent but to their sides) are
 * subtracted from these sums. High sum responses result in higher confidence of
 * a chessboard corner.
 *
 * The diff response takes the sum of the absolute difference between opposite
 * samples. In a chessboard, the pixel values of opposite pixels are likely to
 * be the same; therefore, this value should be as close to zero as possible.
 *
 * The mean response is the absolute difference between the average of all
 * points around the circle and the average of the five points surrounding the
 * center. This helps mitigate against stripes that score high on sum response
 * and low on diff response.
 *
 * Assumes the image is an unsigned char grayscale image and r and c are within
 * the bounds of the image.
 *
 * @param image     An image of type CV_8UC1
 * @param r         The row to retrieve the pixel form
 * @param c         The column to retrieve the pixel form
 * @return          The overall response of the pixel at row r and column c
 */
static double calcResponse(const cv::Mat &image, int r, int c) {
  // note that we can assume all points on the circle will be within the image;
  // we do not sample pixels on the edges and corners of the image (it's only an
  // 11 x 11 area anyways)
  double sumRes = 0.0;
  double diffRes = 0.0;
  double meanNeighbors = 0.0;

  // iterate over all offsets (points around the circle)
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

  // average of the center pixels
  double meanLocal = (getDouble(image, r, c) + getDouble(image, r - 1, c) +
                      getDouble(image, r, c + 1) + getDouble(image, r + 1, c) +
                      getDouble(image, r, c - 1)) /
                     5.0;

  // corners usually have lighter tones close to their centers; stripes are more
  // solid throughout
  double meanRes = std::abs(meanNeighbors - meanLocal);

  // factor of sixteen used to ensure a zero response (more details in paper)
  return sumRes - diffRes - 16 * meanRes;
}

namespace ChessHelper {

/**
 * Execute the sampling algorithm based on Chess-board Extraction by Subtraction
 * and Summation. Before sampling, a Gaussian blur is used to preprocess the
 * image to remove noise. After sampling, the response is normalized to
 * [0.0, 1.0] in order to apply gamma correction.
 *
 * @param image     A BGR image of type CV_8UC3
 * @param ksize     The kernel size of the Gaussian blur
 * @param sigma     The scale of the Gaussian blur
 * @return          The response image of type CV_64FC1
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

/**
 * Discovers the center of the corners using the response from the sampling
 * step. A dilation is applied to keep the max responses within a neighbor.
 * Gamma correction is then used to accentuate high responses while lowering low
 * ones. Ideally, high-confidence corner candidates stay above a certain
 * threshold.
 *
 * After these first few steps, corner candidates are usually connected
 * (adjacent) to one another. Connected components are calculated along with
 * their centroids at their average position.
 *
 * To find the the four outer-points of the board (not including the edge
 * corners), the maximum quadrilateral is taken using the convex hull of the
 * remaining candidate points. This is approximated using the
 * Ramer-Douglas–Peucker algorithm (as implemented in approxPolyDP).
 *
 * Assumes the response is a double-precision floating-point grayscale image
 * normalized to [0.0, 1.0].
 *
 * @param response      A grayscale image of type CV_64FC1
 * @return              An Optional containing the four outer-most points if
 *                          found.
 */
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

  // num includes background; there should be AT LEAST 4 points along with the
  // background label
  if (num < 5) {
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

  // approximate the maximum area quadrilateral using the Ramer-Douglas–Peucker
  // algorithm; bigger epsilons include more points, while smaller epsilons
  // include less
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
