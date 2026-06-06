#pragma once

#include <ChessHelper/utils.h>

#include <opencv2/opencv.hpp>

#include <vector>

namespace ChessHelper {

/**
 * Defines an alias named Mapping that maps a coordinate to another. Mainly used
 * for reusing the corner detection logic on four quadrants.
 */
using Mapping = std::function<cv::Point2i(const cv::Mat &, int, int)>;

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
cv::Mat subSumSample(const cv::Mat &image, int ksize = 5, double sigma = 1.02);

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
 *							found.
 */
Optional<std::vector<cv::Point>> centerCorners(const cv::Mat &response);

/**
 * Defines the mapping for finding points in the top-left section of an image.
 * Because the original algorithm starts in the top-left, the coordinates go
 * unchanged.
 *
 * Assumes r and c lie within the bounds of the image.
 *
 * @param image		The image to attempting to map to.
 * @param r			The original row coordinate.
 * @param c			The original column coordinate.
 * @return			The original coordinates as a Point2i.
 */
static inline cv::Point2i topLeft(const cv::Mat &image, int r, int c) {
  cv::Point2i mapping = {r, c};
  return mapping;
}

/**
 * Defines the mapping for finding points in the top-right section of an image.
 * Column coordinates are switched to their complements with respect to the
 * column size.
 *
 * Assumes r and c lie within the bounds of the image.
 *
 * @param image		The image to attempting to map to.
 * @param r			The original row coordinate.
 * @param c			The original column coordinate.
 * @return			The coordinates with the newly mapped column.
 */
static inline cv::Point2i topRight(const cv::Mat &image, int r, int c) {
  cv::Point2i mapping = {r, image.cols - c - 1};
  return mapping;
}

/**
 * Defines the mapping for finding points in the bottom-left section of an
 * image. Row coordinates are switched to their complements with respect to
 * the row size.
 *
 * Assumes r and c lie within the bounds of the image.
 *
 * @param image		The image to attempting to map to.
 * @param r			The original row coordinate.
 * @param c			The original column coordinate.
 * @return			The coordinates with the newly mapped row.
 */
static inline cv::Point2i botLeft(const cv::Mat &image, int r, int c) {
  cv::Point2i mapping = {image.rows - r - 1, c};
  return mapping;
}

/**
 * Defines the mapping for finding points in the bottom-right section of an
 * image. Bot row and column coordinates are switched to their complements.
 *
 * Assumes r and c lie within the bounds of the image.
 *
 * @param image		The image to attempting to map to.
 * @param r			The original row coordinate.
 * @param c			The original column coordinate.
 * @return			The newly mapped coordinates.
 */
static inline cv::Point2i botRight(const cv::Mat &image, int r, int c) {
  cv::Point2i mapping = {image.rows - r - 1, image.cols - c - 1};
  return mapping;
}

/**
 * Computes the absolute value of the hyperbolic metric. Higher values score
 * better (maximizes distance in one axis while minimizes distance in the
 * other).
 *
 * @param r			The row coordinate of the pixel to measure.
 * @param c			The column coordinate of the pixel to measure.
 * @param targetR	The target row coordinate to measure against.
 * @param targetC	The target column coordinate to measure against.
 * @return			The hyperbolic metric as an integer.
 */
static inline int hyperbolic(int r, int c, int targetR, int targetC) {
  int distR = std::abs(r - targetR);
  int distC = std::abs(c - targetC);
  return std::abs(distR - distC);
}

/**
 * Finds the outer-most corners using the binary mask produced by corner
 * detection. Searches in four orientations (top-left, top-right, bottom-left,
 * bottom-right) to find the four outer-most corners of the chess board.
 *
 * This function is an old implementation for corner discovery used in the
 * command-line interface of the software. The latest iteration, which uses the
 * subSumSample and centerCorners functions, is implemented in the video
 * interface instead.
 *
 * @param mask		The binary mask image produced by Harris corner
 *						detection to search within.
 * @return			An Optional containing a vector of the four
 *						outer-most points if found.
 */
Optional<std::vector<cv::Point2i>> findCorners(cv::Mat &mask);
} // namespace ChessHelper