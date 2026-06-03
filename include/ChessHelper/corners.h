#pragma once

#include <opencv2/opencv.hpp>

#include <vector>

namespace ChessHelper {

struct Point2fCompare {
  bool operator()(const cv::Point2f &a, const cv::Point2f &b) const {
    const float EPSILON = 1e-5f;
    if (std::abs(a.x - b.x) > EPSILON)
      return a.x < b.x;
    if (std::abs(a.y - b.y) > EPSILON)
      return a.y < b.y;
    return false;
  }
};

struct VertexVote {
  int totalIncidentEdges = 0;
  int flaggedEdges = 0;

  float getAnomalyRatio() const {
    if (totalIncidentEdges == 0) {
      return 0.0f;
    }
    return static_cast<float>(flaggedEdges) / totalIncidentEdges;
  }
};

/**
 * TODO document
 */
cv::Mat detectCorners(const cv::Mat &image);

/**
 * TODO document
 */
std::vector<cv::Point2f> collapsePoints(const cv::Mat &corners);

/**
 * TODO document
 */
cv::Subdiv2D delaunay(cv::Mat &image, const std::vector<cv::Point2f> &points,
                      const cv::Scalar &color = cv::Scalar(0, 255, 0));

/**
 * TODO document
 */
std::vector<cv::Point2f> filterVertices(const cv::Subdiv2D &subdiv,
                                        const std::vector<cv::Point2f> &points);

/**
 * TODO document
 */
void drawPoints(cv::Mat &image, std::vector<cv::Point2f> &points,
                const cv::Vec3b &color = cv::Vec3b(0, 0, 255));
} // namespace ChessHelper