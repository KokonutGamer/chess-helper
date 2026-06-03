#include "ChessHelper/corners.h"

static double getMedian(std::vector<double> &src) {
  if (src.empty()) {
    return 0.0;
  }

  auto target = src.begin() + src.size() / 2;
  std::nth_element(src.begin(), target, src.end()); // quickselect

  double median = *target;
  if (src.size() % 2 == 0) {
    median += *(--target);
    median /= 2.0;
  }
  return median;
}

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

/**
 * TODO document
 */
cv::Subdiv2D delaunay(cv::Mat &image, const std::vector<cv::Point2f> &points,
                      const cv::Scalar &color) {
  cv::Rect boundingBox(0, 0, image.cols, image.rows);
  cv::Subdiv2D subdiv(boundingBox);

  if (points.empty()) {
    return subdiv;
  }

  // insert points into subdivision
  for (const auto &p : points) {
    if (boundingBox.contains(p)) {
      subdiv.insert(p);
    }
  }

  // extract triangle list
  std::vector<cv::Vec6f> triangleList;
  subdiv.getTriangleList(triangleList);

  for (const auto &t : triangleList) {
    cv::Point pt1(cvRound(t[0]), cvRound(t[1]));
    cv::Point pt2(cvRound(t[2]), cvRound(t[3]));
    cv::Point pt3(cvRound(t[4]), cvRound(t[5]));

    // Subdiv2D algorithm creates a "super-triangle" outside the bounding box to
    // compute the triangulation; we must filter out any triangles that connect
    // to it
    if (boundingBox.contains(pt1) && boundingBox.contains(pt2) &&
        boundingBox.contains(pt3)) {
      cv::line(image, pt1, pt2, color, 1, cv::LINE_AA);
      cv::line(image, pt2, pt3, color, 1, cv::LINE_AA);
      cv::line(image, pt3, pt1, color, 1, cv::LINE_AA);
    }
  }

  return subdiv;
}

std::vector<cv::Point2f>
filterVertices(const cv::Subdiv2D &subdiv,
               const std::vector<cv::Point2f> &points) {
  std::vector<cv::Vec4f> edges;
  subdiv.getEdgeList(edges);

  // calculate all edge lengths
  std::vector<double> edgeLengths;
  edgeLengths.reserve(edges.size());
  for (const auto &edge : edges) {
    cv::Vec2f p1(edge[0], edge[1]);
    cv::Vec2f p2(edge[2], edge[3]);
    edgeLengths.push_back(cv::norm(p1 - p2));
  }
  if (edgeLengths.empty()) {
    return std::vector<cv::Point2f>{0};
  }
  double median = getMedian(edgeLengths);
  std::vector<double> absdev;

  cv::absdiff(edgeLengths, cv::Scalar(median), absdev);

  double mad = getMedian(absdev);

  std::vector<bool> flagged(edgeLengths.size(), false);
  for (int i = 0; i < edgeLengths.size(); i++) {
    if (std::abs(median - edgeLengths[i]) > 3 * mad) {
      flagged[i] = true;
    }
  }

  std::vector<VertexVote> votes(points.size());

  // reverse lookup map
  std::map<cv::Point2f, int, Point2fCompare> pointToIndex;
  for (int i = 0; i < points.size(); i++) {
    pointToIndex[points[i]] = i;
  }

  // iterate through edges and distribute votes
  for (size_t i = 0; i < edges.size(); i++) {
    cv::Point2f p1(edges[i][0], edges[i][1]);
    cv::Point2f p2(edges[i][2], edges[i][3]);

    // ignore phantom bounding-box points
    if (pointToIndex.count(p1) == 0 || pointToIndex.count(p2) == 0) {
      continue;
    }

    int idx1 = pointToIndex[p1];
    int idx2 = pointToIndex[p2];

    // valid edge connection increases total incident count
    votes[idx1].totalIncidentEdges++;
    votes[idx2].totalIncidentEdges++;

    // edge flagged as anomalous, increase bad edge count
    if (flagged[i]) {
      votes[idx1].flaggedEdges++;
      votes[idx2].flaggedEdges++;
    }
  }

  // filter vertices based on vote ratio
  std::vector<cv::Point2f> res;
  res.reserve(points.size());

  for (int i = 0; i < points.size(); i++) {
    if (votes[i].getAnomalyRatio() <= 0.25f) {
      res.push_back(points[i]);
    }
  }

  return res;
}

void drawPoints(cv::Mat &image, std::vector<cv::Point2f> &points,
                const cv::Vec3b &color) {
  if (image.empty()) {
    throw std::runtime_error("Image must not be empty.");
  }

  if (image.type() != CV_8UC3) {
    throw std::runtime_error("Image must be BGR.");
  }

  cv::Mat mask(image.size(), CV_8U);

  for (const auto &point : points) {
    mask.at<unsigned char>(point.y, point.x) = 255;
  }

  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9));
  cv::Mat dest;
  cv::dilate(mask, dest, kernel);

  for (int i = 0; i < image.rows; i++) {
    for (int j = 0; j < image.cols; j++) {
      if (dest.at<unsigned char>(i, j) == 255) {
        image.at<cv::Vec3b>(i, j) = color;
      }
    }
  }
}

} // namespace ChessHelper