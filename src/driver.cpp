#include "ChessHelper/matching.h"

#include <iostream>
#include <limits>
#include <vector>

#include <opencv2/opencv.hpp>

#include "ChessHelper/utils.h"

#include <numeric>

const int NUM_DOWNSAMPLES = 2;
const int MARGIN = 0; // in pixels

// -- Keybindings --
const int KEY_CALIBRATE = 'c';
const int KEY_ANALYZE = ' ';
const int KEY_QUIT = 27; //this means 'esc' key btw

namespace ch = ChessHelper;

void calibrateBoard(const cv::Mat &currFrame, 
                    ch::PieceIdentifier &pid, 
                    cv::Mat& M, 
                    cv::Mat &arrowOverlay);

void analyzeBoard(const cv::Mat &currFrame, const ch::PieceIdentifier &pid,
                  const cv::Mat &M, cv::Mat &arrowOverlay);

int main() {
  // -- setup --
  cv::VideoCapture videoCap(0);
  if (!videoCap.isOpened()) {
    std::cerr << "Could not open your camera." << std::endl;
    return EXIT_FAILURE;
  }

  cv::namedWindow("Chess Cheater 9000",
                  cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
  ch::PieceIdentifier pieceID("./calibration");
  cv::Mat M;
  cv::Mat arrowOverlay;
  cv::Mat currFrame;

  while (true) {
    videoCap >> currFrame;
    if (currFrame.empty()) {
      std::cerr << "Frame missed in loop" << std::endl;
      break;
    }

    // clone the frame so we can use the original for processing and the clone
    // for display (with chess move arrow)
    cv::Mat displayWithArrow = currFrame.clone();

    if (!arrowOverlay.empty()) {
      // TODO: draw the arrow overlay on top of the display frame.
    }

    cv::imshow("Chess Cheater 9000", displayWithArrow);

    int key = cv::waitKey(1);
    if (key == KEY_QUIT) {
      break;
    } else if (key == KEY_CALIBRATE) {
      calibrateBoard(currFrame, pieceID, M, arrowOverlay);
    } else if (key == KEY_ANALYZE) {
      analyzeBoard(currFrame, pieceID, M, arrowOverlay);
    }
  }

  videoCap.release();
  cv::destroyAllWindows();

  return EXIT_SUCCESS;
}

// Gabe's logic for calibrating the board. This will be called when the user
// presses the calibrate key, and it will find the corners of the chess board
// and warp the perspective so that we have a top-down view of the board, which
// is necessary for piece identification.
void calibrateBoard(const cv::Mat &currFrame, 
                    ch::PieceIdentifier &pid, 
                    cv::Mat& M, 
                    cv::Mat& arrowOverlay) {
  cv::Mat grayOriginal;
  cv::cvtColor(currFrame, grayOriginal, cv::COLOR_BGR2GRAY);

  // found that downsampling is pretty quick and can help with speeding up
  // computation
  cv::Mat grayDownscaled = grayOriginal.clone();
  for (int i = 0; i < NUM_DOWNSAMPLES; i++) {
    cv::pyrDown(grayDownscaled, grayDownscaled);
  }

  // use float for harris corner detection; needs to be normalized to work
  // properly with cv::cornerHarris
  grayDownscaled.convertTo(grayDownscaled, CV_32F, 1.0 / 255.0);

  // returns CV_32FC1 (32-bit float with one color channel)
  cv::Mat corners =
      cv::Mat::zeros(grayDownscaled.size(), grayDownscaled.type());
  cv::cornerHarris(grayDownscaled, corners, 3, 7, 0.05);

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
    std::cout << "Calibration failed: Could not find corners" << std::endl;
    return;
  }

  // we need these points as a float for the perspective transform
  std::vector<cv::Point2f> points(outer.first.begin(), outer.first.end());
  // This needs to be remapped back to the originally sized
  // image so we have enough info for piece identification.
  std::cout << (static_cast<float>(grayOriginal.cols) / grayDownscaled.cols)
            << std::endl;
  for (auto &point : points) {
    point.x *= static_cast<float>(grayOriginal.cols) / grayDownscaled.cols;
    point.y *= static_cast<float>(grayOriginal.rows) / grayDownscaled.rows;
  }

  std::cout << "Found corners: " << points[0];
  for (int i = 1; i < points.size(); i++) {
    std::cout << ", " << points[i];
  }
  std::cout << std::endl;

  // must match source point order (TL, TR, BL, BR)
  std::vector<cv::Point2f> destination = {
      {MARGIN, MARGIN},
      {MARGIN, static_cast<float>(grayOriginal.cols - MARGIN - 1)},
      {static_cast<float>(grayOriginal.rows - MARGIN - 1), MARGIN},
      {static_cast<float>(grayOriginal.cols - MARGIN - 1),
       static_cast<float>(grayOriginal.cols - MARGIN - 1)}};

  M = cv::getPerspectiveTransform(points, destination);
  // This is CV_8U (because grayOrigial is CV_8U).
  cv::Mat warped;
  cv::warpPerspective(grayOriginal, warped, M, grayOriginal.size());

  // The image might not be perfectly square (off by a pixel).
  int smallAxis = std::min(warped.cols - 1, warped.rows - 1);
  warped = warped(cv::Rect(0, 0, smallAxis, smallAxis));

  pid.calibrate(warped);

  // if there was an arrow drawn previously then it was drawn for a now outdated
  // board state, so we need to clear it
  arrowOverlay.release();

  std::cout << "Calibration complete." << std::endl;
}


// Jonah's Piece identification logic will walk the grid and determine pieces
// and then feed the chess engine and determine what move to make. Then, draw
// the move on the flattened board.
void analyzeBoard(const cv::Mat& currFrame,
                  const ch::PieceIdentifier& pid,
                  const cv::Mat& M,
                  cv::Mat& arrowOverlay) {
  if (!pid.isCalibrated()) {
    std::cout << "Cannot analyze board: Select 'c' to calibrate first." << std::endl;
    return;
  }

  cv::Mat gray;
  cv::cvtColor(currFrame, gray, cv::COLOR_BGR2GRAY);


}