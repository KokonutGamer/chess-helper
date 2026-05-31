#include "ChessHelper/engine.h"
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
const int KEY_SETUP = 's';
const int KEY_ANALYZE = ' ';
const int KEY_QUIT = 27; // this means 'esc' key btw

namespace ch = ChessHelper;

ch::Optional<cv::Mat> setupBoard(const cv::Mat &image);

void analyzeBoard(const cv::Mat &image, const ch::PieceIdentifier &pid,
                  const cv::Mat &M, cv::Mat &arrowOverlay);

void videoInterface();

void commandInterface();

int main() {
  while (true) {
    std::string selection;

    std::cout << "Type 'v' to access the video interface (requires a live "
                 "videocamera) or 'c' to access the command-line interface:"
              << std::endl;
    std::cin >> selection;

    if (selection == "v") {
      videoInterface();
    } else if (selection == "c") {
      commandInterface();
    } else {
      // Invalid selection.
      continue;
    }

    // The program is done.
    return EXIT_SUCCESS;
  }
}

/**
 * Gives the user a command-line menu to try
 * the program's functionality.
 */
void commandInterface() {
  ch::PieceIdentifier pieceID("./calibration");

  while (true) {
    std::string selection;

    std::cout << "Enter a board image to load:" << std::endl;
    std::cin >> selection;

    cv::Mat currFrame = cv::imread(selection);

    auto boardRes = setupBoard(currFrame);
    if (!boardRes.second) {
      std::cerr << "Could not setup board (not all corners could be found)."
                << std::endl;
      exit(EXIT_FAILURE);
    }
    cv::Mat M = boardRes.first;

    cv::Mat warped = ch::grayWarp(currFrame, M);
    cv::imwrite("ch-warped.png", warped);
    std::cout << "Wrote warped image to ch-warped.png" << std::endl;

    std::cout << std::endl
              << "Board calibration involves loading an image of a chess board "
                 "in a starting layout so that the piece identifier can learn "
                 "to identify them."
              << std::endl;
    std::cout << "Board analysis will use the piece identifier to print out "
                 "the board's current arrangement."
              << std::endl;

    std::cout << "Type 'c' to calibrate the board, 'a' to analyze the board, "
                 "or 'q' to quit:"
              << std::endl;
    std::cin >> selection;

    if (selection == "c") {
      pieceID.calibrate(warped);

      std::cout << "Calibration complete." << std::endl;
    } else if (selection == "a") {
      auto pieces = pieceID.identifyBoard(warped);
      auto fen = ch::fenEncode(pieces, 'w');

      std::cout << "FEN: " << fen << std::endl;
      std::cout << "Use an online viewer such as "
                   "https://fujibit.live/chess/fen-viewer/ to visualize."
                << std::endl;
    } else {
      // Invalid selection.
      continue;
    }

    std::cout << "Would you like to quit (y/n):" << std::endl;
    std::cin >> selection;

    if (selection == "y") {
      break;
    }
  }
}

/**
 * Displays an interactive video feed of the board
 * and best moves.
 */
void videoInterface() {
  // -- setup --
  cv::VideoCapture videoCap(0);
  if (!videoCap.isOpened()) {
    std::cerr << "Could not open your camera." << std::endl;
    exit(EXIT_FAILURE);
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
    } else if (key == KEY_SETUP) {
      auto mat = setupBoard(currFrame);
      if (mat.second) {
        // Success.
        M = mat.first;

        // if there was an arrow drawn previously then it was drawn for a now
        // outdated board state, so we need to clear it
        arrowOverlay.release();

        // TODO: Maybe draw corner points onto arrowOverlay using inverse(M)?
      } else {
        std::cerr << "Could not setup board (not all corners could be found)."
                  << std::endl;
      }
    } else if (key == KEY_ANALYZE) {
      analyzeBoard(currFrame, pieceID, M, arrowOverlay);
    }
  }

  videoCap.release();
  cv::destroyAllWindows();
}

/**
 * Extracts the corner points from the input chess board image, and if they
 * can be found, returns a perspective transform matrix to make the image
 * contain only the entire chessboard.
 * @param image is the image to analyze.
 * @return a perspective transform matrix (or empty if one couldn't be found).
 */
ch::Optional<cv::Mat> setupBoard(const cv::Mat &image) {
  cv::Mat grayOriginal;
  cv::cvtColor(image, grayOriginal, cv::COLOR_BGR2GRAY);

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
    return ch::empty<cv::Mat>();
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

  return ch::value(cv::getPerspectiveTransform(points, destination));
}

/**
 * Identifies all the pieces on the board, sends them to the chess engine,
 * and draws an arrow to indicate the move on the screen.
 * If the piece identifier isn't already calibrated, this will abort the
 * program.
 * @param image is the image to identify pieces on. It should be the same
 *              image the perspective transform was extracted from (i.e., not
 *              warped).
 * @param pid is the piece identifier to use (must be calibrated).
 * @param M is the perspective transform matrix to correct the input image.
 * @param arrowOverlay is an image with the same shape as `image`, and which the
 *                     arrow will be drawn into by this function.
 */
void analyzeBoard(const cv::Mat &image, const ch::PieceIdentifier &pid,
                  const cv::Mat &M, cv::Mat &arrowOverlay) {
  if (!pid.isCalibrated()) {
    std::cerr << "Cannot analyze board: enter command-line mode and calibrate "
                 "first (or check that there's a calibration folder in your "
                 "working directory)."
              << std::endl;
    exit(EXIT_FAILURE);
  }

  cv::Mat warped = ch::grayWarp(image, M);
}