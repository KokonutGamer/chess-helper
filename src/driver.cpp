#include "ChessHelper/engine.h"
#include "ChessHelper/matching.h"

#include <iostream>
#include <limits>
#include <vector>

#include <opencv2/opencv.hpp>

#include "ChessHelper/corners.h"
#include "ChessHelper/utils.h"

#include <numeric>

constexpr int NUM_DOWNSAMPLES = 2;
constexpr int MARGIN = 0; // in pixels

// -- Keybindings --
constexpr int KEY_SETUP = 's';
constexpr int KEY_ANALYZE = ' ';
constexpr int KEY_QUIT = 27;   // this means 'esc' key btw
constexpr int KEY_DEBUG = 100; // lowercase 'd'

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
                   "https://scriptchess.com/tools/fen-visualizer to visualize."
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
  static bool debug = false;

  // -- setup --
  cv::VideoCapture videoCap(0);
  if (!videoCap.isOpened()) {
    std::cerr << "Could not open camera." << std::endl;
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
    cv::Mat display = currFrame.clone();

    if (!arrowOverlay.empty()) {
      cv::Mat warpedArrow(arrowOverlay.size(), arrowOverlay.type());

      cv::warpPerspective(
          arrowOverlay, warpedArrow, M, arrowOverlay.size(),
          // We need nearest to prevent it from interpolating
          // alpha values, since the code below only handles binary
          // visible vs invisible.
          // We also need to invert the transformation, since we draw
          // the arrow in the warped space and want to display it onto
          // the original camera input.
          cv::INTER_NEAREST | cv::WARP_INVERSE_MAP);

      // Extract the alpha channel, which will only be 0 or 255,
      // so that we can selectively draw arrow items from it.
      cv::Mat mask;
      cv::extractChannel(warpedArrow, mask, 3);

      // Drop the alpha channel, since display doesn't have one.
      cv::cvtColor(warpedArrow, warpedArrow, cv::COLOR_BGRA2BGR);

      cv::copyTo(warpedArrow, display, mask);
    }

    cv::imshow("Chess Cheater 9000", display);

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
        // BGRA
        arrowOverlay = cv::Mat::zeros(currFrame.size(), CV_8UC4);

        // TODO: Maybe draw corner points onto arrowOverlay using inverse(M)?
      } else {
        std::cerr << "Could not setup board (not all corners could be found)."
                  << std::endl;
      }
    } else if (key == KEY_ANALYZE) {
      if (M.empty()) {
        std::cerr << "Cannot analyze board: press 's' first to find corners."
                  << std::endl;
      } else {
        analyzeBoard(currFrame, pieceID, M, arrowOverlay);
      }
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
  cv::Mat response = ch::sample(image);
  ch::Optional<std::vector<cv::Point>> outer = ch::centerCorners(response);

  if (!outer.second) {
    std::cout << "Calibration failed: Could not find corners" << std::endl;
    return ch::empty<cv::Mat>();
  }

  // we need these points as a float for the perspective transform
  std::vector<cv::Point2f> points(outer.first.begin(), outer.first.end());

  std::cout << "Found corners: " << points[0];
  for (int i = 1; i < points.size(); i++) {
    std::cout << ", " << points[i];
  }
  std::cout << std::endl;

  // must match source point order (TL, BL, BR, TR)
  float size = static_cast<float>(std::min(image.rows, image.cols));
  float margin = size * 0.125f;
  std::vector<cv::Point2f> destination = {
      {size - margin - 1.0f, size - margin - 1.0f},
      {margin, size - margin - 1.0f},
      {margin, margin},
      {size - margin - 1.0f, margin}};

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
  auto board = pid.identifyBoard(warped);

  for (int row = 0; row < ch::CELLS_PER_SIDE; row++) {
    for (int col = 0; col < ch::CELLS_PER_SIDE; col++) {
      auto piece = board[row][col];
      if (!piece.second) {
        continue;
      }

      auto pieceChar = ch::PIECE_TO_FEN[static_cast<int>(piece.first.first)];
      std::string pieceText;
      pieceText += pieceChar;

      cv::putText(
          arrowOverlay, pieceText,
          // I had to do a slight offset or else the text would overflow and get
          // clipped.
          cv::Point(10 + col * warped.cols / ChessHelper::CELLS_PER_SIDE,
                    50 + row * warped.rows / ChessHelper::CELLS_PER_SIDE),
          cv::FONT_HERSHEY_DUPLEX, 1.0, cv::Scalar(20, 20, 255, 255), 4);
    }
  }

  // TODO: Allow changing color?
  auto bestMove = ch::findMove(board, 'w');
  if (!bestMove.second) {
    std::cerr << "Could not find a move." << std::endl;
    return;
  }

  std::cout << "Best move: " << bestMove.first[0] << std::endl;
}