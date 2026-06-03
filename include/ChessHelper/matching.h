#pragma once

#include <opencv2/opencv.hpp>

#include "ChessHelper/utils.h"

namespace ChessHelper {

enum class ChessPiece {
  King = 0,
  Queen,
  Rook,
  Bishop,
  Knight,
  Pawn,
};

enum class ChessColor {
  White = 0,
  Black,
};

constexpr int CELLS_PER_SIDE = 8;

constexpr int NUM_PIECE_TYPES = 6;

constexpr ChessPiece PIECE_TYPES[NUM_PIECE_TYPES] = {
    ChessPiece::King,   ChessPiece::Queen,  ChessPiece::Rook,
    ChessPiece::Bishop, ChessPiece::Knight, ChessPiece::Pawn};

/**
 * How many bins we'll divide the histogram of
 * tangent line angles.
 * Changing this requires recalibration!
 */
constexpr int MATCH_HISTOGRAM_BINS = 32;

class PieceIdentifier {
public:
  /**
   * Creates a new chess piece identifier and loads
   * calibration data from the calibration directory.
   * @param calibrationDir is the directory to store/load
   *                       calibration data.
   */
  PieceIdentifier(std::string calibrationDir);

  /**
   * Returns whether the identifier has been calibrated.
   * This will be true if all the calibration data it needs
   * is available.
   */
  bool isCalibrated() const;

  /**
   * Determines what type of chess piece the given image contains.
   * Returns none if the piece cannot be determined, or if the
   * identifier isn't calibrated.
   * @param image is a square RGB (CV_8UC3) image containing only an entire
   *              cell.
   */
  Optional<std::pair<ChessPiece, ChessColor>>
  identifyPiece(const cv::Mat &image) const;

  /**
   * Runs piece identification on every cell in the chess board.
   * The input image must be a square RGB image
   * (CV_8UC3) containing only the entire chessboard, with pieces
   * arranged in a default chess configuration.
   * @param image is the chess board image to run identification on.
   * @return is the output board, in board[row][column] order, where empty
   *         values represent no piece detected.
   */
  std::vector<std::vector<Optional<std::pair<ChessPiece, ChessColor>>>>
  identifyBoard(const cv::Mat &image) const;

  /**
   * Runs calibration for a starting chess board ad saves the
   * data to a file.
   * The input image must be a square RGB image
   * (CV_8UC3) containing only the entire chessboard, with pieces
   * arranged in a default chess configuration.
   * @param image is the chess board image to calibrate with.
   */
  void calibrate(const cv::Mat &image);

  /**
   * Slices a square RGB image (CV_8UC3) containing only
   * the entire chessboard into individual images for each cell.
   * @param image is the chess board image to slice.
   * @return a nested array of board cell images, organized
   *         as images[row][col].
   */
  static std::vector<std::vector<cv::Mat>> sliceBoard(const cv::Mat &image);

private:
  /**
   * Loads the data stored in the calibration directory (if it exists).
   * Skips loading if no data exists.
   * Aborts the program on failure.
   */
  void loadData();

  /**
   * Saves the calibration data into the calibration directory,
   * creating it if it doesn't already exist.
   * Aborts the program on failure, or if the identifier
   * isn't already calibrated.
   */
  void saveData() const;

  /**
   * Extracts the shapes of every chess piece and stores
   * them for later identification (does not mark calibrated as true).
   * @param allPieces is a list of images of different chess pieces.
   *                  Array values are in the same order as ChessPiece
   *                  (King, Queen, Rook, Bishop, Knight, Pawn).
   *                  Each image must be a square 8-bit RGB CV_8UC3 image.
   */
  void calibrateShape(const cv::Mat allPieces[NUM_PIECE_TYPES]);

  /**
   * Extracts the average color from a white and black piece and
   * stores them for later identification (does not mark calibrated as true).
   * Both input images must be square 8-bit RGB CV_8UC3 images.
   * @param white is an image of the white piece's cell.
   * @param black is an image of the black piece's cell.
   */
  void calibrateColor(const cv::Mat &white, const cv::Mat &black);

  std::string calibrationDir;

  bool calibrated = false;

  /**
   * Undefined if calibrated == false.
   */
  float histogramsByPiece[NUM_PIECE_TYPES][MATCH_HISTOGRAM_BINS]{};

  /**
   * The mean color found for black pieces during calibration (RGB [0, 255]).
   */
  int blackColor[3]{};

  /**
   * The mean color found for white pieces during calibration (RGB [0, 255]).
   */
  int whiteColor[3]{};
};
} // namespace ChessHelper