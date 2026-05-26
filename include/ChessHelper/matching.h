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
   */
  Optional<std::pair<ChessPiece, ChessColor>>
  identifyPiece(const cv::Mat &image) const;

  /**
   * Extracts the shapes of every chess piece and saves them
   * to a calibration file, to be used for later identification.
   * @param allPieces is a list of images of different chess pieces.
   *                  Array values are in the same order as ChessPiece
   *                  (King, Queen, Rook, Bishop, Knight, Pawn).
   *                  Each image must be a 1x1 8-bit grayscale CV_U8 image.
   */
  void calibrate(const cv::Mat allPieces[NUM_PIECE_TYPES]);

  /**
   * Runs calibration for a starting chess board.
   * The input image must be a square grayscale image
   * (CV_8U) containing only the entire chessboard, with pieces
   * arranged in a default chess configuration.
   * @param image is the chess board image to calibrate with.
   */
  void calibrate(const cv::Mat &image);

private:
  bool calibrated = false;

  /**
   * Undefined if calibrated == false.
   */
  float histogramsByPiece[NUM_PIECE_TYPES][MATCH_HISTOGRAM_BINS];
};
} // namespace ChessHelper