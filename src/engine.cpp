#include "ChessHelper/engine.h"

#include <fstream>
#include <regex>

namespace ch = ChessHelper;

/**
 * Converts a board to a FEN string.
 *
 * @param board         The board, formatted as board[row][column], with 8x8=64
 *                          total cells.
 * @param activeColor   The color ('b'  for black or 'w' for white) which
 *                          dictates which color the chess engine tries to find
 *                          the next move for.
 * @return              The full FEN string that can be sent to the chess
 *                          engine.
 */
std::string ChessHelper::fenEncode(
    std::vector<std::vector<Optional<std::pair<ChessPiece, ChessColor>>>> board,
    char activeColor) {
  std::string result;

  for (int row = 0; row < CELLS_PER_SIDE; row++) {
    for (int col = 0; col < CELLS_PER_SIDE; col++) {
      auto cell = board[row][col];
      if (!cell.second) {
        // No piece.
        // Technically, FEN encodes runs of
        // empty pieces, but I found that the
        // API is fine to just use a "1" for each
        // missing piece, which simplifies this code.
        result += "1";
        continue;
      }

      char pieceChar = PIECE_TO_FEN[static_cast<int>(cell.first.first)];
      // pieceChar is uppercase (white pieces), and needs to be
      // modified for black pieces.
      if (cell.first.second == ChessColor::Black) {
        pieceChar = std::tolower(pieceChar);
      }

      result += pieceChar;
    }

    if (row != CELLS_PER_SIDE - 1) {
      result += "/";
    }
  }

  result += " ";
  result += activeColor;
  // Dummy values.
  result += " - - 1 1";

  return result;
}

/**
 * Sends a request to the chess engine API and returns the best move it finds.
 *
 * @param board         The board, formatted as board[row][column], with 8x8=64
 *                          total cells.
 * @param activeColor   The color ('b'  for black or 'w' for white) which
 *                          dictates which color the chess engine tries to find
 *                          the next move for.
 * @return              The best move as [fromRow, fromCol, toRow, toCol] (or
 *                          empty if an error occurred).
 */
ch::Optional<std::array<int, 4>> ChessHelper::findMove(
    std::vector<std::vector<Optional<std::pair<ChessPiece, ChessColor>>>> board,
    char activeColor) {
  std::string fen = fenEncode(board, activeColor);
  // Spaces need to be URL-encoded for curl to accept them.
  fen = std::regex_replace(fen, std::regex(" "), "%20");

  // I honestly couldn't find a better way to do this that didn't
  // involve pulling in additional libraries.
  // Based on https://en.cppreference.com/cpp/utility/program/system.
  int ret = system(("curl \"https://stockfish.online/api/s/v2.php?fen=" + fen +
                    "&depth=12\" -o temp.txt")
                       .c_str());
  if (ret != 0) {
    return ch::empty<std::array<int, 4>>();
  }

  std::ifstream file("temp.txt");
  std::string res;
  getline(file, res);

  // Lazy extraction.
  // The API returns it as JSON, but we can
  // just use very unrobust string parsing to avoid
  // pulling in a JSON parser library as well.
  std::string search = "bestmove ";
  size_t searchIdx = res.find(search);
  if (searchIdx == std::string::npos) {
    return ch::empty<std::array<int, 4>>();
  }

  // Formatted like "bestmove f6e4"
  // ['a', 'f'] is a column, [1, 8] is a row.
  int fromCol = static_cast<int>(res[searchIdx + search.size()] - 'a');
  int fromRow = static_cast<int>(res[searchIdx + search.size() + 1] - '1');
  int toCol = static_cast<int>(res[searchIdx + search.size() + 2] - 'a');
  int toRow = static_cast<int>(res[searchIdx + search.size() + 3] - '1');

  std::array<int, 4> move = {fromRow, fromCol, toRow, toCol};
  return value(move);
}