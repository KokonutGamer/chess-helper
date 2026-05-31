#include "ChessHelper/engine.h"

#include "httplib.h"

namespace ch = ChessHelper;

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

ch::Optional<std::array<int, 4>> ChessHelper::findMove(
    std::vector<std::vector<Optional<std::pair<ChessPiece, ChessColor>>>> board,
    char activeColor) {
  std::string fen = fenEncode(board, activeColor);

  httplib::Client client("https://stockfish.online");
  auto res = client.Get("/api/s/v2.php?fen=" + fen + "&depth=12");
  if (!res || res->status != 200) {
    return ch::empty<std::array<int, 4>>();
  }

  // Lazy extraction.
  // The API returns it as JSON, but we can
  // just use very unrobust string parsing to avoid
  // pulling in a JSON parser library as well.
  std::string search = "bestmove ";
  size_t searchIdx = res->body.find(search);
  if (searchIdx == std::string::npos) {
    return ch::empty<std::array<int, 4>>();
  }

  // Formatted like "bestmove f6e4"
  // ['a', 'f'] is a column, [1, 8] is a row.
  int fromCol = static_cast<int>(res->body[searchIdx + search.size()] - 'a');
  int fromRow =
      static_cast<int>(res->body[searchIdx + search.size() + 1] - '1');
  int toCol = static_cast<int>(res->body[searchIdx + search.size() + 2] - 'a');
  int toRow = static_cast<int>(res->body[searchIdx + search.size() + 3] - '1');

  std::array<int, 4> move = {fromRow, fromCol, toRow, toCol};
  return value(move);
}