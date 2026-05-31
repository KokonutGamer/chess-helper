#include "matching.h"
#include "utils.h"

#include <string>
namespace ChessHelper {

/**
 * PIECE_TO_FEN[ChessPiece] gives the FEN char encoding for a piece.
 * This returns uppercase (white pieces), if the piece's color is black,
 * this needs to be converted to lowercase.
 */
constexpr char PIECE_TO_FEN[NUM_PIECE_TYPES] = {'K', 'Q', 'R', 'B', 'N', 'P'};

/**
 * Converts a board to a FEN string
 * @param board is the board, formatted as board[row][column], with 8x8=64 total
 *              cells.
 * @param activeColor should be 'b' (black) or 'w' (white), and dictates which
 *                    color the chess engine tries to find the next move for.
 * @return the full FEN string that can be sent to the chess engine.
 */
std::string fenEncode(std::vector<std::vector<Optional<std::pair<ChessPiece, ChessColor>>>> board, char activeColor);

/**
 * Sends a request to the chess engine API and returns the best move it finds.
 * @param board is the board, formatted as board[row][column], with 8x8=64 total
 * cells.
 * @param activeColor should be 'b' (black) or 'w' (white), and dictates which
 *                    color the chess engine tries to find the next move for.
 * @return the best move as [fromRow, fromCol, toRow, toCol] (or empty if an error occurred).
 */
Optional<std::array<int, 4>> findMove(std::vector<std::vector<Optional<std::pair<ChessPiece, ChessColor>>>> board, char activeColor);
}