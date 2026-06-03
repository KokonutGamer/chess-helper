#include "ChessHelper/matching.h"

#include <numeric>
#include <sys/stat.h>

#if defined(_WIN32)
#include <direct.h>
#endif

namespace ch = ChessHelper;

/**
 * Collects stats about a contour that can be used to
 * eliminate artifacts.
 * @param size is the size of the square image (pixels).
 * @param contour is a set of points along the contour.
 * @param center is the variable to write the center (mean) of the contour in.
 * @param spread is the variable to write the spread in ([0, 1000]),
 *               where the maximum represents a perfect circle at
 *               the center with radius = size/2.
 * @param sqrDistance is the variable to write the mean square distance
 *                    from the center in ([0, 1000]).
 */
void getContourStats(int size, const std::vector<cv::Point> &contour,
                     cv::Point2f &center, int &spread, int &sqrDistance) {
  cv::Point sum;
  sum = std::accumulate(contour.begin(), contour.end(), sum);

  center = cv::Point2f(sum.x / contour.size(), sum.y / contour.size());

  // Mean distance from the center of mass.
  spread = 0;
  for (auto &point : contour) {
    int x = point.x - center.x;
    int y = point.y - center.y;

    spread += sqrt(x * x + y * y);
  }

  spread /= contour.size();

  // We need to scale by the image size
  // to do real comparisons.
  // 1000 is arbitrary, big enough to give us
  // enough precision to threshold on.
  spread *= 1000;
  // Square image, max distance is at the corners,
  // which is sqrt(radius^2 + radius^2), where radius = width/2.
  // sqrt(2 * radius^2)
  spread /= sqrt(size * size / 2);

  sqrDistance = (center.x - size / 2) * (center.x - size / 2) +
                (center.y - size / 2) * (center.y - size / 2);
  // Same scaling as above.
  sqrDistance *= 1000;
  // 2*radius^2 is the biggest sqr distance, where radius=width/2
  sqrDistance /= size * size;
}

/**
 * Updates the histogram with information from the contour.
 * For each set of points, this calculates the angle between
 * them, normalizes [0, 180], then counts it in the histogram.
 * @param outHistogram is the histogram to write into.
 * @param histogramCount is incremented for each item added to the histogram.
 * @param contour is the contour to analyze.
 */
void collectContourHistogram(float outHistogram[ch::MATCH_HISTOGRAM_BINS],
                             int &histogramCount,
                             const std::vector<cv::Point> &contour) {
  // We need to skip forward by some amount of pixels
  // or else our vectors have essentially zero resolution,
  // since there are only 8 possible ways that two neighboring points
  // can be arranged.
  // TODO: Should this be scaled by size?
  constexpr int SAMPLING_SKIP = 7;

  for (int i = SAMPLING_SKIP; i < contour.size(); i++) {
    cv::Point prev = contour[i - SAMPLING_SKIP];
    cv::Point cur = contour[i];

    // Order doesn't matter, since we
    // use 180 degrees instead of the full
    // 360 degree range.
    cv::Point vec = prev - cur;
    if (vec.x == 0 && vec.y == 0)
      continue;

    // atan2 returns between -pi and pi, so
    // it needs to be corrected to a positive angle.
    // Then, we need to normalize to [0, 180deg].
    float angle = std::fmod(atan2(vec.y, vec.x) + 2.0 * M_PI, M_PI);

    // Remap from [0, 180deg] to [0, MATCH_HISTOGRAM_BINS - 1].
    int bin =
        static_cast<int>(floor(angle * (ch::MATCH_HISTOGRAM_BINS / M_PI)));
    bin = std::max(0, std::min(ch::MATCH_HISTOGRAM_BINS - 1, bin));

    outHistogram[bin]++;
    histogramCount++;
  }
}

/**
 * Computes various information about a cell to identify which piece
 * it contains.
 *
 * Computes a histogram of gradient angles for the chess piece
 * contained in the image, and writes it into outHistogram.
 * This will be a normalized vector.
 *
 * Determines whether there's a piece in the cell.
 *
 * The image must be a square grayscale CV_U8 image of a single
 * chess board cell.
 *
 * @param outHistogram is the histogram to write into.
 * @param hasPiece is written with whether this cell contains something (not
 *                 empty).
 * @param averageColor is written with the average color of the center of the
 *                     piece as RGB [0, 255].
 * @param image is the image to extract the shape from.
 */
void writePieceInfo(float outHistogram[ch::MATCH_HISTOGRAM_BINS],
                    bool &hasPiece, int averageColor[3], const cv::Mat &image) {
  if (image.rows != image.cols) {
    throw new std::runtime_error(
        "writeGradientHistogram received a non-square input!");
  }

  int size = image.rows;

  cv::Mat edges;

  // Parameters found experimentally.
  cv::cvtColor(image, edges, cv::COLOR_BGR2GRAY);
  cv::GaussianBlur(edges, edges, cv::Size(3, 3), 0.8);
  cv::Canny(edges, edges, 60, 200);

  // This will group lines on the chess board.
  // It could split the main shape into multiple lines,
  // and the direction along those curves are undefined.
  // Both possible directions are 180 degrees apart from
  // each other, so we can eliminate the issue by normalizing
  // the angle to always be within [0, pi].
  //
  // The other curves will be the chess piece's containing circle,
  // and other artifiacts (lighting, board lines), which we need
  // to eliminate.
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(edges, contours, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);

  memset(outHistogram, 0, sizeof(float) * ch::MATCH_HISTOGRAM_BINS);
  int histogramCount = 0;

  int totalContourPoints = 0;

  // To identify piece color, we want to find the color
  // at the center of the piece.
  // We can do that by finding the center of all the contour
  // points, and sampling there, which will hopefully not lie
  // along a contour.
  cv::Point2f averageCenter;

  for (int i = 0; i < contours.size(); i++) {
    auto &contour = contours[i];

    // Eliminate artifacts.
    int spread;
    int sqrDistance;
    cv::Point2f center;
    getContourStats(size, contour, center, spread, sqrDistance);

    // Eliminate circles, which have a large spread.
    if (spread > 400)
      continue;

    // Eliminate partial circle artifacts.
    // The chess piece itself might not be centered,
    // but it's always closer to the center, and larger
    // segments will generally cancel out, whereas circle
    // artifacts will be far and won't cancel out (since
    // they have a low spread).
    if (sqrDistance > 70)
      continue;

    // Useful for debugging the thresholds above:
    /*std::cout << "sqrDistance: " << sqrDistance << ", distance: " << spread <<
    std::endl; cv::drawContours(image, contours, i, cv::Scalar(0, 0, 255), 1,
    cv::LINE_8); cv::imshow("Canny", image); cv::waitKey(0);*/

    collectContourHistogram(outHistogram, histogramCount, contour);
    totalContourPoints += static_cast<int>(contour.size());
    averageCenter += static_cast<float>(contour.size()) * center;
  }

  // The histogram must be normalized to avoid
  // number of contour points having an influence,
  // as this could change with scale.
  if (histogramCount != 0) {
    for (int i = 0; i < ch::MATCH_HISTOGRAM_BINS; i++) {
      outHistogram[i] /= static_cast<float>(histogramCount);
    }
  }

  hasPiece = totalContourPoints > 200;

  // Collect center color for piece color identification.
  if (totalContourPoints != 0) {
    averageCenter /= static_cast<float>(totalContourPoints);

    // Sample points in a small grid to avoid noise from only sampling
    // a single pixel.
    // The image is single-channel, so we can just extract the first channel
    // value.
    auto samplePoints =
        image(cv::Rect(static_cast<int>(round(averageCenter.x - 2)),
                       static_cast<int>(round(averageCenter.y - 2)), 5, 5));

    // Useful for debugging (e.g., if it lies along a contour):
    /*cv::imshow("samplePoints", samplePoints);
    cv::waitKey(0);*/

    auto mean = cv::mean(samplePoints);
    averageColor[0] = static_cast<int>(mean[0]);
    averageColor[1] = static_cast<int>(mean[1]);
    averageColor[2] = static_cast<int>(mean[2]);
  }
}

/**
 * Computes the sum of absolute differences between two
 * piece histograms at a certain offset (rotation).
 *
 * histogram1[i + offset] is compared against histogram2.
 * @param histogram1 is one of the histograms to compare.
 * @param histogram2 is the other histogram to compare.
 * @param offset is an index offset for histogram1.
 */
float sadAtOffset(const float histogram1[ch::MATCH_HISTOGRAM_BINS],
                  const float histogram2[ch::MATCH_HISTOGRAM_BINS],
                  int offset) {
  float sum = 0;

  for (int i = 0; i < ch::MATCH_HISTOGRAM_BINS; i++) {
    sum += abs(histogram1[(i + offset) % ch::MATCH_HISTOGRAM_BINS] -
               histogram2[i]);
  }

  return sum;
}

ch::PieceIdentifier::PieceIdentifier(std::string calibrationDir) {
  this->calibrationDir = calibrationDir;
  this->loadData();
}

bool ch::PieceIdentifier::isCalibrated() const { return this->calibrated; }

/**
 * Returns the square euclidean distance of the two input 3D vectors.
 */
int sqrDistance(const int a[3], const int b[3]) {
  return (a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) +
         (a[2] - b[2]) * (a[2] - b[2]);
}

ch::Optional<std::pair<ch::ChessPiece, ch::ChessColor>>
ch::PieceIdentifier::identifyPiece(const cv::Mat &image) const {
  ChessPiece bestPiece;
  float minScore = std::numeric_limits<float>::max();

  float testHistogram[MATCH_HISTOGRAM_BINS];
  bool hasPiece;
  int averageColor[3];
  writePieceInfo(testHistogram, hasPiece, averageColor, image);

  if (!hasPiece) {
    return ch::empty<std::pair<ch::ChessPiece, ch::ChessColor>>();
  }

  for (int pieceIdx = 0; pieceIdx < NUM_PIECE_TYPES; pieceIdx++) {
    // We need to try every orientation of the piece.
    for (int offset = 0; offset < MATCH_HISTOGRAM_BINS; offset++) {
      float score =
          sadAtOffset(testHistogram, this->histogramsByPiece[pieceIdx], offset);

      if (score < minScore) {
        minScore = score;
        bestPiece = static_cast<ChessPiece>(pieceIdx);
      }
    }
  }

  // Piece color is just the closest average color.
  ChessColor color = sqrDistance(averageColor, this->whiteColor) <
                             sqrDistance(averageColor, this->blackColor)
                         ? ChessColor::White
                         : ChessColor::Black;

  return ch::value(std::make_pair(bestPiece, color));
}

std::vector<std::vector<ChessHelper::Optional<
    std::pair<ChessHelper::ChessPiece, ChessHelper::ChessColor>>>>
ChessHelper::PieceIdentifier::identifyBoard(const cv::Mat &image) const {
  auto cells = sliceBoard(image);

  std::vector<std::vector<Optional<std::pair<ChessPiece, ChessColor>>>>
      identifiedCells(CELLS_PER_SIDE,
                      std::vector<Optional<std::pair<ChessPiece, ChessColor>>>(
                          CELLS_PER_SIDE));

  for (int row = 0; row < CELLS_PER_SIDE; row++) {
    for (int col = 0; col < CELLS_PER_SIDE; col++) {
      identifiedCells[row][col] = this->identifyPiece(cells[row][col]);
    }
  }

  return identifiedCells;
}

void ChessHelper::PieceIdentifier::calibrateShape(
    const cv::Mat allPieces[NUM_PIECE_TYPES]) {
  for (int i = 0; i < NUM_PIECE_TYPES; i++) {
    bool hasPiece;
    int _averageColor[3];
    writePieceInfo(this->histogramsByPiece[i], hasPiece, _averageColor,
                   allPieces[i]);

    if (!hasPiece) {
      throw std::runtime_error("Piece " + std::to_string(i) +
                               " could not be found!");
    }
  }
}

void ChessHelper::PieceIdentifier::calibrateColor(const cv::Mat &white,
                                                  const cv::Mat &black) {
  bool hasPiece;
  float _histogram[MATCH_HISTOGRAM_BINS];

  writePieceInfo(_histogram, hasPiece, this->whiteColor, white);
  if (!hasPiece) {
    throw std::runtime_error("White piece could not be found!");
  }

  writePieceInfo(_histogram, hasPiece, this->blackColor, black);
  if (!hasPiece) {
    throw std::runtime_error("Black piece could not be found!");
  }
}

void ChessHelper::PieceIdentifier::calibrate(const cv::Mat &image) {
  auto cells = sliceBoard(image);

  cv::Mat allPieces[NUM_PIECE_TYPES] = {
      // King
      cells[0][4],
      // Queen
      cells[0][3],
      // Rook
      cells[0][0],
      // Bishop
      cells[0][2],
      // Knight
      cells[0][1],
      // Pawn, this particular one calibrates
      // well in the test image for some reason.
      cells[6][1],
  };
  this->calibrateShape(allPieces);

  // Queen (3) is the most circular, so more likely to get
  // a good calibration, and white pieces are at the bottom.
  this->calibrateColor(cells[7][3], cells[1][3]);

  this->calibrated = true;
  this->saveData();

  // For testing, print out all the pieces.
  // TODO: Remove.
  for (int i = 0; i < CELLS_PER_SIDE; i++) {
    for (int j = 0; j < CELLS_PER_SIDE; j++) {
      auto piece = this->identifyPiece(cells[i][j]);

      if (piece.second) {
        std::cout << static_cast<int>(piece.first.first);
        std::cout << (piece.first.second == ChessColor::White ? 'w' : 'b');
      } else {
        std::cout << "--";
      }
    }

    std::cout << std::endl;
  }
}

std::vector<std::vector<cv::Mat>>
ChessHelper::PieceIdentifier::sliceBoard(const cv::Mat &image) {
  if (image.rows != image.cols) {
    throw std::runtime_error("sliceBoard received a non-square input!");
  }

  // The board needs to be a multiple of CELLS_PER_SIDE
  // so that we can evenly divide it into subregions without
  // issues.
  cv::Mat board;
  cv::resize(image, board,
             cv::Size(floor(image.rows / CELLS_PER_SIDE) * CELLS_PER_SIDE,
                      floor(image.cols / CELLS_PER_SIDE) * CELLS_PER_SIDE));

  int cellSize = board.rows / CELLS_PER_SIDE;

  std::vector<std::vector<cv::Mat>> cells(CELLS_PER_SIDE,
                                          std::vector<cv::Mat>(CELLS_PER_SIDE));

  for (int row = 0; row < CELLS_PER_SIDE; row++) {
    for (int col = 0; col < CELLS_PER_SIDE; col++) {
      // Rect is (x, y)
      cells[row][col] =
          board(cv::Rect(col * cellSize, row * cellSize, cellSize, cellSize));
    }
  }

  return cells;
}

void ChessHelper::PieceIdentifier::loadData() {
  struct stat dir{};
  if (stat(this->calibrationDir.c_str(), &dir) != 0) {
    // Directory doesn't exist, nothing we can load.
    return;
  }

  std::string calibrationPath = this->calibrationDir + "/calibration.dat";
  FILE *calibrationFile = fopen(calibrationPath.c_str(), "rb");
  if (calibrationFile == nullptr) {
    // File doesn't exist.
    return;
  }

  // SAFETY: To my knowledge, a float array has the exact same
  //         layout (4 bytes per item, 4 byte alignment, no padding)
  //         across x86_64 and arm64, and Linux and Windows.
  fread(&this->histogramsByPiece, sizeof(float),
        MATCH_HISTOGRAM_BINS * NUM_PIECE_TYPES, calibrationFile);

  // SAFETY: int has the same layout between platforms.
  fread(&this->blackColor, sizeof(int), 3, calibrationFile);
  fread(&this->whiteColor, sizeof(int), 3, calibrationFile);

  fclose(calibrationFile);

  // We're now calibrated.
  this->calibrated = true;
}

void ChessHelper::PieceIdentifier::saveData() const {
  if (!this->calibrated) {
    throw std::runtime_error(
        "saveData called on a non-calibrated PieceIdentifier!");
  }

  struct stat dir{};
  if (stat(this->calibrationDir.c_str(), &dir) != 0) {
    // Directory doesn't exist, we need to create it.
    int nError = 0;

#if defined(_WIN32)
    nError = _mkdir(this->calibrationDir.c_str());
#else
    nError = mkdir(this->calibrationDir.c_str(), 0777);
#endif

    if (nError != 0) {
      throw std::runtime_error("Failed to create calibration directory!");
    }
  }

  std::string calibrationPath = this->calibrationDir + "/calibration.dat";
  FILE *calibrationFile = fopen(calibrationPath.c_str(), "wb");
  if (calibrationFile == nullptr) {
    throw std::runtime_error("Failed to write into calibration file!");
  }

  // SAFETY: To my knowledge, a float array has the exact same
  //         layout (4 bytes per item, 4 byte alignment, no padding)
  //         across x86_64 and arm64, and Linux and Windows.
  fwrite(&this->histogramsByPiece, sizeof(float),
         MATCH_HISTOGRAM_BINS * NUM_PIECE_TYPES, calibrationFile);

  // SAFETY: int has the same layout between platforms.
  fwrite(&this->blackColor, sizeof(int), 3, calibrationFile);
  fwrite(&this->whiteColor, sizeof(int), 3, calibrationFile);

  fclose(calibrationFile);
}