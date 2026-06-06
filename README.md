# Chess Helper

This is a computer vision project to recognize pieces on a chess board. See [images/](./images/) for some example images to use.

## Usage

This program has two modes: command-line or video. Command-line lets you calibrate the piece detector and analyze still images,
video uses images taken from your webcam.

To analyze a board, enter command mode by typing `c`, then use `images/init-phone-cropped.jpg` (which has been created as
a best-case image for the program). Next, type `a` to analyze the board. If it gives a calibration error (which can
happen if your working directory isn't the project root, just go back and type `c` instead to calibrate, then go through
this process). This will give you a board output you can visualize to verify how well it works.

In video mode (type `v` at the start to enter), first press `s` to find the image corners, then press ` ` (enter) to analyze
the board, which will show the detected pieces on screen. This mode requires calibration first (we've included a calibration
file here, but if your working directory isn't correct or something else is wrong, you can use the instructions above
to recalibrate).

Note: command-line and video mode use two different corner detection modes. This is because we found that high-quality
images work better with the initial method, whereas lower quality images (i.e., from a webcam) work better with the new
method. However, the piece identifier works much better with higher quality images (otherwise it can't reliably find the
edges, as they're too blurry).

## Code

Source files (cpp) are stored in [src/](./src/), and header files (h) are stored in [include/ChessHelper](./include/ChessHelper).

* `src/driver.cpp` - contains the user interface.
* `src/corners.cpp` - contains the code for finding board corners (there are two methods).
* `src/matching.cpp` - contains the code for piece matching (presence, color, and type).
* `src/engine.cpp` - contains the code for communicating with a chess engine (to find the best move).
* `src/utils.cpp` - contains some misc helper functions (currently just one to do a perspective transform).