cmake_minimum_required(VERSION 3.23)
project(ChessHelper CXX)

# Note that the professor uses C++14 in Visual Studio 2022 and 2026
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/${CMAKE_BUILD_TYPE})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/${CMAKE_BUILD_TYPE})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/${CMAKE_BUILD_TYPE})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# OpenCV config
# Make sure you have the OpenCV binaries and headers installed
find_package(OpenCV REQUIRED)

add_executable(ChessHelper "src/driver.cpp" "src/utils.cpp" "src/matching.cpp")
target_link_libraries(ChessHelper ${OpenCV_LIBS})
# TODO link include directories when added

target_include_directories(ChessHelper PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
