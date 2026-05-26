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

# MSVC's <cmath> only exposes M_PI when _USE_MATH_DEFINES is defined first
target_compile_definitions(ChessHelper PRIVATE _USE_MATH_DEFINES)

target_link_libraries(ChessHelper ${OpenCV_LIBS})
target_include_directories(ChessHelper PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

# Set the working directory for debugging in Visual Studio to the project root
set_target_properties(ChessHelper PROPERTIES
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)