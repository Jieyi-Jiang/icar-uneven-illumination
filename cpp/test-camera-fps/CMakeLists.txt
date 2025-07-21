cmake_minimum_required(VERSION 3.5)
project( test_fps )
find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )

include_directories(./)

add_executable( test_fps main.cpp )
target_link_libraries( test_fps ${OpenCV_LIBS} )

