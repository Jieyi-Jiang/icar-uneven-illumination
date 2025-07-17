cmake_minimum_required(VERSION 3.5)
project( test_icar )
find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )

include_directories(./)

add_executable( test_icar main.cpp )
target_link_libraries( test_icar ${OpenCV_LIBS} )

