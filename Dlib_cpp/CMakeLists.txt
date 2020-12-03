cmake_minimum_required(VERSION 2.8.12)
project(examples)

find_package( OpenCV REQUIRED )
add_subdirectory(./dlib dlib_build)
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( main main.cpp )
target_link_libraries( main ${OpenCV_LIBS} )
target_link_libraries(main dlib::dlib)

