cmake_minimum_required(VERSION 2.8)

project(spoofingDetection)

find_package( OpenCV REQUIRED )

add_executable( spoofingDetection main.cpp )

target_link_libraries( spoofingDetection ${OpenCV_LIBS} )
