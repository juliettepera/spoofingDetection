cmake_minimum_required(VERSION 2.8)

project(spoofingDetection)

find_package( OpenCV REQUIRED )

add_executable( spoofingDetection spoofing.cpp spoofingUtils.h spoofingIO.h)

target_link_libraries( spoofingDetection ${OpenCV_LIBS} )
