add_executable (CornerDeterctor "CornerDeterctor.cpp" "CornerDeterctor.h")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET CornerDeterctor PROPERTY CXX_STANDARD 20)
endif()

find_package(OpenCV REQUIRED)
target_link_libraries(CornerDeterctor ${OpenCV_LIBS})