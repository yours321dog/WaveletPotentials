# CMakeList.txt : CMake project for wavepotential_cmake, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)


file(GLOB wavepotential_SRC
     "*.h"
     "*.cpp"
     "*.cuh"
     "*.cu"
)

# Add source to this project's executable.
add_executable (wavepotential_cmake 
    ${wavepotential_SRC})

# TODO: Add tests and install targets if needed.
# link cuda libs
target_link_libraries(wavepotential_cmake PRIVATE CUDA::cufft)