cmake_minimum_required(VERSION 3.0)

# project name
project(cppImp)

# define path to the libtorch extracted folder
set(CMAKE_PREFIX_PATH /mnt/d/mywork/libtorch_work/torchinstall/libtorch) # ADD YOUR PATH HERE

# find torch library and all necessary files
find_package(Torch REQUIRED)

#include_directories(${onnxruntime_INCLUDE_DIRS}/onnxruntime/include)

#include_directories(${onnxruntime_INCLUDE_DIRS}/include)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# executable to add that we want to compile and runcppImp

add_executable(cppImp main.cpp)

# link torch libraries to our executable
target_link_libraries(cppImp "${TORCH_LIBRARIES}")

set_property(TARGET cppImp PROPERTY CXX_STANDARD 17)
