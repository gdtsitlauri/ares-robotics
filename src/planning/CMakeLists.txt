cmake_minimum_required(VERSION 3.18)
project(ares_planning_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

add_library(planning_core MODULE planning_core.cpp)
target_compile_options(planning_core PRIVATE -O3)
target_include_directories(planning_core PRIVATE ${Python3_INCLUDE_DIRS})

find_package(pybind11 QUIET)
if(pybind11_FOUND)
  target_compile_definitions(planning_core PRIVATE ARES_WITH_PYBIND=1)
  target_link_libraries(planning_core PRIVATE pybind11::module)
else()
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_include())"
    OUTPUT_VARIABLE PYBIND11_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE PYBIND11_STATUS
  )
  if(PYBIND11_STATUS EQUAL 0)
    target_compile_definitions(planning_core PRIVATE ARES_WITH_PYBIND=1)
    target_include_directories(planning_core PRIVATE ${PYBIND11_INCLUDE_DIR})
  endif()
endif()
