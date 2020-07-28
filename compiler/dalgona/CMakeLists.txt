find_package(PythonLibs QUIET)
if(NOT ${PYTHONLIBS_FOUND})
  message(STATUS "Build dalgona: FAILED (Python lib is missing)")
  return()
endif()

nnas_find_package(Pybind11)
if(NOT Pybind11_FOUND)
  message(STATUS "Build dalgona: FAILED (missing Pybind11)")
  return()
endif(NOT Pybind11_FOUND)

nnas_find_package(HDF5 COMPONENTS STATIC QUIET)
if(NOT HDF5_FOUND)
  message(STATUS "Build dalgona: FAILED (missing HDF5)")
  return()
endif(NOT HDF5_FOUND)

add_subdirectory(tests)

set(DRIVER "driver/Driver.cpp")

file(GLOB_RECURSE SOURCES "src/*.cpp")

add_executable(dalgona ${DRIVER} ${SOURCES})
target_include_directories(dalgona PRIVATE include)
target_include_directories(dalgona PRIVATE ${PYTHON_INCLUDE_DIRS})
target_include_directories(dalgona PRIVATE ${Pybind11_INCLUDE_DIRS})
target_include_directories(dalgona PRIVATE ${HDF5_INCLUDE_DIRS})

target_link_libraries(dalgona INTERFACE pybind11::embed)
target_link_libraries(dalgona PRIVATE ${HDF5_CXX_LIBRARIES})
target_link_libraries(dalgona PRIVATE ${PYTHON_LIBRARIES})
target_link_libraries(dalgona PRIVATE arser)
target_link_libraries(dalgona PRIVATE safemain)
target_link_libraries(dalgona PRIVATE luci_import)
target_link_libraries(dalgona PRIVATE luci_interpreter)
