# volumetric_iou._cgal: one stable-ABI (cp312-abi3) nanobind extension.
#
# Header-only dependencies, each either found locally or downloaded from GitHub:
#   CGAL 6.0.1     -DVIOU_CGAL_DIR=<CGAL-6.0.1 source>       else the release tarball
#   Boost >= 1.72  find_package(Boost), or -DVIOU_BOOST_DIR=<boost-1.86.0-cmake>, else its tarball
#   Eigen 3.4      -DVIOU_EIGEN_DIR=<dir with Eigen/>, or find_package(Eigen3), else its tarball (gitlab.com)
#   libigl f68f01c -DVIOU_IGL_DIR=<libigl source>  else the GitHub archive of that commit
#                  (the one the libigl 2.6.3 bindings on PyPI build; used header-only)
# Exact number type of CGAL: boost multiprecision (default, no runtime library);
# -DVIOU_GMP=ON links GMP/MPFR instead.
cmake_minimum_required(VERSION 3.18)
project(volumetric_iou_cgal LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(VIOU_GMP "CGAL exact numbers on GMP/MPFR instead of boost multiprecision" OFF)
set(VIOU_CGAL_DIR "" CACHE PATH "CGAL 6.0.1 source tree (contains include/CGAL)")
set(VIOU_BOOST_DIR "" CACHE PATH "boost-1.86.0-cmake source tree (libs/*/include)")
set(VIOU_EIGEN_DIR "" CACHE PATH "Eigen 3.4 include dir (contains Eigen/)")
set(VIOU_IGL_DIR "" CACHE PATH "libigl source tree (contains include/igl)")

include(FetchContent)

if(NOT VIOU_CGAL_DIR)
  FetchContent_Declare(cgal
    URL https://github.com/CGAL/cgal/releases/download/v6.0.1/CGAL-6.0.1-library.tar.xz
    URL_MD5 ea827f6778063e00554ae41f4c845492)
  FetchContent_GetProperties(cgal)
  if(NOT cgal_POPULATED)
    FetchContent_Populate(cgal)
  endif()
  set(VIOU_CGAL_DIR ${cgal_SOURCE_DIR})
endif()

set(BOOST_INCLUDES "")
if(NOT VIOU_BOOST_DIR)
  find_package(Boost 1.72 QUIET)
  if(Boost_FOUND)
    set(BOOST_INCLUDES ${Boost_INCLUDE_DIRS})
  else()
    FetchContent_Declare(boost
      URL https://github.com/boostorg/boost/releases/download/boost-1.86.0/boost-1.86.0-cmake.tar.gz)
    FetchContent_GetProperties(boost)
    if(NOT boost_POPULATED)
      FetchContent_Populate(boost)
    endif()
    set(VIOU_BOOST_DIR ${boost_SOURCE_DIR})
  endif()
endif()
if(VIOU_BOOST_DIR)
  # the cmake tarball keeps each library's headers apart: libs/<lib>/include
  file(GLOB BOOST_INCLUDES LIST_DIRECTORIES true
    ${VIOU_BOOST_DIR}/libs/*/include ${VIOU_BOOST_DIR}/libs/numeric/*/include)
endif()

if(NOT VIOU_EIGEN_DIR)
  find_package(Eigen3 3.4 QUIET NO_MODULE)
  if(Eigen3_FOUND)
    get_target_property(VIOU_EIGEN_DIR Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
  else()
    FetchContent_Declare(eigen
      URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz)
    FetchContent_GetProperties(eigen)
    if(NOT eigen_POPULATED)
      FetchContent_Populate(eigen)
    endif()
    set(VIOU_EIGEN_DIR ${eigen_SOURCE_DIR})
  endif()
endif()

if(NOT VIOU_IGL_DIR)
  FetchContent_Declare(libigl
    URL https://github.com/libigl/libigl/archive/f68f01c23ac82351e2543ce8c529b6d4c89fdf16.tar.gz)
  FetchContent_GetProperties(libigl)
  if(NOT libigl_POPULATED)
    FetchContent_Populate(libigl)
  endif()
  set(VIOU_IGL_DIR ${libigl_SOURCE_DIR})
endif()

# libigl's closest_facet (extract_cells: which cell of another component a
# component lies in), closest point inside a face: libigl orders the faces
# around the face's first edge as seen from the query point; at an edge where
# two bodies cross at a shallow angle that returns the other body's face, and
# a component outside is taken as nested -- the union drops it.
# cpp/libigl/closest_facet.cpp is libigl's file with that case redone; the
# headers are copied to the build tree and the file put in its place, for
# exactly the file it was made from.
set(VIOU_IGL_INCLUDE ${CMAKE_BINARY_DIR}/igl_patched)
file(REMOVE_RECURSE ${VIOU_IGL_INCLUDE})
file(COPY ${VIOU_IGL_DIR}/include/igl DESTINATION ${VIOU_IGL_INCLUDE})
set(_cf ${VIOU_IGL_INCLUDE}/igl/copyleft/cgal/closest_facet.cpp)
file(MD5 ${_cf} _cf_md5)
if(NOT _cf_md5 STREQUAL "f44cdb57c6874ef6ab00adf9794335e6")
  message(FATAL_ERROR "libigl closest_facet.cpp is not the one cpp/libigl/closest_facet.cpp "
                      "was made from (MD5 ${_cf_md5}): libigl f68f01c is expected")
endif()
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cpp/libigl/closest_facet.cpp ${_cf} COPYONLY)

# Keep libigl's lazy environment lookup/cache, but do not infer the machine's
# CPU count when the caller has not selected IGL_NUM_THREADS. Patch only the
# private build copy, leaving the supplied dependency source untouched.
set(_nt ${VIOU_IGL_INCLUDE}/igl/default_num_threads.cpp)
file(READ ${_nt} _nt_source)
set(_nt_old "const unsigned int hw_num_threads = std::thread::hardware_concurrency();")
string(FIND "${_nt_source}" "${_nt_old}" _nt_pos)
if(_nt_pos EQUAL -1)
  message(FATAL_ERROR "Unexpected libigl default_num_threads.cpp: thread-default patch cannot be applied")
endif()
string(REPLACE "${_nt_old}" "const unsigned int hw_num_threads = 1u; // volumetric-iou default"
  _nt_source "${_nt_source}")
file(WRITE ${_nt} "${_nt_source}")

find_package(Python 3.12 COMPONENTS Interpreter Development.Module Development.SABIModule REQUIRED)
execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

# NOMINSIZE: nanobind adds -Os by default; CGAL predicates run ~2x slower with it
nanobind_add_module(_cgal STABLE_ABI NB_STATIC NOMINSIZE cpp/module.cpp)
# numpy's operation order in cpp/clean.h: no product fused into an addition
target_compile_options(_cgal PRIVATE -ffp-contract=off)
target_include_directories(_cgal SYSTEM PRIVATE ${VIOU_CGAL_DIR}/include ${BOOST_INCLUDES}
  ${VIOU_EIGEN_DIR} ${VIOU_IGL_INCLUDE})
if(VIOU_GMP)
  find_path(GMP_INCLUDE gmp.h REQUIRED)
  find_library(GMP_LIB gmp REQUIRED)
  find_library(MPFR_LIB mpfr REQUIRED)
  target_include_directories(_cgal SYSTEM PRIVATE ${GMP_INCLUDE})
  target_link_libraries(_cgal PRIVATE ${MPFR_LIB} ${GMP_LIB})
  # the module finds the GMP/MPFR it was built against (a conda prefix, not the system)
  get_filename_component(GMP_LIBDIR ${GMP_LIB} DIRECTORY)
  set_target_properties(_cgal PROPERTIES INSTALL_RPATH ${GMP_LIBDIR})
else()
  target_compile_definitions(_cgal PRIVATE CGAL_DISABLE_GMP=1)
endif()

install(TARGETS _cgal LIBRARY DESTINATION volumetric_iou)
