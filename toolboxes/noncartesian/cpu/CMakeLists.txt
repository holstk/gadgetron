#include_directories(
#  ${CMAKE_SOURCE_DIR}/toolboxes/mri/pmri/gpu
#  ${CMAKE_SOURCE_DIR}/toolboxes/fft/cpu
#  ${CMAKE_SOURCE_DIR}/toolboxes/core
#  ${CMAKE_SOURCE_DIR}/toolboxes/core/cpu
#  ${CMAKE_SOURCE_DIR}/toolboxes/core/cpu/hostutils
#  ${CMAKE_SOURCE_DIR}/toolboxes/core/cpu/math
#  ${CMAKE_SOURCE_DIR}/toolboxes/noncartesian/cpu/nfft
#  ${CMAKE_SOURCE_DIR}/toolboxes/noncartesian/cpu/ga_radial
#)

#add_executable(nffth_test main.cpp)

#target_link_libraries(nffth_test 
#                      gadgetron_toolbox_hostutils 
#		      gadgetron_toolbox_cpureg 
#		      gadgetron_toolbox_cpucore 
#		      gadgetron_toolbox_cpucore_math
#		      gadgetron_toolbox_cpufft
#		      cpu_gridder
#		      ga_radial_trajectory		      
#		      ${ARMADILLO_LIBRARIES})

#install(TARGETS ho_nffth_ga_radial DESTINATION bin COMPONENT main)

add_subdirectory(nfft)
add_subdirectory(ga_radial)
