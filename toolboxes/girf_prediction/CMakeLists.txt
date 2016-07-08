include_directories(

  ${CMAKE_SOURCE_DIR}/toolboxes/mri/pmri/gpu
  ${CMAKE_SOURCE_DIR}/toolboxes/fft/cpu
  ${CMAKE_SOURCE_DIR}/toolboxes/core
  ${CMAKE_SOURCE_DIR}/toolboxes/core/cpu
  ${CMAKE_SOURCE_DIR}/toolboxes/core/cpu/math
  ${CMAKE_SOURCE_DIR}/toolboxes/nfft/cpu
)

#add_executable(cpu_gridder main_nffth_ga.cpp)

add_library(girf_prediction SHARED
            girf_trajectory_correction.h girf_trajectory_correction.cpp)

target_link_libraries(girf_prediction
                      gadgetron_toolbox_hostutils 
		      gadgetron_toolbox_cpureg 
		      gadgetron_toolbox_cpucore 
		      gadgetron_toolbox_cpucore_math 
		      gadgetron_toolbox_cpufft 
		      ${ARMADILLO_LIBRARIES}
		      )

install (FILES  girf_trajectory_correction.h  DESTINATION ${GADGETRON_INSTALL_INCLUDE_PATH} COMPONENT main)

install(TARGETS girf_prediction DESTINATION lib COMPONENT main)

