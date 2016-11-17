include_directories(

  ${CMAKE_SOURCE_DIR}/toolboxes/mri/pmri/gpu
  ${CMAKE_SOURCE_DIR}/toolboxes/fft/cpu
  ${CMAKE_SOURCE_DIR}/toolboxes/core
  ${CMAKE_SOURCE_DIR}/toolboxes/core/cpu
  ${CMAKE_SOURCE_DIR}/toolboxes/core/cpu/math
  ${CMAKE_SOURCE_DIR}/toolboxes/nfft/cpu
)

#add_executable(cpu_gridder main_nffth_ga.cpp)

add_library(ga_radial_trajectory SHARED
            ga_trajectory.h ga_trajectory.cpp
	    linear_radial_trajectory.h linear_radial_trajectory.cpp
            rotate_kspace.h rotate_kspace.cpp
)

target_link_libraries(ga_radial_trajectory
                      gadgetron_toolbox_hostutils 
		      gadgetron_toolbox_cpureg 
		      gadgetron_toolbox_cpucore 
		      gadgetron_toolbox_cpucore_math 
		      gadgetron_toolbox_cpufft 
		      ${ARMADILLO_LIBRARIES}
		      )

install (FILES  ga_trajectory.h  linear_radial_trajectory.h  DESTINATION ${GADGETRON_INSTALL_INCLUDE_PATH} COMPONENT main)

install(TARGETS ga_radial_trajectory DESTINATION lib COMPONENT main)

