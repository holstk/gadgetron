include_directories(

  ${CMAKE_SOURCE_DIR}/toolboxes/mri/pmri/gpu
  ${CMAKE_SOURCE_DIR}/toolboxes/fft/cpu
)

#add_executable(cpu_gridder main_nffth_ga.cpp)

add_library(cpu_gridder SHARED
            gridder.h gridder.cpp)

target_link_libraries(cpu_gridder
                      gadgetron_toolbox_hostutils 
		      gadgetron_toolbox_cpureg 
		      gadgetron_toolbox_cpucore 
		      gadgetron_toolbox_cpucore_math 
		      gadgetron_toolbox_cpufft 
		      ${ARMADILLO_LIBRARIES})

install (FILES  gridder.h  DESTINATION ${GADGETRON_INSTALL_INCLUDE_PATH} COMPONENT main)

install(TARGETS cpu_gridder DESTINATION bin COMPONENT main)

