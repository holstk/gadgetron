include_directories( 
  ${CMAKE_SOURCE_DIR}/toolboxes/mri/pmri/cpu
  ${CMAKE_SOURCE_DIR}/toolboxes/nfft/ga_radial
  ${CMAKE_SOURCE_DIR}/toolboxes/nfft/cpu

  ${CMAKE_SOURCE_DIR}/toolboxes/core
  ${CMAKE_SOURCE_DIR}/toolboxes/core/cpu
  ${CMAKE_SOURCE_DIR}/toolboxes/core/cpu/math
  ${CMAKE_SOURCE_DIR}/toolboxes/fft/cpu
  ${CMAKE_SOURCE_DIR}/toolboxes/operators
  ${Boost_INCLUDE_DIR}
  ${CMAKE_SOURCE_DIR}/toolboxes/nfft
  ${CMAKE_SOURCE_DIR}/toolboxes/mri/pmri
  ${CMAKE_SOURCE_DIR}/toolboxes/nfft/ga_radial
  )

add_executable(cpu_sense_cg_2d cpu_cg.cpp)
add_executable(cpu_sense_cg_3d cpu_cg_3d.cpp)

target_link_libraries(
  cpu_sense_cg_2d 
  #gadgetron_toolbox_cpuoperators 
  gadgetron_toolbox_cpucore 
  gadgetron_toolbox_gpucore 
  gadgetron_toolbox_cpuparallelmri 
  gadgetron_toolbox_cpunfft 
  gadgetron_toolbox_hostutils 
  gadgetron_toolbox_cpu_solver 
  #${CUDA_LIBRARIES}
  ga_radial_trajectory
  )

target_link_libraries(
  cpu_sense_cg_3d 
  #gadgetron_toolbox_cpuoperators 
  gadgetron_toolbox_cpucore 
  gadgetron_toolbox_gpucore 
  gadgetron_toolbox_cpuparallelmri 
  gadgetron_toolbox_cpunfft 
  gadgetron_toolbox_hostutils 
  gadgetron_toolbox_cpu_solver 
  #${CUDA_LIBRARIES}
  ga_radial_trajectory
  )

install(TARGETS cpu_sense_cg_2d cpu_sense_cg_3d DESTINATION bin COMPONENT main)

#add_subdirectory(radial)
