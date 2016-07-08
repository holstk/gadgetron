include_directories(

  ${CMAKE_SOURCE_DIR}/toolboxes/mri/pmri/gpu
  ${CMAKE_SOURCE_DIR}/toolboxes/fft/cpu
)

#add_executable(test main_test.cpp)
add_executable(ho_nffth_ga_radial main_nffth_ga.cpp)

#target_link_libraries(test gadgetron_toolbox_hostutils gadgetron_toolbox_cpureg gadgetron_toolbox_cpucore gadgetron_toolbox_cpucore_math gadgetron_toolbox_cpufft ${ARMADILLO_LIBRARIES})
target_link_libraries(ho_nffth_ga_radial gadgetron_toolbox_hostutils gadgetron_toolbox_cpureg gadgetron_toolbox_cpucore gadgetron_toolbox_cpucore_math gadgetron_toolbox_cpufft ${ARMADILLO_LIBRARIES})
#target_link_libraries(ho_nffth_ga2d_radial gadgetron_toolbox_gpucore gadgetron_toolbox_gpuoperators gadgetron_toolbox_gpunfft gadgetron_toolbox_hostutils ${CUDA_LIBRARIES})

#install (FILES  gridder.h  DESTINATION ${GADGETRON_INSTALL_INCLUDE_PATH} COMPONENT main)

#install(TARGETS test DESTINATION bin COMPONENT main)
install(TARGETS ho_nffth_ga_radial DESTINATION bin COMPONENT main)

