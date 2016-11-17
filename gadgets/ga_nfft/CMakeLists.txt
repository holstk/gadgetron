#IF (WIN32)
#  ADD_DEFINITIONS(-D__BUILD_GADGETRON_RADIAL__)
#ENDIF (WIN32)

#find_package(Ismrmrd REQUIRED)

include_directories(
  ${CMAKE_SOURCE_DIR}/gadgets/mri_core
  ${CMAKE_SOURCE_DIR}/gadgets/pmri
  #${CMAKE_SOURCE_DIR}/toolboxes/nfft/gpu
  ${CMAKE_SOURCE_DIR}/toolboxes/core/cpu
  ${CMAKE_SOURCE_DIR}/toolboxes/core/cpu/math
  #${CMAKE_SOURCE_DIR}/toolboxes/fft/gpu
  ${CMAKE_SOURCE_DIR}/toolboxes/solvers
  #${CMAKE_SOURCE_DIR}/toolboxes/solvers/gpu
  ${CMAKE_SOURCE_DIR}/toolboxes/operators
  ${CMAKE_SOURCE_DIR}/toolboxes/nfft/cpu
  ${CMAKE_SOURCE_DIR}/toolboxes/nfft
  #${CMAKE_SOURCE_DIR}/toolboxes/operators/gpu

  ${CMAKE_SOURCE_DIR}/toolboxes/mri_core
  ${CMAKE_SOURCE_DIR}/toolboxes/fft/cpu

  #${CMAKE_SOURCE_DIR}/toolboxes/noncartesian/cpu/nfft
  ${CMAKE_SOURCE_DIR}/toolboxes/nfft/ga_radial

  ${ARMADILLO_INCLUDE_DIRS}

  ${CMAKE_SOURCE_DIR}/toolboxes/core/gpu
  ${CMAKE_SOURCE_DIR}/toolboxes/mri/pmri/cpu
  ${CMAKE_SOURCE_DIR}/toolboxes/mri/pmri/gpu
  ${CMAKE_SOURCE_DIR}/toolboxes/girf_prediction
  )

#include_directories(${CUDA_INCLUDE_DIRS})

#if (ARMADILLO_FOUND)
#  list(APPEND PHASE_GADGET RadialPhaseCorrectionGadget.h RadialPhaseCorrectionGadget.cpp)
#  install (FILES  RadialPhaseCorrectionGadget.h DESTINATION ${GADGETRON_INSTALL_INCLUDE_PATH} COMPONENT main)
#elseif (ARMADILLO_FOUND)
#  MESSAGE("Armadillo not found, not compiling radial phase correction gadget")
#endif (ARMADILLO_FOUND)

add_library(gadgetron_cpu_nfft SHARED 
  #gadgetron_radial_export.h
  #gpuRadialPrepGadget.h gpuRadialPrepGadget.cpp 
  #gpuRadialSensePrepGadget.h gpuRadialSensePrepGadget.cpp 
  #gpuRadialSpiritPrepGadget.h gpuRadialSpiritPrepGadget.cpp 
  #gpuRetroGatedSensePrepGadget.h gpuRetroGatedSensePrepGadget.cpp

  #hoNFFTGadget.h hoNFFTGadget.cpp
  cpuNFFTGadget2d.h cpuNFFTGadget2d.cpp
  cpuNFFTGadget3d.h cpuNFFTGadget3d.cpp
  cpuNFFTGadget3d_lin.h cpuNFFTGadget3d_lin.cpp
  cpu_girf_NFFTGadget2d.h cpu_girf_NFFTGadget2d.cpp
  cpu_csm_gadget_3d.h cpu_csm_gadget_3d.cpp
  #cpuCgSenseGadget2d.h cpuCgSenseGadget2d.cpp
  cpuNFFTGadget3d_bin.h cpuNFFTGadget3d_bin.cpp

  #${PHASE_GADGET}
  )

#set_target_properties(gadgetron_gpuradial PROPERTIES VERSION ${GADGETRON_VERSION_STRING} SOVERSION ${GADGETRON_SOVERSION})
set_target_properties(gadgetron_cpu_nfft PROPERTIES VERSION ${GADGETRON_VERSION_STRING} SOVERSION ${GADGETRON_SOVERSION})

target_link_libraries(gadgetron_cpu_nfft
  gadgetron_gadgetbase
  gadgetron_toolbox_log
  #gadgetron_toolbox_gpunfft gadgetron_toolbox_gpusolvers gadgetron_toolbox_gpuoperators 
  gadgetron_toolbox_cpucore 
  gadgetron_toolbox_cpucore_math 
  gadgetron_toolbox_gpucore

  #cpu_gridder
  ga_radial_trajectory	

  gadgetron_toolbox_cpunfft
  gadgetron_toolbox_cpuparallelmri
  girf_prediction

  ${ISMRMRD_LIBRARIES} ${FFTW3_LIBRARIES} ${CUDA_LIBRARIES}
  optimized ${ACE_LIBRARIES} debug ${ACE_DEBUG_LIBRARY}
  )

#target_link_libraries(gadgetron_gpuradial gadgetron_toolbox_gpuparallelmri )
#target_link_libraries(gadgetron_cpu_nfft)

install (FILES  #gadgetron_radial_export.h
                #gpuRadialPrepGadget.h 
                #gpuRadialSensePrepGadget.h 
                #gpuRadialSpiritPrepGadget.h 
                #gpuRetroGatedSensePrepGadget.h

		#hoNFFTGadget.h
		cpuNFFTGadget2d.h
		cpuNFFTGadget3d.h
		cpuNFFTGadget3d_lin.h
		cpu_girf_NFFTGadget2d.h
		cpu_csm_gadget_3d.h
		cpuNFFTGadget3d_bin.h
		
                DESTINATION ${GADGETRON_INSTALL_INCLUDE_PATH} COMPONENT main)

#install (TARGETS gadgetron_gpuradial DESTINATION lib COMPONENT main)
install (TARGETS gadgetron_cpu_nfft DESTINATION lib COMPONENT main)

add_subdirectory(config)
#add_subdirectory(2d)
#add_subdirectory(3d)
