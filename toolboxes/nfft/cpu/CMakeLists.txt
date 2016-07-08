#if (WIN32)
#  ADD_DEFINITIONS(-D__BUILD_GADGETRON_GPUNFFT__)
#  ADD_DEFINITIONS(-D_USE_MATH_DEFINES)
#endif (WIN32)

#if(WIN32)
#  link_directories(${Boost_LIBRARY_DIRS})
#endif(WIN32)

include_directories( 
  ${CMAKE_SOURCE_DIR}/toolboxes/core
  ${CMAKE_SOURCE_DIR}/toolboxes/core/cpu
  ${CMAKE_SOURCE_DIR}/toolboxes/core/cpu/math
  ${CMAKE_SOURCE_DIR}/toolboxes/fft/cpu
  ${CMAKE_SOURCE_DIR}/toolboxes/operators
  ${Boost_INCLUDE_DIR}
  ${CMAKE_SOURCE_DIR}/toolboxes/nfft/cpu
  ${CMAKE_SOURCE_DIR}/toolboxes/nfft
  )

add_library(gadgetron_toolbox_cpunfft SHARED 
    ../NFFT.h
    hoNFFT.h
    hoNFFTOperator.h
    hoNFFT.cpp 
    hoNFFTOperator.cpp
    KaiserBessel_operators.h
    KaiserBessel_operators.cpp
    NFFT_preprocess.cpp
    NFFT_C2NC_conv_kernel.cpp
  )

set_target_properties(gadgetron_toolbox_cpunfft PROPERTIES VERSION ${GADGETRON_VERSION_STRING} SOVERSION ${GADGETRON_SOVERSION})

target_link_libraries(gadgetron_toolbox_cpunfft 
  gadgetron_toolbox_cpufft
  gadgetron_toolbox_cpucore
  gadgetron_toolbox_log
  )

install(TARGETS gadgetron_toolbox_cpunfft DESTINATION lib COMPONENT main)

install(FILES 
  hoNFFT.h 
  hoNFFTOperator.h 
  KaiserBessel_operators.h
  DESTINATION ${GADGETRON_INSTALL_INCLUDE_PATH} COMPONENT main)
