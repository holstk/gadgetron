if (CUDA_FOUND)
  add_subdirectory(gpu)
endif ()

add_subdirectory(cpu)
add_subdirectory(ga_radial)

install(FILES NFFT.h DESTINATION ${GADGETRON_INSTALL_INCLUDE_PATH} COMPONENT main)