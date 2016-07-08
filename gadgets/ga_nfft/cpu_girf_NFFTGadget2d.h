#ifndef CPU_GIRF_NFFTGADGET2D_H
#define CPU_GIRF_NFFTGADGET2D_H

#include "Gadget.h"
#include "gadgetron_mricore_export.h"

#include "mri_core_data.h"

#include "hoNDArray_fileio.h"
//#include "gridder.h"
#include "ga_trajectory.h"
#include "math.h"
#include "hoNDArray_reductions.h"
#include "girf_trajectory_correction.h"

namespace Gadgetron{

  class EXPORTGADGETSMRICORE cpu_girf_NFFTGadget2d : 
  public Gadget1<IsmrmrdReconData>
    {
    public:
      GADGET_DECLARE(cpu_girf_NFFTGadget2d)
      cpu_girf_NFFTGadget2d();
	
    protected:
      virtual int process(GadgetContainerMessage<IsmrmrdReconData>* m1);
      long long image_counter_;
      unsigned int Ndims;
      //std::vector<size_t> im_dims(2);      
      int im_dims[3];
      int grid_dims[3];
      int im_size;
      int grid_size;
      float grid_oversampling_factor;
      float ro_oversampling_factor;
      int cut_off;
      int Nspokes;
      int NsamplesPerSpoke;

      //hoNDArray<typename reald<float,2>::Type> trajectory;
      //hoNDArray<float> trajectory;
      hoNDArray<float> DCF;
      hoNDArray<complext<float>> kspace;
      hoNDArray<complext<float>> samples_single_coil;
  
    };
}
#endif //CPU_GIRF_NFFTGADGET2D_H
