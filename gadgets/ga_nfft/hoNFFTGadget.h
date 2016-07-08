#ifndef HONFFTGADGET_H
#define HONFFTGADGET_H

#include "Gadget.h"
#include "gadgetron_mricore_export.h"

#include "mri_core_data.h"

#include "hoNDArray_fileio.h"
//#include "gridder.h"
#include "ga_trajectory.h"
#include "math.h"

namespace Gadgetron{

  class EXPORTGADGETSMRICORE hoNFFTGadget : 
  public Gadget1<IsmrmrdReconData>
    {
    public:
      GADGET_DECLARE(hoNFFTGadget)
      hoNFFTGadget();
	
    protected:
      virtual int process(GadgetContainerMessage<IsmrmrdReconData>* m1);
      long long image_counter_;
      unsigned int Ndims;
      //std::vector<size_t> im_dims(3);      
      int im_dims[3];
      int grid_dims[3];
      int im_size;
      int grid_size;
      float grid_oversampling_factor;
      float ro_oversampling_factor;
      int cut_off;
      int Nspokes;
      int NsamplesPerSpoke;
      
      //hoNDArray<typename reald<float,3>::Type> trajectory;
      //hoNDArray<float> trajectory;
      hoNDArray<float> DCF;
      hoNDArray<complext<float>> kspace;
      hoNDArray<complext<float>> samples_single_coil;
  
    };
}
#endif //HONFFTGADGET_H
