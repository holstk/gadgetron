#ifndef CPU_CSM_GADGET_3D_H
#define CPU_CSM_GADGET_3D_H

#include "Gadget.h"
#include "gadgetron_mricore_export.h"

#include "mri_core_data.h"

#include "hoNDArray_fileio.h"
//#include "gridder.h"
#include "ga_trajectory.h"
#include "math.h"
#include "hoNDArray_reductions.h"

namespace Gadgetron{

  class EXPORTGADGETSMRICORE cpu_csm_gadget_3d : 
  public Gadget1<IsmrmrdReconData>
    {
    public:
      GADGET_DECLARE(cpu_csm_gadget_3d)
      cpu_csm_gadget_3d();
	
    protected:
      virtual int process(GadgetContainerMessage<IsmrmrdReconData>* m1);
      long long image_counter_;
      int im_size;
      int im_size_os;
      float grid_oversampling_factor;
      float ro_oversampling_factor;
      int cut_off;
      int num_spokes;
      int num_samples_per_spoke;
      typename uint64d<3>::Type matrix_size;
      typename uint64d<3>::Type matrix_size_os;
      float alpha;
      float kernel_width;

      hoNDArray<float> DCF;
      hoNDArray<complext<float>> kspace;
      hoNDArray<complext<float>> samples_single_coil;
  
    };
}
#endif //CPU_CSM_GADGET_3D_H
