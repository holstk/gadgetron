#pragma once

#include "hoNDArray.h"
#include "complext.h"
//#include "gpupmri_export.h"

namespace Gadgetron{

// Multiply with coil sensitivities
//

template< class REAL, unsigned int D> void
csm_mult_M( hoNDArray< complext<REAL> > *in, 
	    hoNDArray< complext<REAL> > *out, 
	    hoNDArray< complext<REAL> > *csm );


// Multiply with adjoint of coil sensitivities
//

template< class REAL, unsigned int D> void
csm_mult_MH( hoNDArray< complext<REAL> > *in, 
	     hoNDArray< complext<REAL> > *out, 
	     hoNDArray< complext<REAL> > *csm );
}
