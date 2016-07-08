/** \file b1_map.h
    \brief Utility to estimate b1 maps (MRI coil sensitivities), GPU based. 
*/

#pragma once

//#include "gpupmri_export.h"
#include "hoNDArray.h"
#include "vector_td.h"
#include "complext.h"

#include <boost/shared_ptr.hpp>

namespace Gadgetron{

    /** 
     * \brief Estimate b1 map (coil sensitivities) of single or double precision according to REAL and of dimensionality D.
     * \param data Reconstructed reference images from the individual coils. Dimensionality is D+1 where the latter dimensions denotes the coil images.
     * \param taget_coils Denotes the number of target coils. Cannot exceed the size of dimension D of the data. A negative value indicates that sensitivity maps are computed for the full coil image dimension.
     */
    //template<class REAL, unsigned int D>  boost::shared_ptr< hoNDArray<complext<REAL> > >
    //estimate_b1_map( hoNDArray<complext<REAL> > *data, int target_coils = -1 );
    template<class REAL, unsigned int D> void
    estimate_b1_map( hoNDArray<complext<REAL> > *data_in, hoNDArray<complext<REAL> > *data_out, int target_coils);

}
