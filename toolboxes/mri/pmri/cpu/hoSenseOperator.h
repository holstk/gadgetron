/** \file hoSenseOperator.h
    \brief Base class for the GPU based Sense operators
*/

#pragma once

#include "hoNDArray.h"
#include "senseOperator.h"
#include "linearOperator.h"
#include "hoNDArray_elemwise.h"
#include "vector_td.h"
#include "complext.h"
#include "hoNDArray_reductions.h"

namespace Gadgetron{

template<class REAL, unsigned int D> class hoSenseOperator : public senseOperator< hoNDArray< complext<REAL> >, D >
    {
	
    public:
	
	hoSenseOperator() : senseOperator<hoNDArray< complext<REAL> >,D >() {}
	virtual ~hoSenseOperator() {}

//virtual void mult_M( hoNDArray< complext<REAL> > *in, hoNDArray< complext<REAL> > *out, bool accumulate = false ) = 0;
//virtual void mult_MH( hoNDArray< complext<REAL> > *in, hoNDArray< complext<REAL> > *out, bool accumulate = false ) = 0;
	
	virtual void mult_csm( hoNDArray< complext<REAL> > *in, hoNDArray< complext<REAL> > *out );
	virtual void mult_csm_conj_sum( hoNDArray< complext<REAL> > *in, hoNDArray< complext<REAL> > *out );    


    };
}
