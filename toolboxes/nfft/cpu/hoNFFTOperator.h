#pragma once

#include "hoNDArray_math.h"
#include "linearOperator.h"
#include "hoNFFT.h"
//#include "gpunfft_export.h"

namespace Gadgetron{
    
    template <class T, unsigned int D> class hoNFFTOperator : public virtual linearOperator < hoNDArray <complext<T> > >
    {  
    public:
	
	hoNFFTOperator() : linearOperator<hoNDArray< complext<T> > >() {
	    plan_ = boost::shared_ptr< hoNFFT_plan<T, D> >( new hoNFFT_plan<T, D>() );
	}
	
	virtual ~hoNFFTOperator() {}
	
	virtual void set_dcw( boost::shared_ptr< hoNDArray<T> > dcw ) { dcw_ = dcw; }
	inline boost::shared_ptr< hoNDArray<T> > get_dcw() { return dcw_; }
	
	inline boost::shared_ptr< hoNFFT_plan<T, D> > get_plan() { return plan_; }
	
	virtual void setup( typename uint64d<D>::Type matrix_size, typename uint64d<D>::Type matrix_size_os, T W );
	virtual void preprocess( hoNDArray<typename reald<T,D>::Type> *trajectory );
	
	virtual void mult_M( hoNDArray< complext<T> > *in, hoNDArray< complext<T> > *out, bool accumulate = false );
	virtual void mult_MH( hoNDArray< complext<T> > *in, hoNDArray< complext<T> > *out, bool accumulate = false );
	virtual void mult_MH_M( hoNDArray< complext<T> > *in, hoNDArray< complext<T> > *out, bool accumulate = false );


  protected:
    boost::shared_ptr< hoNFFT_plan<T, D> > plan_;
    boost::shared_ptr< hoNDArray<T> > dcw_;
  };
}
