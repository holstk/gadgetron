/** \file hoNonCartesianSenseOperator.h
    \brief Non-Cartesian Sense operator, CPU based.
*/

#pragma once

#include "hoSenseOperator.h"
#include "hoNFFT.h"

namespace Gadgetron{

    template<class REAL, unsigned int D> class hoNonCartesianSenseOperator : public hoSenseOperator<REAL,D>
    {
  
    public:
  
	typedef typename uint64d<D>::Type _uint64d;
	typedef typename reald<REAL,D>::Type _reald;

	hoNonCartesianSenseOperator() : hoSenseOperator<REAL,D>() { 
	    plan_ = boost::shared_ptr< hoNFFT_plan<REAL, D> >( new hoNFFT_plan<REAL, D>() );
	    is_preprocessed_ = false;
	}
    
	virtual ~hoNonCartesianSenseOperator() {}
    
    
	inline boost::shared_ptr< hoNFFT_plan<REAL, D> > get_plan() { return plan_; }
	inline boost::shared_ptr< hoNDArray<REAL> > get_dcw() { return dcw_; }
	inline bool is_preprocessed() { return is_preprocessed_; } 
     
	virtual void mult_M( hoNDArray< complext<REAL> >* in, hoNDArray< complext<REAL> >* out, bool accumulate = false );
	virtual void mult_MH( hoNDArray< complext<REAL> >* in, hoNDArray< complext<REAL> >* out, bool accumulate = false );

	virtual void setup( _uint64d matrix_size, _uint64d matrix_size_os, REAL W );
	virtual void preprocess( hoNDArray<_reald> *trajectory );
	virtual void set_dcw( boost::shared_ptr< hoNDArray<REAL> > dcw );


      
    protected:
	boost::shared_ptr< hoNFFT_plan<REAL, D> > plan_;
        boost::shared_ptr< hoNDArray<REAL> > dcw_;
	bool is_preprocessed_;
      
    };
  
    //Atomics can't be used with doubles
    //template<unsigned int D> class hoNonCartesianSenseOperator<double,D>{};
      
}
