/** \file cgCallback.h
    \brief Class to specify the termination criteria for the conjugate gradient solver through a callback mechanism.
*/

#pragma once

#include "real_utilities.h"
#include "hoCgSolver.h"

namespace Gadgetron{

    template <class T, class REAL> class hoCgSolver;

    template <class T, class REAL> class hoCgTerminationCallback
    {

    public:

	//typedef typename ARRAY_TYPE::element_type ELEMENT_TYPE;
	//typedef typename realType<ELEMENT_TYPE>::Type REAL;

	hoCgTerminationCallback() {}
	virtual ~hoCgTerminationCallback() {}
  
	virtual bool initialize( hoCgSolver<T, REAL> *hoCg ){hoCg_ = hoCg; return true;}
	virtual bool iterate( unsigned int iteration, REAL *tc_metric, bool *tc_terminate ) = 0;

    protected:

	hoCgSolver<T, REAL> *hoCg_;

	REAL get_rq(){
	    return hoCg_->rq_;
	}

	REAL get_rq0(){
	    return hoCg_->rq0_;
	}

	REAL get_alpha(){
	    return hoCg_->alpha_;
	}

	boost::shared_ptr<hoNDArray<T> > get_x(){
	    return hoCg_->x_;
	}

	boost::shared_ptr<hoNDArray<T> > get_p(){
	    return hoCg_->p_;
	}

	boost::shared_ptr<hoNDArray<T> > get_r(){
	    return hoCg_->r_;
	}
    };

    template <class T, class REAL> class hoRelativeResidualTCB
	: public hoCgTerminationCallback<T, REAL>
    {

    protected:
	typedef hoCgTerminationCallback<T, REAL > hoCgTC;

    public:

	//typedef typename ARRAY_TYPE::element_type ELEMENT_TYPE;
	//typedef typename realType<ELEMENT_TYPE>::Type REAL;

	hoRelativeResidualTCB() : hoCgTerminationCallback<T, REAL >() {
	    rq_0_ = REAL(0); 
	    tc_last_ = get_max<REAL>();
	}
  
	virtual ~hoRelativeResidualTCB() {}
  
	virtual bool initialize( hoCgSolver<T, REAL > *hoCg )
	{
	    hoCgTC::initialize(hoCg);
	    tc_last_ = get_max<REAL>();
	    rq_0_ = hoCgTC::get_rq0();
	    return true;
	}
  
	virtual bool iterate( unsigned int iteration, REAL *tc_metric, bool *tc_terminate )
	{
	    *tc_metric = hoCgTC::get_rq()/rq_0_;

	    std::cout << "hoCgCallback: tc_metric = " << *tc_metric << std::endl;
    
	    if( hoCgTC::hoCg_->get_output_mode() >= solver<hoNDArray<T>,hoNDArray<T> >::OUTPUT_WARNINGS ) {
		if( hoCgTC::hoCg_->get_output_mode() >= solver<hoNDArray<T>,hoNDArray<T> >::OUTPUT_VERBOSE ) {
		    GDEBUG_STREAM("Iteration " << iteration << ". rq/rq_0 = " << *tc_metric << std::endl);
		}
		if( (tc_last_-(*tc_metric)) < REAL(0) ){
		    GDEBUG_STREAM("Warning: conjugate gradient residual increase." << std::endl);
		}
	    }
    
	    *tc_terminate = ( *tc_metric < hoCgTC::hoCg_->get_tc_tolerance() );
	    tc_last_ = *tc_metric;
	    return true;
	}
  
    protected:
	REAL rq_0_;
	REAL tc_last_;
    };

    template <class T, class REAL> class hoResidualTCB
	: public hoCgTerminationCallback<T, REAL>
    {

    protected:

	typedef hoCgTerminationCallback<T, REAL> hoCgTC;

    public:

	//typedef typename ARRAY_TYPE::element_type ELEMENT_TYPE;
	//typedef typename realType<ELEMENT_TYPE>::Type REAL;

	hoResidualTCB() : hoCgTerminationCallback<T, REAL>() {
	    tc_last_ = get_max<REAL>();
	}

	virtual ~hoResidualTCB() {}

	virtual bool initialize( hoCgSolver<T, REAL > *hoCg )
	{
	    hoCgTC::initialize(hoCg);
	    tc_last_ = get_max<REAL>();
	    return true;
	}

	virtual bool iterate( unsigned int iteration, REAL *tc_metric, bool *tc_terminate )
	{
	    *tc_metric = hoCgTC::get_rq();
	    if( hoCgTC::hoCg_->get_output_mode() >= solver<hoNDArray<T>,hoNDArray<T> >::OUTPUT_WARNINGS ) {
		if( hoCgTC::hoCg_->get_output_mode() >= solver<hoNDArray<T>,hoNDArray<T> >::OUTPUT_VERBOSE ) {
		    GDEBUG_STREAM("Iteration " << iteration << ". rq/rq_0 = " << *tc_metric << std::endl);
		}
		if( (tc_last_-(*tc_metric)) < REAL(0) ){
		    GDEBUG_STREAM("----- Warning: HOCG residual increase. Stability problem! -----" << std::endl);
		}
	    }
	    *tc_terminate = ( *tc_metric < hoCgTC::hoCg_->get_tc_tolerance() );
	    tc_last_ = *tc_metric;
	    return true;
	}

    protected:

	REAL tc_last_;
    };

    template <class T, class REAL > class hoUpdateTCB
	: public hoCgTerminationCallback<T, REAL >
    {

    protected:
	typedef hoCgTerminationCallback<T, REAL > hoCgTC;

    public:
	//typedef typename ARRAY_TYPE::element_type ELEMENT_TYPE;
	//typedef typename realType<ELEMENT_TYPE>::Type REAL;

	hoUpdateTCB() : hoCgTerminationCallback<T, REAL >() {

	    tc_last_ = get_max<REAL>();
	}

	virtual ~hoUpdateTCB() {}

	virtual bool initialize( hoCgSolver<T, REAL > *hoCg )
	{
	    hoCgTC::initialize(hoCg);
	    tc_last_ = get_max<REAL>();
	    return true;
	}

	virtual bool iterate( unsigned int iteration, REAL *tc_metric, bool *tc_terminate )
	{
	    *tc_metric = hoCgTC::hoCg_->solver_dot(hoCgTC::get_p().get(),hoCgTC::get_p().get());
	    if( hoCgTC::hoCg_->get_output_mode() >= solver<hoNDArray<T>,hoNDArray<T> >::OUTPUT_WARNINGS ) {
		if( hoCgTC::hoCg_->get_output_mode() >= solver<hoNDArray<T>,hoNDArray<T> >::OUTPUT_VERBOSE ) {
		    GDEBUG_STREAM("Iteration " << iteration << ". rq/rq_0 = " << *tc_metric << std::endl);
		}
		if( (tc_last_-(*tc_metric)) < REAL(0) ){
		    GDEBUG_STREAM("----- Warning: HOCG residual increase. Stability problem! -----" << std::endl);
		}
	    }
	    *tc_terminate = ( *tc_metric < hoCgTC::hoCg_->get_tc_tolerance() );
	    tc_last_ = *tc_metric;
	    return true;
	}

    protected:

	REAL tc_last_;
    };
}
