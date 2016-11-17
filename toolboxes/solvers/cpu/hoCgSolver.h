/** \file hoCgSolver.h
    \brief Instantiation of the conjugate gradient solver on the cpu.

    The file hoCgSolver.h is a convienience wrapper for the device independent cgSolver class.
    The class hoCgSolver instantiates the cgSolver for the hoNDArray
    and the header otherwise includes other neccessary header files.
*/

#pragma once

#include "cgSolver.h"
#include "hoNDArray_math.h"

#include "linearOperatorSolver.h"
#include "hoCgCallback.h"
#include "cgPreconditioner.h"
#include "real_utilities.h"
#include "complext.h"

#include <vector>
#include <iostream>
#include <limits>

namespace Gadgetron{

    /** \class hoCgSolver
	\brief Instantiation of the conjugate gradient solver on the cpu.
	
	The class hoCgSolver is a convienience wrapper for the device independent cgSolver class.
	hoCgSolver instantiates the cgSolver for type hoNDArray<T>.
    */
    template <class T, class REAL> class hoCgSolver : public cgSolver<hoNDArray<T> >
    {
    public:
	hoCgSolver() : cgSolver<hoNDArray<T> >() {
	    std::cout << "In hoCgSolver costructor" << std::endl;
	    alpha_ = std::numeric_limits<T>::quiet_NaN();
	    iterations_ = 10;
	    tc_tolerance_ = (REAL)1e-3;
	    cb_ = boost::shared_ptr< hoRelativeResidualTCB<T,REAL> >( new hoRelativeResidualTCB<T,REAL>() );
	}
	virtual ~hoCgSolver() {}



	virtual void set_preconditioner( boost::shared_ptr< cgPreconditioner<hoNDArray<T> > > precond ) {
	    precond_ = precond;
	}
  

	virtual void set_termination_callback( boost::shared_ptr< hoCgTerminationCallback<T,REAL> > cb ){
	    cb_ = cb;
	}
  

	virtual void set_max_iterations( unsigned int iterations ) { iterations_ = iterations; }
	virtual unsigned int get_max_iterations() { return iterations_; }  


	virtual void set_tc_tolerance( REAL tolerance ) { tc_tolerance_ = tolerance; }
	virtual REAL get_tc_tolerance() { return tc_tolerance_; }
  


	virtual void solver_dump( hoNDArray<T> *in, unsigned int it ) {
	    hoNDArray<std::complex<T> > go(in->get_dimensions(), (std::complex<T>*) in->get_data_ptr());
	    std::string savename = "/mnt/scratch/karen/go/solve_output" + std::to_string(it) + ".cplx";
	    write_nd_array(&go, savename.c_str());

	}



	// ============================================== //
	// ================= Main method ================ //
	// ============================================== //
	
	virtual boost::shared_ptr<hoNDArray<T> > solve( hoNDArray<T> *d )
	{	    

	    boost::shared_ptr<hoNDArray<T> > rhs = compute_rhs( d );
	    boost::shared_ptr<hoNDArray<T> > result =  solve_from_rhs( rhs.get() );
	    return result;
	}
	

	
	virtual boost::shared_ptr<hoNDArray<T> > solve_from_rhs( hoNDArray<T>  *rhs ) 
	{
	    if( iterations_ == 0 ){
		return boost::shared_ptr<hoNDArray<T> >( new hoNDArray<T> (*rhs) );
	    }
	    
	    initialize(rhs);
	    
	    if( this->output_mode_ >= solver<hoNDArray<T>, hoNDArray<T> >::OUTPUT_VERBOSE ){
		GDEBUG_STREAM("Iterating..." << std::endl);
	    }
	    
	    for( unsigned int it=0; it<iterations_; it++ ){
		
		REAL tc_metric;
		bool tc_terminate;
		
		this->iterate( it, &tc_metric, &tc_terminate );
		
		solver_dump( x_.get(), it);
		
		if( tc_terminate )
		    break;
	    }

	    boost::shared_ptr<hoNDArray<T> > tmpx = x_;
	    deinitialize();
	    return tmpx;
	}
	
	
	virtual boost::shared_ptr<hoNDArray<T> > compute_rhs( hoNDArray<T> *d )
	{
    
	    if( this->encoding_operator_.get() == 0 ){
		throw std::runtime_error( "Error: hoCgSolver::compute_rhs : no encoding operator is set" );
	    } 

	    boost::shared_ptr< std::vector<size_t> > image_dims = this->encoding_operator_->get_domain_dimensions();
	    if( image_dims->size() == 0 ){
		throw std::runtime_error( "Error: hoCgSolver::compute_rhs : encoding operator has not set domain dimension" );
	    }

	    boost::shared_ptr<hoNDArray<T> > result = boost::shared_ptr<hoNDArray<T> >(new hoNDArray<T>(image_dims.get()));
	    clear(result.get());

	    hoNDArray<T> tmp(image_dims.get() );

	    this->encoding_operator_->mult_MH( d, &tmp );

	    axpy(T(this->encoding_operator_->get_weight()), &tmp, result.get() );
    
	    return result;
	}	
	


	//TEMPORARY SOLUTION!!

	REAL rq_;
	REAL rq0_;
	//T alpha_;
	//boost::shared_ptr<hoNDArray<T> > x_, p_, r_;




 protected:

	virtual void initialize( hoNDArray<T> *rhs )
	{
	    if( !rhs || rhs->get_number_of_elements() == 0 ){
		throw std::runtime_error( "Error: hoCgSolver::initialize : empty or NULL rhs provided" );
	    }

	    x_ = boost::shared_ptr<hoNDArray<T> >( new hoNDArray<T>(rhs->get_dimensions()) );

	    r_ = boost::shared_ptr<hoNDArray<T> >( new hoNDArray<T>(*rhs) );
	    p_ = boost::shared_ptr<hoNDArray<T> >( new hoNDArray<T>(*r_) );
    
	    if( !this->get_x0().get() ){ // no starting image provided      
		clear(x_.get());
	    }

	    // Apply preconditioning, twice (should change preconditioners to do this)
	    if( precond_.get() ) {	
		precond_->apply( p_.get(), p_.get() );
		precond_->apply( p_.get(), p_.get() );
	    }

	    rq0_ = real(dot( r_.get(), p_.get() ));

	    std::cout << "rp0_ = " << rq0_ << std::endl;

	    if (this->get_x0().get()){
	
		if( !this->get_x0()->dimensions_equal( rhs )){
		    throw std::runtime_error( "Error: hoCgSolver::initialize : RHS and initial guess must have same dimensions" );
		}
	
		*x_ = *(this->get_x0());
        
		hoNDArray<T> mhmX( rhs->get_dimensions());

		if( this->output_mode_ >= solver<hoNDArray<T>, hoNDArray<T> >::OUTPUT_VERBOSE ) {
		    GDEBUG_STREAM("Preparing guess..." << std::endl);
		}
        
		mult_MH_M( this->get_x0().get(), &mhmX );
        
		*r_ -= mhmX;
		*p_ = *r_;
        
		// Apply preconditioning, twice (should change preconditioners to do this)
		if( precond_.get() ){
		    precond_->apply( p_.get(), p_.get() );
		    precond_->apply( p_.get(), p_.get() );
		}
	    }
      
	    rq_ = real( dot( r_.get(), p_.get() ));

	    std::cout << "rp_ = " << rq_ << std::endl;
      
	    cb_->initialize(this);
	 }




	virtual void deinitialize()
	{
	    p_.reset();
	    r_.reset();
	    x_.reset();
	}




	virtual void iterate( unsigned int iteration, REAL *tc_metric, bool *tc_terminate )
	{
	    hoNDArray<T> q = hoNDArray<T>(x_->get_dimensions());

	    mult_MH_M( p_.get(), &q );
    
	    // Update solution
	    //

	    alpha_ = rq_/dot( p_.get(), &q );
	    axpy( alpha_, p_.get(), x_.get());

	    // Update residual
	    //

	    axpy( -alpha_, &q, r_.get());

	    // Apply preconditioning
	    //

	    if( precond_.get() ){

		precond_->apply( r_.get(), &q );
		precond_->apply( &q, &q );
        
		REAL tmp_rq = real(dot( r_.get(), &q ));      
		*p_ *= T((tmp_rq/rq_));
		axpy( T(1), &q, p_.get() );
		rq_ = tmp_rq;
	    } 
	    else{
        
		REAL tmp_rq = real(dot( r_.get(), r_.get()) );
		*p_ *= T((tmp_rq/rq_));           
		axpy( T(1), r_.get(), p_.get() );
		rq_ = tmp_rq;      
	    }

	    std::cout << "rq_ = " << rq_ << std::endl;
	    std::cout << "tc_metric = " << *tc_metric << std::endl;
      
	    // Invoke termination callback iteration
	    //

	    if( !cb_->iterate( iteration, tc_metric, tc_terminate ) ){
		throw std::runtime_error( "Error: hoCgSolver::iterate : termination callback iteration failed" );
	    }    
	 }




	void mult_MH_M( hoNDArray<T> *in, hoNDArray<T> *out )
	{
	    // Basic validity checks
	    //

	    if( !in || !out ){
		throw std::runtime_error( "Error: hoCgSolver::mult_MH_M : invalid input pointer(s)" );
	    }

	    if( in->get_number_of_elements() != out->get_number_of_elements() ){
		throw std::runtime_error( "Error: hoCgSolver::mult_MH_M : array dimensionality mismatch" );
	    }
    
	    // Intermediate storage
	    //

	    hoNDArray<T> q = hoNDArray<T>(in->get_dimensions());

	    // Start by clearing the output
	    //
	    clear(out);

	    // Apply encoding operator
	    //

	    this->encoding_operator_->mult_MH_M( in, &q, false );
	    axpy( this->encoding_operator_->get_weight(), &q, out );

	    // Iterate over regularization operators
	    //

	    for( unsigned int i=0; i<this->regularization_operators_.size(); i++ ){      
		this->regularization_operators_[i]->mult_MH_M( in, &q, false );
		axpy( this->regularization_operators_[i]->get_weight(), &q, out );
	    }      
	 }




protected:

	// Preconditioner
	boost::shared_ptr< cgPreconditioner<hoNDArray<T> > > precond_;

	// Termination criterium callback
	boost::shared_ptr< hoCgTerminationCallback<T,REAL> > cb_;

	// Termination criterium threshold
	REAL tc_tolerance_;

	// Maximum number of iterations
	unsigned int iterations_;

	// Internal variables. 
	//REAL rq_;
	//REAL rq0_;
	T alpha_;
	boost::shared_ptr<hoNDArray<T> > x_, p_, r_;


	
    };
}
