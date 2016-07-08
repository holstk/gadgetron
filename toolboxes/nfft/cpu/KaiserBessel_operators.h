#ifndef KAISERBESSEL_OPERATORS_H
#define KAISERBESSEL_OPERATORS_H

#include "hoNDArray.h"
#include "hoNDFFT.h"
#include "hoNDArray_fileio.h"

using namespace Gadgetron;

template <class T> class KaiserBessel_operators
{
public:
    //KaiserBessel_operators();
    //~KaiserBessel_operators();
    
    T bessi0(T x); 

    T denominator;
    T numerator;
    T z;

   


// Kaiser Bessel according to Beatty et. al. IEEE TMI 2005;24(6):799-808.
// There is a slight difference wrt Jackson's formulation, IEEE TMI 1991;10(3):473-478.

    T KaiserBessel( T u, T matrix_size_os, T one_over_W, T beta );
	
//
// Below the intended interface
//
   
    T KaiserBessel( const Gadgetron::vector_td<T,1> &u, const Gadgetron::vector_td<T,1> &matrix_size_os, T one_over_W, const vector_td<T,1> &beta );

    T KaiserBessel( const Gadgetron::vector_td<T,2> &u, const Gadgetron::vector_td<T,2> &matrix_size_os, T one_over_W, const vector_td<T,2> &beta );

    T KaiserBessel( const Gadgetron::vector_td<T,3> &u, const Gadgetron::vector_td<T,3> &matrix_size_os, T one_over_W, const vector_td<T,3> &beta );

    T KaiserBessel( const Gadgetron::vector_td<T,4> &u, const Gadgetron::vector_td<T,4> &matrix_size_os, T one_over_W, const vector_td<T,4> &beta );

};


#endif //KAISERBESSEL
