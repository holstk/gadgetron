#include "KaiserBessel_operators.h"


//template<class T> KaiserBessel_operators<T>::KaiserBessel_operators()
//{
//}

//
// Kaiser-Bessel convolution kernels
//


template<class T> T KaiserBessel_operators<T>::bessi0(T x)
{
   T denominator;
   T numerator;
   T z;

   if (x == 0.0) {
      return 1.0;
   } 
   else {
      z = x * x;
      numerator = (z* (z* (z* (z* (z* (z* (z* (z* (z* (z* (z* (z* (z* 
                     (z* 0.210580722890567e-22  + 0.380715242345326e-19 ) +
                         0.479440257548300e-16) + 0.435125971262668e-13 ) +
                         0.300931127112960e-10) + 0.160224679395361e-7  ) +
                         0.654858370096785e-5)  + 0.202591084143397e-2  ) +
                         0.463076284721000e0)   + 0.754337328948189e2   ) +
                         0.830792541809429e4)   + 0.571661130563785e6   ) +
                         0.216415572361227e8)   + 0.356644482244025e9   ) +
                         0.144048298227235e10);

      denominator = (z*(z*(z-0.307646912682801e4)+
                       0.347626332405882e7)-0.144048298227235e10);
   }

   return -numerator/denominator;
 }


// Kaiser Bessel according to Beatty et. al. IEEE TMI 2005;24(6):799-808.
// There is a slight difference wrt Jackson's formulation, IEEE TMI 1991;10(3):473-478.

template<class T> T KaiserBessel_operators<T>::KaiserBessel( T u, T matrix_size_os, T one_over_W, T beta )
{
  T _tmp = 2.0*u*one_over_W;
  T tmp = _tmp*_tmp;
  T arg = beta*std::sqrt(1.0-tmp);
  T bessi = bessi0(arg);
  T ret = matrix_size_os*bessi*one_over_W;
  return ret;
}


//
// Below the intended interface
//

template<class T> T KaiserBessel_operators<T>::KaiserBessel( const Gadgetron::vector_td<T,1> &u, const Gadgetron::vector_td<T,1> &matrix_size_os, T one_over_W, const vector_td<T,1> &beta )
{
    T phi_x = KaiserBessel( u.vec[0], matrix_size_os.vec[0], one_over_W, beta[0] );
    return phi_x;
}





template<class T> T KaiserBessel_operators<T>::KaiserBessel( const Gadgetron::vector_td<T,2> &u, const Gadgetron::vector_td<T,2> &matrix_size_os, T one_over_W, const vector_td<T,2> &beta )
{
  T phi_x = KaiserBessel( u.vec[0], matrix_size_os.vec[0], one_over_W, beta[0] );
  T phi_y = KaiserBessel( u.vec[1], matrix_size_os.vec[1], one_over_W, beta[1] );
  return phi_x*phi_y;
}

template<class T> T KaiserBessel_operators<T>::KaiserBessel( const Gadgetron::vector_td<T,3> &u, const Gadgetron::vector_td<T,3> &matrix_size_os, T one_over_W, const vector_td<T,3> &beta )
{
  T phi_x = KaiserBessel( u.vec[0], matrix_size_os.vec[0], one_over_W, beta[0] );
  T phi_y = KaiserBessel( u.vec[1], matrix_size_os.vec[1], one_over_W, beta[1] );
  T phi_z = KaiserBessel( u.vec[2], matrix_size_os.vec[2], one_over_W, beta[2] );
  return phi_x*phi_y*phi_z;
}

template<class T> T KaiserBessel_operators<T>::KaiserBessel( const Gadgetron::vector_td<T,4> &u, const Gadgetron::vector_td<T,4> &matrix_size_os, T one_over_W, const vector_td<T,4> &beta )
{
  T phi_x = KaiserBessel( u.vec[0], matrix_size_os.vec[0], one_over_W, beta[0] );
  T phi_y = KaiserBessel( u.vec[1], matrix_size_os.vec[1], one_over_W, beta[1] );
  T phi_z = KaiserBessel( u.vec[2], matrix_size_os.vec[2], one_over_W, beta[2] );
  T phi_w = KaiserBessel( u.vec[3], matrix_size_os.vec[3], one_over_W, beta[3] );
  return phi_x*phi_y*phi_z*phi_w;
}

template class KaiserBessel_operators<float>;
template class KaiserBessel_operators<double>;
