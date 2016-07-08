//
// NFFT_H preprocessing kernels
//

#include "hoNDArray.h"
#include "vector_td.h"
#include "hoNDArray_elemwise.h"
#include "hoNDArray_utils.h"
#include "vector_td_utilities.h"
#include "vector_td_io.h"

//using namespace std;
using std::vector;
//using namespace thrust;
using namespace Gadgetron;

// convert input trajectory in [-1/2;1/2] to [0;matrix_size_os+matrix_size_wrap]

/**
template<class REAL, unsigned int D> struct trajectory_scale
{
  typename reald<REAL,D>::Type matrix, bias;
  
  trajectory_scale( const typename reald<REAL,D>::Type &m, const typename reald<REAL,D>::Type &b ){
    matrix = m;
    bias = b;
  }
  
  __host__ __device__
  typename reald<REAL,D>::Type operator()(const typename reald<REAL,D>::Type &in) const { 
    return component_wise_mul<REAL,D>(in,matrix)+bias;
  }
};



template<class T, unsigned int D> compute_num_cells_per_sample(T half_W) {
    unsigned int num_cells = 1;
    for( unsigned int d = 0; d < D; d++ ){
	unsigned int upper_limit = (unsigned int)  floor(  (  ( (float*)&p ) [dim]  )  +  half_W  );
	unsigned int lower_limit = (unsigned int)ceil((((float*)&p)[dim])-half_W);
	num_cells *= (upper_limit-lower_limit+1);
    }
    return num_cells;
}
*/



template<class T> void output_pairs( //NOT INCLUDING FRAMES
				    typename reald<T,1>::Type *traj_positions, 
				    int traj_length,
				    typename uintd<1>::Type matrix_size_os, 
				    typename uintd<1>::Type matrix_size_wrap, 
				    typename uintd<1>::Type *grid_cell_pos,
				    unsigned int *traj_idx,
				    T half_W, 
				    int num_cells )
{
    
    typename reald<T,1>::Type p;
    for (int i = 0; i < traj_length; i++){
	
	p = traj_positions[i];
	
	unsigned int lower_limit_x = (unsigned int)ceil(p.vec[0]-half_W); //Conv kernel limits
	unsigned int upper_limit_x = (unsigned int)floor(p.vec[0]+half_W);
	
	unsigned int count = 0;
	
	for( unsigned int x = lower_limit_x; x <= upper_limit_x; x++ ){	
	    typename uintd<1>::Type co; co.vec[0] = x;
	    
	    //grid_cell_pos[i*num_cells+count] = co_to_idx<1>(co, matrix_size_os+matrix_size_wrap);  //+frame_offset;
	    /**
	    typename uintd<1>::Type cell_pos;
	    int i_tmp = i;
	    for(int d = 0; d < 1; d++){
		cell_pos[d] = i_tmp % (matrix_size_os[d]+matrix_size_wrap[d]);
		i_tmp -= cell_pos[d];
		i_tmp /= (matrix_size_os[d]+matrix_size_wrap[d]);
	    }
	    */
	    grid_cell_pos[i*num_cells+count] = co;//cell_pos;
	    traj_idx[i*num_cells+count] = i;
	    count++;
	}
    }
}

template<class T> void output_pairs( //NOT INCLUDING FRAMES
				    typename reald<T,2>::Type *traj_positions, 
				    int traj_length,
				    typename uintd<2>::Type matrix_size_os, 
				    typename uintd<2>::Type matrix_size_wrap, 
				    typename uintd<2>::Type *grid_cell_pos,
				    unsigned int *traj_idx,
				    T half_W, 
				    int num_cells )
{
    
    typename reald<T,2>::Type p;
    for (int i = 0; i < traj_length; i++){
	
	p = traj_positions[i];

    	unsigned int lower_limit_x = (unsigned int)ceil(p.vec[0]-half_W); //Conv kernel limits
	unsigned int upper_limit_x = (unsigned int)floor(p.vec[0]+half_W);
	unsigned int lower_limit_y = (unsigned int)ceil(p.vec[1]-half_W); //Conv kernel limits
	unsigned int upper_limit_y = (unsigned int)floor(p.vec[1]+half_W);
	
	unsigned int count = 0;
	
	for( unsigned int x = lower_limit_x; x <= upper_limit_x; x++ ){	
	    for( unsigned int y = lower_limit_y; y <= upper_limit_y; y++ ){	
		typename uintd<2>::Type co; co.vec[0] = x; co.vec[1] = y;
		/**
		typename uintd<2>::Type cell_pos;// = co_to_idx<2>(co, matrix_size_os+matrix_size_wrap);;
		
		unsigned int idx = 0;
		unsigned int block_size = 1;
		for (unsigned int d=0; d<2; d++) {
		    idx += (block_size*co[d]);
		    block_size *= (matrix_size_os[d]+matrix_size_wrap[d]);
		}
		return idx;

		int i_tmp = i;
		for(int d = 0; d < 2; d++){
		    cell_pos[d] = i_tmp % (matrix_size_os[d]+matrix_size_wrap[d]);
		    i_tmp -= cell_pos[d];
		    i_tmp /= (matrix_size_os[d]+matrix_size_wrap[d]);
		    }
		*/
		grid_cell_pos[i*num_cells+count] = co;//cell_pos;
		traj_idx[i*num_cells+count] = i;
		count++;
	    }
	}
	if (i == ceil(traj_length/2)){
	    //std::cout << "i: " << i << std::endl;
	    //std::cout << "traj_pos: " << p << std::endl;
	    //std::cout << "x limits: " << lower_limit_x << ", " << upper_limit_x << std::endl;
	    //std::cout << "y limits: " << lower_limit_y << ", " << upper_limit_y << std::endl;
	}
    }
}

template<class T> void output_pairs( //NOT INCLUDING FRAMES
				    typename reald<T,3>::Type *traj_positions, 
				    int traj_length,
				    typename uintd<3>::Type matrix_size_os, 
				    typename uintd<3>::Type matrix_size_wrap, 
				    typename uintd<3>::Type *grid_cell_pos,
				    unsigned int *traj_idx,
				    T half_W, 
				    int num_cells )
{
    
    typename reald<T,3>::Type p;
    for (int i = 0; i < traj_length; i++){
	
	p = traj_positions[i];
	
	unsigned int lower_limit_x = (unsigned int)ceil(p.vec[0]-half_W); //Conv kernel limits
	unsigned int upper_limit_x = (unsigned int)floor(p.vec[0]+half_W);
	unsigned int lower_limit_y = (unsigned int)ceil(p.vec[1]-half_W); //Conv kernel limits
	unsigned int upper_limit_y = (unsigned int)floor(p.vec[1]+half_W);
	unsigned int lower_limit_z = (unsigned int)ceil(p.vec[2]-half_W); //Conv kernel limits
	unsigned int upper_limit_z = (unsigned int)floor(p.vec[2]+half_W);
	
	unsigned int count = 0;
	
	for( unsigned int x = lower_limit_x; x <= upper_limit_x; x++ ){	
	    for( unsigned int y = lower_limit_y; y <= upper_limit_y; y++ ){
		for( unsigned int z = lower_limit_z; z <= upper_limit_z; z++ ){		
		    typename uintd<3>::Type co; co.vec[0] = x; co.vec[1] = y; co.vec[2] = z;
		    /**
		    typename uintd<3>::Type cell_pos;
		    
		    int i_tmp = i;
		    for(int d = 0; d < 3; d++){
			cell_pos[d] = i_tmp % (matrix_size_os[d]+matrix_size_wrap[d]);
			i_tmp -= cell_pos[d];
			i_tmp /= (matrix_size_os[d]+matrix_size_wrap[d]);
		    }
		    */
		    grid_cell_pos[i*num_cells+count] = co;//cell_pos;
		    traj_idx[i*num_cells+count] = i;
		    count++;
		}
	    }
	}
    }
}

template<class T> void output_pairs( //NOT INCLUDING FRAMES
				    typename reald<T,4>::Type *traj_positions, 
				    int traj_length,
				    typename uintd<4>::Type matrix_size_os, 
				    typename uintd<4>::Type matrix_size_wrap, 
				    typename uintd<4>::Type *grid_cell_pos,
				    unsigned int *traj_idx,
				    T half_W, 
				    int num_cells )
{
    
    typename reald<T,4>::Type p;
    for (int i = 0; i < traj_length; i++){
	
	p = traj_positions[i];
	
	unsigned int lower_limit_x = (unsigned int)ceil(p.vec[0]-half_W); //Conv kernel limits
	unsigned int upper_limit_x = (unsigned int)floor(p.vec[0]+half_W);
	unsigned int lower_limit_y = (unsigned int)ceil(p.vec[1]-half_W); //Conv kernel limits
	unsigned int upper_limit_y = (unsigned int)floor(p.vec[1]+half_W);
	unsigned int lower_limit_z = (unsigned int)ceil(p.vec[2]-half_W); //Conv kernel limits
	unsigned int upper_limit_z = (unsigned int)floor(p.vec[2]+half_W);
	unsigned int lower_limit_w = (unsigned int)ceil(p.vec[3]-half_W); //Conv kernel limits
	unsigned int upper_limit_w = (unsigned int)floor(p.vec[3]+half_W);
	
	unsigned int count = 0;
	
	for( unsigned int x = lower_limit_x; x <= upper_limit_x; x++ ){	
	    for( unsigned int y = lower_limit_y; y <= upper_limit_y; y++ ){
		for( unsigned int z = lower_limit_z; z <= upper_limit_z; z++ ){
		    for( unsigned int w = lower_limit_w; w <= upper_limit_w; w++ ){	
			typename uintd<4>::Type co; co.vec[0] = x; co.vec[1] = y; co.vec[2] = z; co.vec[3] = w;
			/**
			typename uintd<4>::Type cell_pos;
			
			int i_tmp = i;
			for(int d = 0; d < 4; d++){
			    cell_pos[d] = i_tmp % (matrix_size_os[d]+matrix_size_wrap[d]);
			    i_tmp -= cell_pos[d];
			    i_tmp /= (matrix_size_os[d]+matrix_size_wrap[d]);
			}
			*/
			grid_cell_pos[i*num_cells+count] = co;//cell_pos;
			traj_idx[i*num_cells+count] = i;
			count++;
		    }
		}
	    }
	}
    }
}

