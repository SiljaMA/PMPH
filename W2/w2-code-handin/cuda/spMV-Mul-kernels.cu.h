#ifndef SP_MV_MUL_KERS
#define SP_MV_MUL_KERS

__global__ void
replicate0(int tot_size, char* flags_d) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < tot_size){
        flags_d[i] = 0; 
    }
}

__global__ void
mkFlags(int mat_rows, int* mat_shp_sc_d, char* flags_d) {  
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x; 
    if(i < mat_rows){
        if(i == 0){
            flags_d[i] = 1; 
        }else {
            int index = mat_shp_sc_d[i-1];  
            flags_d[index] = 1; 
        }
    }
}

__global__ void 
mult_pairs(int* mat_inds, float* mat_vals, float* vct, int tot_size, float* tmp_pairs) {
   unsigned int i = blockIdx.x*blockDim.x + threadIdx.x; 
   if(i < tot_size){
       int vec_ind = mat_inds[i]; 
       tmp_pairs[i] = mat_vals[i] * vct[vec_ind]; 
   }
}

__global__ void
select_last_in_sgm(int mat_rows, int* mat_shp_sc_d, float* tmp_scan, float* res_vct_d) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x; 
    if (i < mat_rows){
        int seg_index = mat_shp_sc_d[i]; 
        res_vct_d[i] = tmp_scan[seg_index - 1];  
    }
}

#endif
