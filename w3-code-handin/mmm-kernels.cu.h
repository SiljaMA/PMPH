#ifndef MULT_KERNELS
#define MULT_KERNELS

// widthA = heightB
template <class ElTp> 
__global__ void matMultKer(ElTp* A, ElTp* B, ElTp* C, int heightA, int widthB, int widthA) {
  ElTp accum = 0.0f;

  int gidx = blockIdx.x*blockDim.x + threadIdx.x;
  int gidy = blockIdx.y*blockDim.y + threadIdx.y; 

  if( (gidx >= widthB) || (gidy >= heightA) ) return;

  for(int k = 0; k < widthA; k ++) {
      accum += A[gidy*widthA + k] * B[k*widthB + gidx];
  }

  C[gidy*widthB + gidx] = accum;
}


// widthA = heightB
template <class ElTp, int T> 
__global__ void matMultTiledKer(ElTp* A, ElTp* B, ElTp* C, int heightA, int widthB, int widthA) {
  ElTp accum = 0.0f;

  __shared__ ElTp Ash[T][T];
  __shared__ ElTp Bsh[T][T];

  int gidx = blockIdx.x*blockDim.x + threadIdx.x;
  int gidy = blockIdx.y*blockDim.y + threadIdx.y; 

  for(int kk = 0; kk < widthA; kk += T) {
      Ash[threadIdx.y][threadIdx.x] = ((gidy < heightA) && (kk+threadIdx.x < widthA)) ?
            A[gidy*widthA + kk + threadIdx.x] : 0.0;
      Bsh[threadIdx.y][threadIdx.x] = ((gidx < widthB)  && (kk+threadIdx.y < widthA)) ?
            B[(threadIdx.y+kk)*widthB + gidx] : 0.0;
      __syncthreads();

      #pragma unroll
      for(int k = 0; k < T; k++)
          accum += Ash[threadIdx.y][k] * Bsh[k][threadIdx.x];
      __syncthreads();
  }

  if( (gidx < widthB) && (gidy < heightA) )
    C[gidy*widthB + gidx] = accum;
}

// widthA = heightB
template <class ElTp, int T> 
__global__ void matMultCacheKer(ElTp* A, ElTp* B, ElTp* C, int heightA, int widthB, int widthA) {
  ElTp accum = 0.0f;

  int gidx = blockIdx.x*blockDim.x + threadIdx.x;
  int gidy = blockIdx.y*blockDim.y + threadIdx.y; 

  for(int kk = 0; kk < widthA; kk += T) {
      __syncthreads();
      #pragma unroll
      for(int k = 0; k < T; k++)
        accum += A[gidy*widthA + kk + k] * B[gidy*widthB + (kk+k)];
  }

  if( (gidx < widthB) && (gidy < heightA) )
    C[gidy*widthB + gidx] = accum;
}


template <class ElTp, int T> 
__global__ void matMultRegTiledKer(ElTp* A, ElTp* B, ElTp* C, int heightA, int widthB, int widthA) {
    // ToDo: fill in the kernel implementation of register+block tiled 
    //       matrix-matrix multiplication here

  int ii = blockIdx.y * T; 
  int jjj = blockIdx.x * T*T; 
  int jj = threadIdx.y * T + jjj; 
  int j = jj + threadIdx.x; 
  int ty = threadIdx.y; 
  int tx = threadIdx.x; 
  __shared__ float Ash[T][T]; 

  float cs[T]; 
  for(int i = 0; i < T; i++){
    cs[i] = 0.0f; 
  }

  for(int kk = 0; kk < widthA; kk+= T){
    // Læs fra slice
    // A[ii:ii+T, kk:kk+T]
    int rowA = ii + ty; 
    int colA = kk + tx; 
    if((rowA < heightA) && (colA < widthB)){
      Ash[ty][tx] = A[widthA * rowA + colA];
    }
    else{
      Ash[ty][tx] = 0.0f; 
    }
    __syncthreads(); 
    for(int k = 0; k < T; k++){
      int rowB = kk + k;
      float b = 0.0f;
      if ((rowB < widthA) && (j < widthB)){
        b = B[rowB * widthB + j]; 
      }
      for(int i = 0; i < T; i++){
        cs[i] += Ash[i][k] * b;
      }
    }
    __syncthreads();
  
  }
  for(int i = 0; i < T; i++){
    int rowC = ii + i; 
    if (rowC < heightA && j < widthB){
      C[widthB * (rowC) + j] = cs[i]; 
    }
  }
}


#endif
