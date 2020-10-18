#include <iostream>
#include <stdlib.h> 
#include <cuda.h> 
#include <cuda_runtime.h>
#include "curand.h"


int main(){
    curandGenerator_t gen; 
    curandGenerator(&gen, CURAND_RNG_PSEUDO_MTGP, 32); 
    

}
