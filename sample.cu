#include <stdlib.h>
#include <stdio.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <time.h> 

#define GPU_RUNS 100

void squareSerial(float* d_in, float* d_out, int N){
    for (unsigned int i = 0; i < N; ++i){
        d_out[i] = pow(d_in[i]/(d_in[i]-2.3), 3);
    }
}

__global__ void squareKernel(float* d_in, float* d_out, int N){
    const unsigned int lid = threadIdx.x; 
    const unsigned int gid = blockIdx.x*blockDim.x + lid; 
    if(gid < N){
        d_out[gid] = pow(d_in[gid]/(d_in[gid]-2.3), 3);
    }
}


int timeval_substract(struct timeval* result, struct timeval* t2, struct timeval* t1){
    unsigned int resolution = 1000000; 
    long int diff = (t2 -> tv_usec + resolution * t2 -> tv_sec) - (t1 -> tv_usec + resolution * t1 -> tv_sec);
    result -> tv_sec = diff/resolution; 
    result -> tv_usec = diff % resolution; 
    return (diff <0); 
}

int main(int argc, char** argv){
    unsigned int N = 753411; //størrelsen på arrayet
    unsigned int mem_size = N*sizeof(float); //størrelsen på hukommelsen der skal bruges til arrayet
    unsigned int block_size = 256; //størrelsen på en block
    unsigned int num_blocks = ((N + (block_size -1))/block_size); //antallet af blocks


    //For measure the time 
    unsigned long int elaped; struct timeval t_start, t_end, t_diff; 
    gettimeofday(t_start, NULL); 

    //allocates host-memory
    float* h_in = (float*) malloc(mem_size);
    float* h_out = (float*) malloc(mem_size);

    //initialize the memory
    for(unsigned int i = 0; i <N; ++i){
            h_in[i] = float(i);
    }

    //allocate device memory
    float* d_in;
    float* d_out;
    cudaMalloc((void**)&d_in, mem_size);
    cudaMalloc((void**)&d_out, mem_size);

    //copy host memory to device
    cudaMemcpy(d_in, h_in, mem_size, cudaMemcpyHostToDevice);

    //execute the kernel and calculates the square using gpu 
    for(int i = 0; i < GPU_RUNS; i++){
        squareKernel <<<num_blocks, block_size>>>(d_in, d_out, N);
    }cudaThreadSynchronize(); 

    gettimeofday(&t_diff, NULL); 
    timeval_substract(&t_diff, &t_end, &t_start); 
    elaped = (t_diff.tv_sec*1e6+t_diff.tv_usec)/GPU_RUNS; 
    printf("Took %d microseconds (%.2fms)\n", elaped, elaped/1000.0);

    //copy result from device to host
    cudaMemcpy(h_out, d_out, mem_size, cudaMemcpyDeviceToHost);

    //print result
    //for(unsigned int i = 0; i <N; i++) printf("%.6f\n", h_out[i]);

    //Calculates squareSerial using the cpu
    float* cpu_res = (float*) malloc(mem_size);
    squareSerial(h_in, cpu_res, N); 

    //Checks the results are the same
     
    for (unsigned int j = 0; j < N; ++j){
        if(fabs(cpu_res[j] - h_out[j]) < 0.0001){
            printf("VALID \n");
        }else{
            printf("INVALID\n");
        }
    }
    

    //mål tiden 
    //undersøg hvornår gpuen bliver hurtigere 


    //clean-up memory
    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);

}