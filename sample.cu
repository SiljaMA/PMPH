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
    unsigned long int elapsed_gpu; struct timeval t_start_gpu, t_end_gpu, t_diff_gpu;  
    unsigned long int elapsed_cpu; struct timeval t_start_cpu, t_end_cpu, t_diff_cpu;  

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

    gettimeofday(&t_start_gpu, NULL); 
    //execute the kernel and calculates the square using gpu 
    squareKernel <<<num_blocks, block_size>>>(d_in, d_out, N);
    gettimeofday(&t_end_gpu, NULL); 

    //copy result from device to host
    cudaMemcpy(h_out, d_out, mem_size, cudaMemcpyDeviceToHost);

    //print result
    //for(unsigned int i = 0; i <N; i++) printf("%.6f\n", h_out[i]);

    float* cpu_res = (float*) malloc(mem_size);

    gettimeofday(&t_start_cpu, NULL); 
    //Calculates squareSerial using the cpu
    squareSerial(h_in, cpu_res, N); 
    gettimeofday(&t_start_cpu, NULL); 


    //Checks the results are the same     
    for (unsigned int j = 0; j < N; ++j){
        if(fabs(cpu_res[j] - h_out[j]) < 0.0001){
            printf("VALID \n");
        }else{
            printf("INVALID\n");
        }
    }

    //time for kernel gpu
    timeval_substract(&t_diff_gpu, &t_end_gpu, &t_start_gpu); 
    elapsed_gpu = (t_diff_gpu.tv_sec*1e6+t_diff_gpu.tv_usec); 
    printf("GPU took %d microseconds (%.2fms)\n", elapsed_gpu, elapsed_gpu/1000.0);
  
    //Time for serial on cpu
    timeval_substract(&t_diff_cpu, &t_end_cpu, &t_start_cpu); 
    elapsed_cpu = (t_diff_cpu.tv_sec*1e6+t_diff_cpu.tv_usec); 
    printf("CPU took %d microseconds (%.2fms)\n", elapsed_cpu, elapsed_cpu/1000.0);

    //clean-up memory
    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);

}