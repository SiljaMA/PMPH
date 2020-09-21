#include <stdlib.h>
#include <stdio.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <time.h> 
#define GPU_RUNS 100

//Calculates (x/x-2.3)^3 serial for every x in the array using the cpu
void squareSerial(float* d_in, float* d_out, int N){
    for (unsigned int i = 0; i < N; ++i){
        float x = d_in[i]/(d_in[i]-2.3);
        d_out[i] = x*x*x;
        /*pow(d_in[i]/(d_in[i]-2.3), 3)*/
    }
}

//Calculates (x/x-2.3)^3 for every x in the array using the gpu
__global__ void squareKernel(float* d_in, float* d_out, int N){
    const unsigned int lid = threadIdx.x; 
    const unsigned int gid = blockIdx.x*blockDim.x + lid; 
    if(gid < N){
        float x = d_in[gid]/(d_in[gid]-2.3);
        d_out[gid] = x*x*x;
        /*pow(d_in[gid]/(d_in[gid]-2.3), 3)*/
    }
}

//Calculates the time different between two timevals
int timeval_substract(struct timeval* result, struct timeval* t2, struct timeval* t1){
    unsigned int resolution = 1000000; 
    long int diff = (t2 -> tv_usec + resolution * t2 -> tv_sec) - (t1 -> tv_usec + resolution * t1 -> tv_sec);
    result -> tv_sec = diff/resolution; 
    result -> tv_usec = diff % resolution; 
    return (diff <0); 
}


int main(int argc, char *argv[]){
unsigned int N; 
unsigned long int runs; 
    if(argc != 3){
        printf("Missing value for N and runs.\n"); 
        N = 100;//sæt tilbage til 753411
        runs = 2; 
        printf("Using default-values N = %d runs = %d\n", N, runs); 
    }else {
        N = atoi(argv[1]); 
        runs = atoi(argv[2]); 
    }

    bool cpuisfaster = true;

    while(cpuisfaster){
        unsigned long int cpu_time = 0; 
        unsigned long int gpu_time = 0; 
        for(int run = 1; run < runs; run++){
            unsigned int mem_size = N*sizeof(float); 
            unsigned int block_size = 256; 
            unsigned int num_blocks = ((N + (block_size -1))/block_size); 

            //For measure the time 
            unsigned long int elapsed_gpu; struct timeval t_start_gpu, t_end_gpu, t_diff_gpu;  
            unsigned long int elapsed_cpu; struct timeval t_start_cpu, t_end_cpu, t_diff_cpu;  

            //allocates memory for the arrays
            float* h_in = (float*) malloc(mem_size);
            float* gpu_res = (float*) malloc(mem_size);
            float* cpu_res = (float*) malloc(mem_size);

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


            //starts the time for the gpu
            gettimeofday(&t_start_gpu, NULL); 
            //execute the kernel and calculates the square using gpu 
            squareKernel <<<num_blocks, block_size>>>(d_in, d_out, N);
            //waits for the gpu to finish, so we can calculate the exact time 
            cudaDeviceSynchronize(); 
            //ends the time for the gpu
            gettimeofday(&t_end_gpu, NULL); 

            //copy result from device to host
            cudaMemcpy(gpu_res, d_out, mem_size, cudaMemcpyDeviceToHost);

            //starts the time for cpu
            gettimeofday(&t_start_cpu, NULL); 
            //Calculates squareSerial using the cpu
            squareSerial(h_in, cpu_res, N); 
            //ends the time for the cpu
            gettimeofday(&t_end_cpu, NULL); 


            //Checks the results are the same    
            int valid, invalid;
            valid = invalid = 0; 
            for (unsigned int j = 0; j < N; ++j){
                if(fabs(cpu_res[j] - gpu_res[j]) < 0.0001){
                    valid++;
                }else{
                    invalid++;
                }
            }
            //printf("Valid: %d, Invalid: %d \n", valid, invalid);

            //time for kernel gpu
            timeval_substract(&t_diff_gpu, &t_end_gpu, &t_start_gpu); 
            elapsed_gpu = (t_diff_gpu.tv_sec*1e6+t_diff_gpu.tv_usec); 
            printf("GPU took %d microseconds (%.2fms)\n", elapsed_gpu, elapsed_gpu/1000.0);

            //Time for serial on cpu
            timeval_substract(&t_diff_cpu, &t_end_cpu, &t_start_cpu); 
            elapsed_cpu = (t_diff_cpu.tv_sec*1e6+t_diff_cpu.tv_usec); 
            printf("CPU took %d microseconds (%.2fms)\n", elapsed_cpu, elapsed_cpu/1000.0);


            cpu_time = cpu_time + elapsed_cpu; 
            gpu_time = gpu_time + elapsed_gpu;

            //clean-up memory
            free(h_in);
            free(cpu_res);
            free(gpu_res); 
            cudaFree(d_in);
            cudaFree(d_out);

        }

        printf("Size of array %d \n", N); 
        printf("Number of runs %d \n", runs);
        printf("Average time for gpu: %.2fms\n", gpu_time/runs);
        printf("Average time for cpu: %.2fms\n", cpu_time/runs);

        if((cpu_time/runs) < (gpu_time/runs)){
            cpuisfaster = false;
        }else{
            N++; 
        }
    }

    printf("%d", N); 
    return 0; 
}