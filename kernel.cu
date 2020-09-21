#include "cuda_runtime.h"
#include "device_launch_parameters.h"



__global__ void addIndex(int* a, int* b, int* c){
    //index, der fortæller hvilken thread vi executer kode i
    //hver thread executer en blok
    int i = threadIdx.x; 
    c[i] = a[i] + b[i]; 
}


void main(){

    //memory på vores host
    const int count = 5; 
    int ha[] = {1, 2, 3, 4, 5}; 
    int hb[] = {10, 20, 30, 40, 50}; 
    int hc[count]; 

    //den plads vi skal bruge på gpu'en til at execute koden for da, db og dc
    const size = count * sizeof(int);
    int *da, *db, *dc; 
    cudaMalloc(&da, size); 
    cudaMalloc(&db, size); 
    cudaMalloc(&dc, size); 

    //kopier koden fra ha til da osv
    cudaMemcpy(da, ha, size, cudaMemcpyKind:: cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, size, cudaMemcpyKind:: cudaMemcpyHostToDevice);
    cudaMemcpy(dc, hc, size, cudaMemcpyKind:: cudaMemcpyHostToDevice);



    //simulere processen 
    for(int i = 0; i <count; ++i){
        addIndex(a, b, c); 
    }


}