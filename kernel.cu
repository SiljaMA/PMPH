#include "cuda_runtime.h"
#include "device_launch_parameters.h"


//global laver en funktion om til en kernel - noget der kører på cpu'en 
//bliver executet med <<<dim3>>>arguments. 
__global__ void addIndex(int* a, int* b, int* c){
    //index, der fortæller hvilken thread vi executer kode i
    //hver thread executer en blok
    int i = threadIdx.x; 
    c[i] = a[i] + b[i]; 
}


//device kører på gpuen og kan kun kaldes derfra 

//host kører på cpu'en 


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

    //kopier koden fra ha til da osv - dvs koden fra cpu'en til gpu'en
    cudaMemcpy(da, ha, size, cudaMemcpyKind:: cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, size, cudaMemcpyKind:: cudaMemcpyHostToDevice);

    //fortæller hvor mange blokke gpuen skal køre med
    addArrays<<<1, count>>>(da, db, dc);

    //kopiere resultet tilbage
    cudaMemcpy(hc, dc, size, cudaMemcpyKind:: cudaMemcpyDeviceToHost);



}