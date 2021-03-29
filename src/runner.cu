//
// Created by pavel on 16.02.21.
//

#include <cudaDefs.h>
#include <iostream>
#include <benchmark.h>

cudaError error = cudaSuccess;
cudaDeviceProp deviceProp = cudaDeviceProp();

constexpr unsigned int TPB = 256;    //threads per block
constexpr unsigned int NOB = 16;    //number of blockss
//constexpr unsigned int MBPTB = 2; //memory block per thread block -> posouvame se po vektoru dat -> kolikrat chci, at se posune

using namespace std;

__global__ void add1(const int* __restrict__ a, const int* __restrict__ b, const unsigned int length, int* __restrict__ c) { //__restrict__ nijak to nebude datove zasahovat do vektoru b - > nebuou mit spolecnou pamet
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int jump = gridDim.x * blockDim.x;

    while (i < length) {
        c[i] = a[i] + b[i];
        i += jump;
    }
}

__global__ void fill(int* __restrict__ m, unsigned int rows, unsigned int cols, size_t pitch) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < rows && y < cols) {
        int *rowStart = (int*)((char*) m + x * pitch);
        rowStart[y] = x + y;
    }
}

int main() {
    initializeCUDA(deviceProp);

    const int rows = 15;
    const int cols = 20;

    //init host memory
    int *dM = nullptr;
    size_t dPitch;
    cudaMallocPitch((void**)&dM, &dPitch, rows * sizeof(int), cols);

    dim3 dimBlock((rows / 8) + 1, (cols / 8) + 1);
    dim3 dimGrid(8, 8, 1); // velke mnozstvi bloku -> nevyhoda
    fill<<<dimBlock, dimGrid>>>(dM, rows, cols, dPitch);

    checkDeviceMatrix<int>(dM, dPitch, rows, cols, "%d ", "Device M");  //helper metoda na výpis dat
    cout << endl;

    int *m = static_cast<int*>(::operator new (sizeof(int)*rows*cols));;
    cudaMemcpy2D(m, sizeof(int) * cols, dM, dPitch, cols * sizeof(int), rows, cudaMemcpyDeviceToHost);
    checkHostMatrix(m, sizeof(int) * cols, rows, cols, "%d ", "Host M");

    //free memory
    delete[] m;
    cudaFree(dM); dM = nullptr;

    return 0;
}


/*
int main() {
    initializeCUDA(deviceProp);

    constexpr unsigned int length = 1000;
    const unsigned int sizeInBytes = length * sizeof(int);

    int *a = static_cast<int*>(::operator new (sizeInBytes)); //C++ paralela k mallocu
    int *b = static_cast<int*>(::operator new (sizeInBytes));
    int *c = static_cast<int*>(::operator new (sizeInBytes));

    for (int i = 0; i < length; i++) {
        a[i] = 1;//rand();
        b[i] = 2;//rand();
    }

    //allocate device memory
    int* da = nullptr;
    int* db = nullptr;
    int* dc = nullptr;
    checkCudaErrors(cudaMalloc((void**)&da, sizeInBytes));
    checkCudaErrors(cudaMalloc((void**)&db, sizeInBytes));
    checkCudaErrors(cudaMalloc((void**)&dc, sizeInBytes));

    //copy data from host to device
    cudaMemcpy(da, a, sizeInBytes, cudaMemcpyKind::cudaMemcpyHostToDevice);
    checkDeviceMatrix<int>(da, sizeInBytes, 1, length, "%d ", "Device A");

    cudaMemcpy(db, b, sizeInBytes, cudaMemcpyKind::cudaMemcpyHostToDevice);
    checkDeviceMatrix<int>(db, sizeInBytes, 1, length, "%d ", "Device B");  //helper metoda na výpis dat

    //Prepare grid and blocks
    dim3 dimBlock(TPB, 1, 1);

    //dim3 dimGrid(NOB, 1, 1); // velke mnozstvi bloku -> nevyhoda
    dim3 dimGrid(getNumberOfParts(length, TPB), 1, 1); // velke mnozstvi bloku -> nevyhoda
    //dim3 dimGrid(getNumberOfParts(length, TPB * MBPTB), 1, 1);

    add1<<<dimGrid, dimBlock>>>(da, db, length, dc);

    cudaMemcpy(c, dc, sizeInBytes, cudaMemcpyDeviceToHost);
    checkHostMatrix(c, sizeInBytes, 1, length, "%d ", "Host C");

    //free memory
    delete[] a;
    delete[] b;
    delete[] c;

    cudaFree(da); da = nullptr;
    cudaFree(db); db = nullptr;
    cudaFree(dc); dc = nullptr;

    return 0;
}
*/


int main() {
    const int width = 10;
    const int height = 10;

    initializeCUDA(deviceProp);
    //constexpr unsigned int length = 1 << 20;
    constexpr unsigned int length = 1000;
    const unsigned int sizeInBytes = length * sizeof(int);

    //init host memory
    int *dM = nullptr;
    size_t pitch;
    cudaMallocPitch((void**)&dM, &pitch, width * sizeof(int), height * sizeof(int));

    dim3 dimBlock(width / 8, height / 8);
    dim3 dimGrid(8, 8, 1); // velke mnozstvi bloku -> nevyhoda
    fill<<<dimBlock, dimGrid>>>(dM, pitch);


    //cudaMemcpy2D();


    //int *a = (int*)malloc(sizeInBytes);   //the C way
    //int *a = new int[length]; //volá interní konstruktory
    int *a = static_cast<int*>(::operator new (sizeInBytes)); //C++ paralela k mallocu
    int *b = static_cast<int*>(::operator new (sizeInBytes));
    int *c = static_cast<int*>(::operator new (sizeInBytes));

    //init data
    for (int i = 0; i < length; i++) {
        a[i] = 1;// rand();
        b[i] = 2;//rand();
        c[i] = 3;
    }

    //allocate device memory
    int* da = nullptr;
    int* db = nullptr;
    int* dc = nullptr;
    checkCudaErrors(cudaMalloc((void**)&da, sizeInBytes));
    checkCudaErrors(cudaMalloc((void**)&db, sizeInBytes));
    checkCudaErrors(cudaMalloc((void**)&dc, sizeInBytes));

    //copy data from host to device
    cudaMemcpy(da, a, sizeInBytes, cudaMemcpyKind::cudaMemcpyHostToDevice);
    checkDeviceMatrix<int>(da, sizeInBytes, 1, length, "%d ", "Device A");

    cudaMemcpy(db, b, sizeInBytes, cudaMemcpyKind::cudaMemcpyHostToDevice);
    checkDeviceMatrix<int>(db, sizeInBytes, 1, length, "%d ", "Device B");  //helper metoda na výpis dat

    cudaMemcpy(dc, c, sizeInBytes, cudaMemcpyKind::cudaMemcpyHostToDevice);
    checkDeviceMatrix<int>(dc, sizeInBytes, 1, length, "%d ", "Device C");  //helper metoda na výpis dat

    //Prepare grid and blocks
    //dim3 dimBlock(TPB, 1, 1);
    //dim3 dimGrid(NOB, 1, 1); // velke mnozstvi bloku -> nevyhoda

    //dim3 dimGrid(getNumberOfParts(length, TPB), 1, 1); // velke mnozstvi bloku -> nevyhoda
    //dim3 dimGrid(getNumberOfParts(length, TPB * MBPTB), 1, 1);

    //Call kernel
    using gpubenchmark::print_ns;
    using gpubenchmark::print_time;

    auto test = [&]() {add1<<<dimGrid, dimBlock>>>(da, db, length, dc);};   //obalim do lambda fce
    print_time("Test ADD", test, 10);

    cudaDeviceSynchronize();
    checkDeviceMatrix<int>(dc, sizeInBytes, 1, length, "%d ", "Device C");  //helper metoda na výpis dat

    //check results
    cudaMemcpy(c, db, sizeInBytes, cudaMemcpyDeviceToHost);
    checkHostMatrix(c, sizeInBytes, 1, length, "%d ", "Host C");
    for (int i = 0; i < 20; i++) {
        cout << c[i] << " ";
    }
    cout << endl;


    //free memory
    delete[] a;
    delete[] b;
    delete[] c;

    cudaFree(da); da = nullptr;
    cudaFree(db); db = nullptr;
    cudaFree(dc); dc = nullptr;
}