//
// Created by pavel on 13.04.21.
//
#include <cudaDefs.h>
#include <imageManager.h>
#include <imageUtils.cuh>
#include <string>
#include <iostream>

using namespace std;

cudaError_t error = cudaSuccess;
cudaDeviceProp deviceProp = cudaDeviceProp();

constexpr unsigned int TPB = 256;
using DT = uint32_t;

__global__ void initHistogram(unsigned int* histogram) {
    unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int jump = gridDim.x * blockDim.x;

    while (offset < 768) {
        histogram[offset] = 0;
        offset += jump;
    }
}

__global__ void calculateHistogram_0(const uint32_t* __restrict__ img, const uint32_t rWidth, const uint32_t rHeight, unsigned int* histograms) {
    const unsigned int dataSize = rWidth * rHeight;

    unsigned int* histR = histograms;
    unsigned int* histG = histograms + 256;
    unsigned int* histB = histograms + 512;

    unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t jump = gridDim.x * blockDim.x;

    uint32_t* pixelData = (uint32_t*) img + offset;

    while (offset < dataSize) {
        uchar4* pixel = (uchar4*) pixelData;
        atomicAdd(&histR[pixel->x], 1);
        atomicAdd(&histG[pixel->y], 1);
        atomicAdd(&histB[pixel->z], 1);

        pixelData += jump;
        offset += jump;
    }

}

int main() {
    initializeCUDA(deviceProp);

    FreeImage_Initialise();
    ImageInfo<DT> img;
    prepareData<false>("../../images/lena.png", img);

    unsigned int* dHistograms;
    checkCudaErrors(cudaMalloc((void**)&dHistograms, 768 * sizeof(unsigned int)));

    dim3 b(1, 1, 1);
    dim3 g(128, 1,1);
    initHistogram<<<b, g>>>(dHistograms);

    b = dim3((img.width * img.height + TPB - 1) / TPB, 1, 1);
    g = dim3(TPB, 1,1);

    auto gpuTime = GPUTIME(1, calculateHistogram_0<<<b, g>>>(img.dPtr, img.width, img.height, dHistograms));
    printf("\x1B[93m[GPU time] %s: %f ms\033[0m\n", "calculate histogram - global", gpuTime);

    unsigned int *hist = static_cast<unsigned int*>(::operator new (768 * sizeof(unsigned int)));
    cudaMemcpy(hist, dHistograms, 768 * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    cout << "red:" << endl;
    for (int i = 0; i < 256; i++) {
        cout << hist[i] << " ";
    }
    cout << endl << endl;

    cout << "green:" << endl;
    for (int i = 256; i < 512; i++) {
        cout << hist[i] << " ";
    }
    cout << endl << endl;

    cout << "blue:" << endl;
    for (int i = 512; i < 768; i++) {
        cout << hist[i] << " ";
    }
    cout << endl;

    FreeImage_DeInitialise();
    if (img.dPtr) cudaFree(img.dPtr);
    return 0;
}