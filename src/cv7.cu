#include <iostream>
#include "Image.h"
#include "cudaDefs.h"

constexpr unsigned int QUERY_WIDTH = 512;

using namespace std;

typedef texture<float, 2, cudaReadModeElementType> tex;


__global__ void calculateDistances(int2* dRes, tex queryTex, int queryWidth, int queryHeight, tex refTex, int refWidth, int refHeight) {
    __shared__ float sDifferences[QUERY_WIDTH];
    unsigned int tid = threadIdx.x;

    if (tid == 0) dRes[blockIdx.x] = int2 {-1, -1};
    sDifferences[tid] = 0;
    if (tid >= queryWidth) return;

    unsigned int resultWidth = (refWidth - queryWidth + 1);
    unsigned int resultHeight = (refHeight - queryHeight + 1);

    unsigned int jump = gridDim.x;
    unsigned int rectY = 0;

    while (rectY < resultHeight) {
        unsigned int rectX = blockDim.x * blockIdx.x;

        while (rectX < resultWidth) {
            for (int dY = 0; dY < queryHeight; dY++) {
                sDifferences[threadIdx.x] += tex2D(refTex, rectX + tid, rectY + dY) + tex2D(refTex, rectX + tid, rectY + dY);
            }

            unsigned int next = QUERY_WIDTH;
            float* src = &sDifferences[threadIdx.x];
            float* src2;
            while (next > 32) { // ABOVE 32
                __syncthreads();
                next >>= 1;
                if (threadIdx.x < next) {
                    src2 = src + next;
                    *src += *src2;
                }
            }

            __syncthreads();
            while (next > 1) {      //32 AND DOWN
                next >>= 1;
                if (threadIdx.x < next) {
                    volatile float* vsrc = &sDifferences[threadIdx.x];
                    volatile float* vsrc2 = vsrc + next;
                    *vsrc += *vsrc2;
                }
            }

            if (threadIdx.x == 0 && sDifferences[0] == 0) {
                dRes[blockIdx.x] = int2{ (int) rectX, (int) rectY };
            }

            rectX += jump;
        }
        rectY += 1;
    }
}

int main() {
    Image reference("/home/pavel/tmp/reference.tif");
    Image query("/home/pavel/tmp/query.tif");

    KernelSetting ks;
    int blockCount = getNumberOfParts(
            (reference.getImageWidth() - query.getImageWidth() + 1) * (reference.getImageHeight() - query.getImageHeight() + 1),
            1000);
    ks.dimGrid = dim3(blockCount, 1, 1);
    ks.dimBlock = dim3(QUERY_WIDTH, 1, 1);

    cudaError_t error = cudaSuccess;

    int2* dResults;
    error = cudaMalloc((void**)&(dResults), blockCount * sizeof(int2));
/*
    calculateDistances<<<ks.dimGrid, ks.dimBlock>>>(
            dResults,
            query.getTexRef(),
            query.getImageWidth(),
            query.getImageHeight(),
            reference.getTexRef(),
            reference.getImageWidth(),
            reference.getImageHeight()
            );
*/
    int2 *hResults = static_cast<int2*>(::operator new (blockCount * sizeof(int2)));
    error = cudaMemcpy(hResults, dResults, blockCount * sizeof(int2), cudaMemcpyDeviceToHost);

    cudaFree(dResults);

    return 0;
}