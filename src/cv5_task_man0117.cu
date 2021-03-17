//
// Created by pavel on 10.03.21.
//

#include <cudaDefs.h>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <limits>
#include <iomanip>

using namespace std;

cudaError_t error = cudaSuccess;

constexpr unsigned int PATTERN_SIZE = 16;
constexpr unsigned int REFERENCE_SIZE = 1 << 16;

constexpr unsigned int MBPTB = 8;
constexpr unsigned int TPB = 128;

struct TmpRes {
    float value;
    unsigned int index;
};

__constant__ float dPattern[PATTERN_SIZE];

vector<float> getRandVector(unsigned int size) {
    random_device rd;
    mt19937_64 mt(rd());
    uniform_real_distribution<float> dist(0.0f, 1.0f);

    vector<float> result(size);
    generate(result.begin(), result.end(), [&dist, &rd]() { return dist(rd); });
    return result;
}

__global__ void initTmpRes(TmpRes* tmpRes, const float value,unsigned int length) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int jump = gridDim.x * blockDim.x;

    while (i < length) {
        tmpRes[i].index = i;
        tmpRes[i].value = value;
        i += jump;
    }
}

__global__ void calculateSimilarities(float * dReference, TmpRes * dTmpRes, int length) {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int jump = gridDim.x * blockDim.x;

    while (idx < length) {
        float sum = 0.0;
        //#pragma unroll
        for (int i = 0; i < PATTERN_SIZE; i++) {
            sum += pow(dPattern[i] - dReference[idx + i], 2);
            __syncthreads();
        }
        dTmpRes[idx].value = sqrt(sum);
        idx += jump;
    }
}

__global__ void reduce(TmpRes * dIn, TmpRes* dOut) {
    __shared__ TmpRes sDistances[TPB];

    TmpRes* start = &dIn[2 * blockDim.x * blockIdx.x];
    unsigned int tid = threadIdx.x;
    unsigned int next = TPB;

    TmpRes* src = &sDistances[tid];
    TmpRes* src2;

    *src = start[tid];
    src2 = &start[tid + next];

    if (src2->value < src->value) {
        *src = *src2;
    }

    __syncthreads();
    next >>= 1;                 //64
    if (tid >= next) return;
    src2 = src + next;
    if (src2->value < src->value) {
        *src = *src2;
    }

    __syncthreads();
    while (next > 1) {      //32 AND DOWN
        next >>= 1;
        if (tid >= next) return;
        volatile TmpRes* vsrc = &sDistances[tid];
        volatile TmpRes* vsrc2 = vsrc + next;
        if (vsrc2->value < vsrc->value) {
            vsrc->value = vsrc2->value;
            vsrc->index = vsrc2->index;
        }
    }

    if (tid == 0)
        dOut[blockIdx.x] = sDistances[0];
}

int main(int argc, char *argv[]) {
    auto hPattern = getRandVector(PATTERN_SIZE);
    auto hReference = getRandVector(REFERENCE_SIZE);

    float hResArr[PATTERN_SIZE];

    cudaMemcpyToSymbol(static_cast<const void*>(&dPattern), static_cast<void*>(hPattern.data()), hPattern.size() * sizeof(float));

    float* dReference = nullptr;
    TmpRes* dTmpRes1 = nullptr;
    TmpRes* dTmpRes2 = nullptr;

    error = cudaMalloc((void**)&dReference, REFERENCE_SIZE * sizeof(float));
    error = cudaMemcpy(dReference, hReference.data(), REFERENCE_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    unsigned int count1 = (((REFERENCE_SIZE - (PATTERN_SIZE - 1)) / 256) + 1) * 256;
    error = cudaMalloc((void**)&dTmpRes1, count1 * sizeof(TmpRes));

    unsigned int count2 = ((count1 / (2 * TPB)) / 256 + 1) * 256;
    error = cudaMalloc((void**)&dTmpRes2, count2 * sizeof(TmpRes));

    KernelSetting ksInitTmpRes1;
    ksInitTmpRes1.dimBlock = dim3(TPB, 1, 1);
    ksInitTmpRes1.dimGrid = dim3(getNumberOfParts(count1, TPB * MBPTB), 1, 1);
    initTmpRes<<<ksInitTmpRes1.dimGrid, ksInitTmpRes1.dimBlock>>>(dTmpRes1,  std::numeric_limits<float>::max(), count1);

    KernelSetting ksInitTmpRes2;
    ksInitTmpRes1.dimBlock = dim3(TPB, 1, 1);
    ksInitTmpRes1.dimGrid = dim3(getNumberOfParts(count2, TPB * MBPTB), 1, 1);
    initTmpRes<<<ksInitTmpRes2.dimGrid, ksInitTmpRes2.dimBlock>>>(dTmpRes2, std::numeric_limits<float>::max(), count2);

    KernelSetting ksCalcSim;
    ksCalcSim.dimBlock = dim3(TPB, 1, 1);
    ksCalcSim.dimGrid = dim3(getNumberOfParts(REFERENCE_SIZE, TPB * MBPTB), 1, 1);
    calculateSimilarities<<<ksCalcSim.dimBlock, ksCalcSim.dimGrid>>>(dReference, dTmpRes1, REFERENCE_SIZE - PATTERN_SIZE + 1);

    KernelSetting ksReduce;
    ksReduce.dimBlock = dim3(TPB, 1, 1);
    ksReduce.dimGrid = dim3(count1 / (2 * ksReduce.dimBlock.x), 1, 1);

    TmpRes* dT1 = dTmpRes1;
    TmpRes* dT2 = dTmpRes2;
    unsigned int dataCount = count1;
    while (ksReduce.dimGrid.x > 1) {
        reduce<<<ksReduce.dimGrid, ksReduce.dimBlock>>>(dT1, dT2);

        TmpRes reduceResult[count2];
        cudaMemcpy(&reduceResult, dT2, count2 * sizeof(TmpRes), cudaMemcpyDeviceToHost);

        std::swap(dT1, dT2);
        dataCount = ksReduce.dimGrid.x;
        ksReduce.dimGrid.x = dataCount / (2 * ksReduce.dimBlock.x) + (dataCount % ksReduce.dimBlock.x ? 1 : 0);
    }
    reduce<<<ksReduce.dimGrid, ksReduce.dimBlock>>>(dT1, dT2);

    TmpRes reduceResult[1];
    cudaMemcpy(&reduceResult, dTmpRes1, sizeof(TmpRes), cudaMemcpyDeviceToHost);

    float result[PATTERN_SIZE];
    cudaMemcpy(&result, &dReference[reduceResult[0].index], PATTERN_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    cout << fixed << setprecision(3);
    cout << "Most similar vector to the pattern: \t\t";
    for (float i : hPattern)
        cout << i << " ";
    cout << endl << "was found in position: " << reduceResult[0].index << " with value:\t";
    for (float x : result)
        cout << x << " ";
    cout << endl << "the distance between them is: " << reduceResult[0].value;
    cout << endl;

    cudaFree(dReference);
    cudaFree(dTmpRes1);
    cudaFree(dTmpRes2);

    return 0;
}