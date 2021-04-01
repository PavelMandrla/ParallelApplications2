#include <cudaDefs.h>
#include <imageManager.h>
#include <imageUtils.cuh>

cudaError_t error = cudaSuccess;
cudaDeviceProp deviceProp = cudaDeviceProp();

using DT = float;

TextureInfo createTextureObjectFrom2DArray(const ImageInfo<DT>& ii) {
    TextureInfo ti;

    // Size info
    ti.size = { ii.width, ii.height, 1 };

    //Texture Data settings
    ti.texChannelDesc = cudaCreateChannelDesc<float>();  // cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);
    checkCudaErrors(cudaMallocArray(&ti.texArrayData, &ti.texChannelDesc, ii.width, ii.height));
    checkCudaErrors(cudaMemcpyToArray(ti.texArrayData, 0, 0, ii.dPtr, ii.pitch * ii.height, cudaMemcpyDeviceToDevice));

    // Specify texture resource
    ti.resDesc.resType = cudaResourceTypeArray;
    ti.resDesc.res.array.array = ti.texArrayData;

    // Specify texture object parameters
    ti.texDesc.addressMode[0] = cudaAddressModeClamp;
    ti.texDesc.addressMode[1] = cudaAddressModeClamp;
    ti.texDesc.filterMode = cudaFilterModePoint;
    ti.texDesc.readMode = cudaReadModeElementType;
    ti.texDesc.normalizedCoords = false;

    // Create texture object
    checkCudaErrors(cudaCreateTextureObject(&ti.texObj, &ti.resDesc, &ti.texDesc, nullptr));

    return ti;
}

ImageInfo<uchar3> allocateNormalMap(ImageInfo<DT> hMap) {
    ImageInfo<uchar3> nMap {
            hMap.width,
            hMap.height,
            hMap.pitch,
            nullptr
    };
    cudaMallocPitch((void**)&nMap.dPtr, &nMap.pitch, hMap.width * sizeof(uchar3), hMap.height);
    return nMap;
}

__device__ float    getSobelX(const cudaTextureObject_t &tHMap, float x, float y) {
    float res = 0.0f;
    res -= tex2D<float>(tHMap, x-1, y-1);
    res -= 2 * tex2D<float>(tHMap, x-1, y);
    res -= tex2D<float>(tHMap, x-1, y+1);

    res += tex2D<float>(tHMap, x+1, y-1);
    res += 2 * tex2D<float>(tHMap, x+1, y);
    res += tex2D<float>(tHMap, x+1, y+1);

    return res;
}

__device__ float getSobelY(const cudaTextureObject_t &tHMap, float x, float y) {
    float res = 0.0f;
    res -= tex2D<float>(tHMap, x-1, y+1);
    res -= 2 * tex2D<float>(tHMap, x, y+1);
    res -= tex2D<float>(tHMap, x+1, y+1);

    res += tex2D<float>(tHMap, x-1, y-1);
    res += 2 * tex2D<float>(tHMap, x, y-1);
    res += tex2D<float>(tHMap, x+1, y-1);

    return res;
}

__device__ float3 normalizeVec(float x, float y, float z) {
    float l = sqrt(x*x + y*y + z*z);
    return float3 {x/l, y/l, z/l};
}

__device__ uchar3 getNormal(const cudaTextureObject_t &tHMap, unsigned int x, unsigned int y) {
    auto normalizedNorm = normalizeVec(
            getSobelX(tHMap, (float) x, (float) y),
            getSobelY(tHMap, (float) x, (float) y),
            5);

    //SHIFT TO UPPER RIGHT QUADRANT
    normalizedNorm.x = (normalizedNorm.x + 1.0f) / 2.0f;
    normalizedNorm.y = (normalizedNorm.y + 1.0f) / 2.0f;
    //normalizedNorm.z = (normalizedNorm.z + 1.0f) / 2.0f;

    // CONVERT FLOAT VECTOR INTO UCHAR;
    return uchar3 {
        (unsigned char) (normalizedNorm.z * 255.0f),
        (unsigned char) (normalizedNorm.y * 255.0f),
        (unsigned char) (normalizedNorm.x * 255.0f)
    };
}

__global__ void calculateSobel(const cudaTextureObject_t tHMap, const ImageInfo<uchar3> nMap) {
    uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t tCount = gridDim.x * blockDim.x;
    uint32_t dataSize = nMap.width * nMap.height;

    while (tid < dataSize) {
        unsigned int x = tid % nMap.width;
        unsigned int y = tid / nMap.width;

        auto rowStart = (uchar3*) ((char*) nMap.dPtr + y * nMap.pitch);
        rowStart[x] = getNormal(tHMap, x, y);
        tid += tCount;
    }

}

void saveTexImage(const char* imageFileName, const uint32_t dstWidth, const uint32_t dstHeight, const uint32_t dstPitch, const uchar3* dstData) {
    FIBITMAP* tmp = FreeImage_Allocate(dstWidth, dstHeight, 24);
    unsigned int tmpPitch = FreeImage_GetPitch(tmp);					// FREEIMAGE align row data ... You have to use pitch instead of width
    checkCudaErrors(cudaMemcpy2D(FreeImage_GetBits(tmp), tmpPitch, dstData, dstPitch, dstWidth * 3, dstHeight, cudaMemcpyDeviceToHost));
    ImageManager::GenericWriter(tmp, imageFileName, FIF_BMP);
    FreeImage_Unload(tmp);
}


int main() {
    initializeCUDA(deviceProp);
    FreeImage_Initialise();

    ImageInfo<DT> hMap;
    ImageInfo<uchar3> nMap;
    prepareData<false>("/home/pavel/prg/cpp/ParallelApplications2/images/terrain3Kx3K.tif", hMap);
    nMap = allocateNormalMap(hMap);

    TextureInfo tiHMap = createTextureObjectFrom2DArray(hMap);

    dim3 block { 10, 1 ,1 };
    dim3 grid { 64, 1, 1 };

    calculateSobel<<<block, grid>>>(tiHMap.texObj, nMap);

    auto hNMap = static_cast<uchar3*>(::operator new (nMap.width * nMap.height * sizeof(uchar3)));
    error = cudaMemcpy2D(hNMap, sizeof(uchar3) * nMap.width, nMap.dPtr, nMap.pitch, nMap.width, nMap.height, cudaMemcpyDeviceToHost);

   // saveTexImage("/home/pavel/res.tif", nMap.width, nMap.height, sizeof(uchar3) * nMap.width, hNMap);
    saveTexImage("/home/pavel/res.tif", nMap.width, nMap.height, nMap.pitch, nMap.dPtr);

    /*
    int *m = static_cast<int*>(::operator new (sizeof(int)*rows*cols));;
    cudaMemcpy2D(m, sizeof(int) * cols, dM, dPitch, cols * sizeof(int), rows, cudaMemcpyDeviceToHost);
    */
    if (tiHMap.texObj) checkCudaErrors(cudaDestroyTextureObject(tiHMap.texObj));
    if (tiHMap.texArrayData) checkCudaErrors(cudaFreeArray(tiHMap.texArrayData));

    FreeImage_DeInitialise();
    if (hMap.dPtr) cudaFree(hMap.dPtr);
    if (nMap.dPtr) cudaFree(nMap.dPtr);
    return 0;
}
