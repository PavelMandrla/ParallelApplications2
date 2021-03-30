//
// Created by pavel on 29.03.21.
//

#include "Image.h"
#include <FreeImage.h>
#include "imageManager.h"
#include "cudaDefs.h"
#include "imageKernels.cuh"

Image::Image(const string& path) {
    unsigned char* dImageData = this->loadSourceImage(path.c_str());
    this->createTextureFromLinearPitchMemory(dImageData);
    cudaFree(dImageData);
    dImageData = nullptr;

}

Image::~Image() {
    cudaUnbindTexture(this->texRef);
    if (this->dLinearPitchTextureData)
        cudaFree(this->dLinearPitchTextureData);
}

unsigned char* Image::loadSourceImage(const char *imageFileName) {
    unsigned char *dImageData = nullptr;
    cudaError_t error = cudaSuccess;

    FreeImage_Initialise();
    FIBITMAP *tmp = ImageManager::GenericLoader(imageFileName, 0);

    imageWidth = FreeImage_GetWidth(tmp);
    imageHeight = FreeImage_GetHeight(tmp);
    imageBPP = FreeImage_GetBPP(tmp);
    imagePitch = FreeImage_GetPitch(tmp);		// FREEIMAGE align row data ... You have to use pitch instead of width

    error = cudaMalloc((void**)&dImageData, imagePitch * imageHeight);
    error = cudaMemcpy(dImageData, FreeImage_GetBits(tmp), imagePitch * imageHeight, cudaMemcpyHostToDevice);

    FreeImage_Unload(tmp);
    FreeImage_DeInitialise();
    return dImageData;
}

void Image::createTextureFromLinearPitchMemory(unsigned char* dImageData) {
    cudaMallocPitch((void**) &dLinearPitchTextureData, &texPitch, imageWidth * sizeof(int), imageHeight);

    KernelSetting ks;
    ks.dimBlock = dim3(8, 8, 1);
    ks.blockSize = 8 * 8;
    ks.dimGrid = dim3(
            (imageWidth + 8 - 1) / 8,
            (imageHeight + 8 - 1) / 8,
            1
            );

    switch (imageBPP) {
        case 8:
            colorToFloat<8><<<ks.dimGrid, ks.dimBlock>>>(dImageData, imageWidth, imageHeight, imagePitch, texPitch/sizeof(float), dLinearPitchTextureData);
            break;
        case 16:
            colorToFloat<16><<<ks.dimGrid, ks.dimBlock>>>(dImageData, imageWidth, imageHeight, imagePitch, texPitch/sizeof(float), dLinearPitchTextureData);
            break;
        case 24:
            colorToFloat<24><<<ks.dimGrid, ks.dimBlock>>>(dImageData, imageWidth, imageHeight, imagePitch, texPitch/sizeof(float), dLinearPitchTextureData);
            break;
        case 32:
            colorToFloat<32><<<ks.dimGrid, ks.dimBlock>>>(dImageData, imageWidth, imageHeight, imagePitch, texPitch/sizeof(float), dLinearPitchTextureData);
            break;
    }
    checkDeviceMatrix<float>(dLinearPitchTextureData, texPitch, 10, 10, "%6.1f ", "Result of Linear Pitch Text");

    //Define texture channel descriptor (texChannelDesc)
    texChannelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);// čti float z R, 0 z G, 0 z B, 0 z A interpretuj to jako float

    //Define texture (texRef) parameters
    texRef.normalized = false; // nechci normalizovany pristup
    texRef.filterMode = cudaFilterModePoint; // nechci, at jsou provadeny nejake interpolace
    texRef.addressMode[0] = cudaAddressModeClamp; // zasahnu-li mimo obraz v x, dostanu hranicni hodnotu
    texRef.addressMode[1] = cudaAddressModeClamp; // zasahnu-li mimo obraz v y, dostanu hranicni hodnotu

    //Bind texture
    cudaBindTexture2D(0, &texRef, dLinearPitchTextureData, &texChannelDesc, imageWidth, imageHeight, texPitch);
}

size_t Image::getTexPitch() const {
    return texPitch;
}

unsigned int Image::getImageBpp() const {
    return imageBPP;
}

unsigned int Image::getImagePitch() const {
    return imagePitch;
}

const texture<float, 2, cudaReadModeElementType> &Image::getTexRef() const {
    return texRef;
}

unsigned int Image::getImageWidth() const {
    return imageWidth;
}

unsigned int Image::getImageHeight() const {
    return imageHeight;
}





