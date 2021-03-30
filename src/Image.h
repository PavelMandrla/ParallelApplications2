//
// Created by pavel on 29.03.21.
//

#ifndef PARALLELAPPLICATIONS2_IMAGE_H
#define PARALLELAPPLICATIONS2_IMAGE_H

#include <string>

using namespace std;

class Image {
private:
    float *dLinearPitchTextureData = nullptr;
    size_t texPitch;

    unsigned int imageWidth = 0;
    unsigned int imageHeight = 0;
    unsigned int imageBPP = 0;
    unsigned int imagePitch = 0;

    cudaChannelFormatDesc texChannelDesc;
    texture<float, 2, cudaReadModeElementType> texRef;

    unsigned char* loadSourceImage(const char* imageFileName);

    void createTextureFromLinearPitchMemory(unsigned char* dImageData);
public:
    explicit Image(const string& path);

    ~Image();

    unsigned int getImageWidth() const;

    unsigned int getImageHeight() const;

    size_t getTexPitch() const;

    unsigned int getImageBpp() const;

    unsigned int getImagePitch() const;

    const texture<float, 2, cudaReadModeElementType> &getTexRef() const;
};

#endif //PARALLELAPPLICATIONS2_IMAGE_H
