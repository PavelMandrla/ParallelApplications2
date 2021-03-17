#include <iostream>
#include <cudaDefs.h>

struct myStruct {
    int a;
    float b;
};


__constant__ int dVal;

__constant__ myStruct dStruct;

__constant__ int dValArr[3];



int main() {
    int hVal = 5;
    int hRes;

    cudaMemcpyToSymbol(static_cast<const void*>(&dVal), static_cast<void*>(&hVal), sizeof(myStruct));
    cudaMemcpyFromSymbol(static_cast<void*>(&hRes), static_cast<const void*>(&dVal), sizeof(myStruct));


    cudaMemcpyToSymbol(dVal, &hVal, sizeof(int));
    cudaMemcpyFromSymbol(&hRes, dVal, sizeof(int));
    std::cout << hRes << std::endl;

    myStruct hStruct {1, 2.0f};
    myStruct hStructRes;
    cudaMemcpyToSymbol(static_cast<const void*>(&dStruct), static_cast<void*>(&hStruct), sizeof(myStruct));
    cudaMemcpyFromSymbol(static_cast<void*>(&hStructRes), static_cast<const void*>(&dStruct), sizeof(myStruct));

    int hValArr[] = {10, 20, 30};
    int hResArr[3];
    cudaMemcpyToSymbol(static_cast<const void*>(&dValArr), static_cast<void*>(&hValArr), sizeof(hValArr));
    cudaMemcpyFromSymbol(static_cast<void*>(hResArr), static_cast<const void*>(&dValArr), 3*sizeof(int));



    for (int i = 0; i < 3; i++) {
        std::cout << hResArr[i] << std::endl;
    }
    return 0;
}