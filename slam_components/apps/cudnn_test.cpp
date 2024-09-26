#include <iostream>
#include <cudnn.h>

int main() {
    cudnnHandle_t cudnn;
    cudnnStatus_t status = cudnnCreate(&cudnn);
    
    if (status != CUDNN_STATUS_SUCCESS) {
        std::cerr << "cuDNN initialization failed: " << cudnnGetErrorString(status) << std::endl;
        return 1;
    }

    int version = cudnnGetVersion();
    std::cout << "cuDNN version: " << version << std::endl;

    // Clean up
    // cudnnDestroy(cudnn);

    return 0;
}
