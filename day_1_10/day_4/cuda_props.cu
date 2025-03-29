#include <iostream>
#include <cuda.h>

int main(){
    int device_cout;
    cudaGetDeviceCount(&device_cout);

    cudaDeviceProp dev_prop;
    for (int i = 0; i < device_cout; i++){
        cudaGetDeviceProperties(&dev_prop, i);
        std::cout<<"Max number of threads per block: "<<dev_prop.maxThreadsPerBlock<<'\n';
        std::cout<<"Number of SM in the device: "<<dev_prop.multiProcessorCount<<'\n';
        std::cout<<"Clock rate: "<<dev_prop.clockRate<<'\n';

        std::cout<<"Max threads per blockDim x: "<<dev_prop.maxThreadsDim[0]<<'\n';
        std::cout<<"Max threads per blockDim y: "<<dev_prop.maxThreadsDim[1]<<'\n';
        std::cout<<"Max threads per blockDim z: "<<dev_prop.maxThreadsDim[2]<<'\n';

        std::cout<<"Max blocks per dimension x: "<<dev_prop.maxGridSize[0]<<'\n';
        std::cout<<"Max blocks per dimension y: "<<dev_prop.maxGridSize[1]<<'\n';
        std::cout<<"Max blocks per dimension z: "<<dev_prop.maxGridSize[2]<<'\n';
        std::cout<<"Warp size: "<<dev_prop.warpSize<<'\n';
        std::cout<<"Shared memory per block: "<<dev_prop.sharedMemPerBlock<<'\n';

    }

}