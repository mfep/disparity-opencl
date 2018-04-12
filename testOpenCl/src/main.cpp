#include <iostream>
#include <cstdint>
#include <string>
#include <fstream>
#include <streambuf>
#include "CL/cl.hpp"
#include "lodepng.h"


void testCl() {
	//get all platforms (drivers)
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    if(all_platforms.size()==0){
        std::cout<<" No platforms found. Check OpenCL installation!\n";
        exit(1);
    }
    cl::Platform default_platform=all_platforms[1];
    std::cout << "Using platform: "<<default_platform.getInfo<CL_PLATFORM_NAME>()<<"\n";

    //get default device of the default platform
    std::vector<cl::Device> all_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if(all_devices.size()==0){
        std::cout<<" No devices found. Check OpenCL installation!\n";
        exit(1);
    }
    cl::Device default_device=all_devices[0];
    std::cout<< "Using device: "<<default_device.getInfo<CL_DEVICE_NAME>()<<"\n";


    cl::Context context({default_device});

    cl::Program::Sources sources;

    // kernel calculates for each element C=A+B
    std::string kernel_code=
            "   void kernel simple_add(global const int* A, global const int* B, global int* C){       "
            "       C[get_global_id(0)]=A[get_global_id(0)]+B[get_global_id(0)];                 "
            "   }                                                                               ";
    sources.push_back({kernel_code.c_str(),kernel_code.length()});

    cl::Program program(context,sources);
    if(program.build({default_device})!=CL_SUCCESS){
        std::cout<<" Error building: "<<program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device)<<"\n";
        exit(1);
    }

    // create buffers on the device
    cl::Buffer buffer_A(context,CL_MEM_READ_WRITE,sizeof(int)*10);
    cl::Buffer buffer_B(context,CL_MEM_READ_WRITE,sizeof(int)*10);
    cl::Buffer buffer_C(context,CL_MEM_READ_WRITE,sizeof(int)*10);

    int A[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int B[] = {0, 1, 2, 0, 1, 2, 0, 1, 2, 0};

    //create queue to which we will push commands for the device.
    cl::CommandQueue queue(context,default_device);

    //write arrays A and B to the device
    queue.enqueueWriteBuffer(buffer_A,CL_TRUE,0,sizeof(int)*10,A);
    queue.enqueueWriteBuffer(buffer_B,CL_TRUE,0,sizeof(int)*10,B);

    //run the kernel
    //cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer> simple_add(cl::Kernel(program,"simple_add"),queue,cl::NullRange,cl::NDRange(10),cl::NullRange);
    //simple_add(buffer_A,buffer_B,buffer_C);

    //alternative way to run the kernel
    cl::Kernel kernel_add=cl::Kernel(program,"simple_add");
    kernel_add.setArg(0,buffer_A);
    kernel_add.setArg(1,buffer_B);
    kernel_add.setArg(2,buffer_C);
    queue.enqueueNDRangeKernel(kernel_add,cl::NullRange,cl::NDRange(10),cl::NullRange);
    queue.finish();

    int C[10];
    //read result C from the device to array C
    queue.enqueueReadBuffer(buffer_C,CL_TRUE,0,sizeof(int)*10,C);

    std::cout<<" result: \n";
    for(int i=0;i<10;i++){
        std::cout<<C[i]<<" ";
    }
	getchar();
}


cl::Context initCl() {
	//get all platforms (drivers)
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    if(all_platforms.size()==0){
        std::cout<<" No platforms found. Check OpenCL installation!\n";
        exit(1);
    }
    cl::Platform default_platform=all_platforms[0];
    std::cout << "Using platform: "<<default_platform.getInfo<CL_PLATFORM_NAME>()<<"\n";

    //get default device of the default platform
    std::vector<cl::Device> all_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if(all_devices.size()==0){
        std::cout<<" No devices found. Check OpenCL installation!\n";
        exit(1);
    }
    cl::Device default_device=all_devices[0];
    std::cout<< "Using device: "<<default_device.getInfo<CL_DEVICE_NAME>()<<"\n";
	return cl::Context({default_device});
}


std::string readFile(const char* filename) {
	std::ifstream t(filename);
	std::string str;

	t.seekg(0, std::ios::end);
	str.reserve(t.tellg());
	t.seekg(0, std::ios::beg);

	str.assign((std::istreambuf_iterator<char>(t)),
				std::istreambuf_iterator<char>());
	return str;
}


int main() {
	std::vector<uint8_t> pixels;
	unsigned width, height;
	unsigned error = lodepng::decode(pixels, width, height, "im0.png", LCT_RGBA);
	if (error != 0) {
		std::cout << "cannot read in im0.png" << std::endl;
		getchar();
		exit(1);
	}
	auto clCtx = initCl();
	int clError = 0;
	cl::Image2D clInImg(clCtx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, cl::ImageFormat(CL_RGBA, CL_UNSIGNED_INT8), width, height, 0, pixels.data(), &clError);
	if (clError != 0) {
		std::cout << "could not create openCL image from png error: " << clError << std::endl;
		getchar();
		exit(1);
	}
	cl::Image2D clOutImg(clCtx, CL_MEM_READ_WRITE, cl::ImageFormat(CL_INTENSITY, CL_FLOAT), width, height, 0, nullptr, &clError);
	if (clError != 0) {
		std::cout << "could not create openCL image for output error: " << clError << std::endl;
		getchar();
		exit(1);
	}

	auto programText = readFile("preprocess.cl");
	cl::Program program(clCtx, programText);
	clError = program.build();
	if (clError != 0) {
		std::cout << "could not build clProgram: " << clError << std::endl;
		getchar();
		exit(1);
	}
	cl::Kernel kernel(program, "preprocess", &clError);
	if (clError != 0) {
		std::cout << "initialize cl::Kernel: " << clError << std::endl;
		getchar();
		exit(1);
	}
	kernel.setArg(0, clInImg);
    kernel.setArg(1, clOutImg);
	cl::CommandQueue queue(clCtx);
    clError = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(width, height), cl::NullRange);
	if (clError != 0) {
		std::cout << "cannot add kernel to command queue: " << clError << std::endl;
		getchar();
		exit(1);
	}
	queue.finish();

	std::vector<float> processedImage(pixels.size());
	cl::size_t<3> size;
	size[0] = width;
	size[1] = height;
	size[2] = 1;
	clError = queue.enqueueReadImage(clOutImg, CL_TRUE, cl::size_t<3>(), size, 0, 0, processedImage.data());
	if (clError != 0) {
		std::cout << "cannot read computed image: " << clError << std::endl;
		getchar();
		exit(1);
	}
    queue.finish();

	std::vector<uint8_t> outputImage(processedImage.size());
	for (size_t i = 0; i < processedImage.size(); i++) {
		outputImage[i] = static_cast<uint8_t>(processedImage[i]);
	}

	error = lodepng::encode("out.png", outputImage, width, height, LCT_GREY, 8);
	if (error != 0) {
		std::cout << "cannot save image file" << std::endl;
		getchar();
		exit(1);
	}
    return 0;
}