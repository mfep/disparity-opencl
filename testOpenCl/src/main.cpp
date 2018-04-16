#include <iostream>
#include <cstdint>
#include <string>
#include <fstream>
#include <streambuf>
#include "CL/cl.hpp"
#include "lodepng.h"
#include "Logger.hpp"


constexpr int WINDOW = 9;

namespace {

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


template<typename T>
void error_quit_program(T error) {
	if (error != 0) {
		getchar();
		exit(1);
	}
}


cl::Kernel loadKernel(const cl::Context& clCtx, const char* filename, const char* kernelname) {
	auto programText = readFile(filename);
	cl::Program prepProgram(clCtx, programText);
	int clError = prepProgram.build();
	Logger::logOpenClError(clError, "build preprocess cl program");
	error_quit_program(clError);

	cl::Kernel kernel(prepProgram, kernelname, &clError);
	Logger::logOpenClError(clError, filename);
	error_quit_program(clError);
	return kernel;
}


cl::Image2D createGrayClImage(const cl::Context& clCtx, unsigned width, unsigned height) {
	int clError = 0;
	cl::Image2D clImg(clCtx, CL_MEM_READ_WRITE, cl::ImageFormat(CL_LUMINANCE, CL_FLOAT), width, height, 0, nullptr, &clError);
	Logger::logOpenClError(clError, "create OpenCL image for preprocess");
	error_quit_program(clError);
	return clImg;
}


void runKernel(const cl::CommandQueue& queue, const cl::Kernel& kernel, const cl::NDRange& globalRange, const char* progressname) {
		Logger::startProgress(progressname);
		int clError = queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalRange, cl::NullRange);
		Logger::logOpenClError(clError, "add kernel to command queue");
		error_quit_program(clError);
		queue.finish();
		Logger::endProgress();
}

}	// namespace


int main() {
	// load input image
	std::vector<uint8_t> pixels;
	unsigned width, height;
	unsigned error = lodepng::decode(pixels, width, height, "im0.png", LCT_RGBA);
	Logger::logLoad(error, "im0.png");
	error_quit_program(error);

	// initialize OpenCL
	auto clCtx = initCl();
	cl::CommandQueue queue(clCtx);
	int clError = 0;

	// create input OpenCL image
	cl::Image2D clInImg(clCtx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, cl::ImageFormat(CL_RGBA, CL_UNSIGNED_INT8), width, height, 0, pixels.data(), &clError);
	Logger::logOpenClError(clError, "create OpenCL image from png");
	error_quit_program(clError);

	// create OpenCL image for the preprocessed data
	const unsigned outWidth = width / 4;
	const unsigned outHeight = height / 4;
	auto clPrepImg = createGrayClImage(clCtx, outWidth, outHeight);

	// run preprocess kernel
	{
		auto preprocessKernel = loadKernel(clCtx, "preprocess.cl", "preprocess");
		preprocessKernel.setArg(0, clInImg);
		preprocessKernel.setArg(1, clPrepImg);
		runKernel(queue, preprocessKernel, cl::NDRange(outWidth, outHeight), "preprocess kernel");
	}

	// create OpenCL image for mean data
	auto clMeansImg = createGrayClImage(clCtx, outWidth, outHeight);

	// run mean kernel
	{
		auto meanKernel = loadKernel(clCtx, "mean.cl", "mean");
		meanKernel.setArg(0, clPrepImg);
		meanKernel.setArg(1, clMeansImg);
		meanKernel.setArg(2, WINDOW);
		runKernel(queue, meanKernel, cl::NDRange(outWidth, outHeight), "mean kernel");
	}

	// create OpenCL image for std data
	auto clStdImg = createGrayClImage(clCtx, outWidth, outHeight);

	// run stdDev kernel
	{
		auto stdDevKernel = loadKernel(clCtx, "std_dev.cl", "stdDev");
		stdDevKernel.setArg(0, clPrepImg);
		stdDevKernel.setArg(1, clMeansImg);
		stdDevKernel.setArg(2, clStdImg);
		stdDevKernel.setArg(3, WINDOW);
		runKernel(queue, stdDevKernel, cl::NDRange(outWidth, outHeight), "std dev kernel");
	}

	std::vector<float> processedImage(outWidth * outHeight);
	cl::size_t<3> size;
	size[0] = outWidth;
	size[1] = outHeight;
	size[2] = 1;
	clError = queue.enqueueReadImage(clStdImg, CL_TRUE, cl::size_t<3>(), size, 0, 0, processedImage.data());
	Logger::logOpenClError(clError, "read computed image");
	error_quit_program(clError);
    queue.finish();

	std::vector<uint8_t> outputImage(processedImage.size());
	for (size_t i = 0; i < processedImage.size(); i++) {
		outputImage[i] = static_cast<uint8_t>(processedImage[i]);
	}

	error = lodepng::encode("out.png", outputImage, outWidth, outHeight, LCT_GREY, 8);
	Logger::logSave(error, "out.png");
	getchar();
    return 0;
}