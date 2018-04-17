#include <iostream>
#include <cstdint>
#include <string>
#include <fstream>
#include <streambuf>
#include "CL/cl.hpp"
#include "lodepng.h"
#include "Logger.hpp"


constexpr int WINDOW = 9;
constexpr int MAX_DISP = 260 / 4;
constexpr int CROSS_TH = 8;
constexpr int MAX_OFFSET = 50;


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


cl::Image2D createGrayClImage(const cl::Context& clCtx, unsigned width, unsigned height, cl_channel_type channelType = CL_FLOAT) {
	int clError = 0;
	cl::Image2D clImg(clCtx, CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, channelType), width, height, 0, nullptr, &clError);
	Logger::logOpenClError(clError, "create OpenCL image");
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


struct PrecalcImage {
	const unsigned width, height;
	cl::Image2D grayImg;
	cl::Image2D means;
	cl::Image2D stdDev;
};

PrecalcImage loadAndPrecalcImage(const cl::Context& clCtx, const cl::CommandQueue& queue, const char* filename) {
	int clError = 0;

	// load input image
	std::vector<uint8_t> pixels;
	unsigned width, height;
	unsigned error = lodepng::decode(pixels, width, height, filename, LCT_RGBA);
	Logger::logLoad(error, filename);
	error_quit_program(error);

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

	// assemble output
	return {outWidth, outHeight, clPrepImg, clMeansImg, clStdImg};
}


cl::Image2D calculateDisparityMap(const cl::Context& clCtx, const cl::CommandQueue& queue, const PrecalcImage& left, const PrecalcImage& right, bool invertD) {
	auto outImg = createGrayClImage(clCtx, left.width, left.height, CL_UNSIGNED_INT8);
	{
		auto dispKernel = loadKernel(clCtx, "disparity.cl", "disparity");
		dispKernel.setArg(0, outImg);
		dispKernel.setArg(1, left.grayImg);
		dispKernel.setArg(2, right.grayImg);
		dispKernel.setArg(3, left.means);
		dispKernel.setArg(4, right.means);
		dispKernel.setArg(5, left.stdDev);
		dispKernel.setArg(6, right.stdDev);
		dispKernel.setArg(7, invertD ? 1 : 0);
		dispKernel.setArg(8, WINDOW);
		dispKernel.setArg(9, MAX_DISP);
		runKernel(queue, dispKernel, cl::NDRange(left.width, left.height), "disparity kernel");
	}
	return outImg;
}

}	// namespace


int main() {
	// initialize OpenCL
	int clError = 0;
	auto clCtx = initCl();
	cl::CommandQueue queue(clCtx);

	// load and calculate image mean&stddev
	auto imDataL = loadAndPrecalcImage(clCtx, queue, "im0.png");
	auto imDataR = loadAndPrecalcImage(clCtx, queue, "im1.png");

	// calculate disparity maps + normalize
	auto dispL = calculateDisparityMap(clCtx, queue, imDataL, imDataR, false);
	auto dispR = calculateDisparityMap(clCtx, queue, imDataR, imDataL, true);

	// cross-check
	auto crossCheckImg = createGrayClImage(clCtx, imDataL.width, imDataL.height, CL_UNSIGNED_INT8);
	{
		auto crossCheckKernel = loadKernel(clCtx, "crossCheck.cl", "crossCheck");
		crossCheckKernel.setArg(0, crossCheckImg);
		crossCheckKernel.setArg(1, dispL);
		crossCheckKernel.setArg(2, dispR);
		crossCheckKernel.setArg(3, CROSS_TH);
		runKernel(queue, crossCheckKernel, cl::NDRange(imDataL.width, imDataL.height), "cross check kernel");
	}

	// postprocess (occlusion fill)
	auto outImg = createGrayClImage(clCtx, imDataL.width, imDataL.height, CL_UNSIGNED_INT8);
	{
		auto occlusionKernel = loadKernel(clCtx, "occlusionFill.cl", "occlusionFill");
		occlusionKernel.setArg(0, outImg);
		occlusionKernel.setArg(1, crossCheckImg);
		occlusionKernel.setArg(2, MAX_OFFSET);
		runKernel(queue, occlusionKernel, cl::NDRange(imDataL.width, imDataL.height), "occlusionFill kernel");
	}

	// save output image
	std::vector<uint8_t> processedImage(imDataL.width * imDataL.height);
	cl::size_t<3> size;
	size[0] = imDataL.width;
	size[1] = imDataL.height;
	size[2] = 1;
	clError = queue.enqueueReadImage(outImg, CL_TRUE, cl::size_t<3>(), size, 0, 0, processedImage.data());
	Logger::logOpenClError(clError, "read computed image");
	error_quit_program(clError);
    queue.finish();

	unsigned error = lodepng::encode("out.png", processedImage, imDataL.width, imDataL.height, LCT_GREY, 8);
	Logger::logSave(error, "out.png");
	getchar();
    return 0;
}