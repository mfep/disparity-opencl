#include "ClUtils.hpp"

#include <string>
#include <fstream>
#include <iostream>
#include <streambuf>
#include "Logger.hpp"
#include "lodepng.h"


cl::Context ClUtils::initCl() {
	//get all platforms (drivers)
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    if(all_platforms.size()==0){
        std::cout<<" No platforms found. Check OpenCL installation!" << std::endl;
        exit(1);
    }
    cl::Platform default_platform=all_platforms[1];
    std::cout << "Using platform: "<<default_platform.getInfo<CL_PLATFORM_NAME>()<<std::endl;

    //get default device of the default platform
    std::vector<cl::Device> all_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if(all_devices.size()==0){
        std::cout<<" No devices found. Check OpenCL installation!" << std::endl;
        exit(1);
    }
    cl::Device default_device=all_devices[0];
    std::cout<< "Using device: "<<default_device.getInfo<CL_DEVICE_NAME>()<<std::endl;
	return cl::Context({default_device});
}


std::string ClUtils::readFile(const char* filename) {
	std::ifstream t(filename);
	std::string str;

	t.seekg(0, std::ios::end);
	str.reserve(t.tellg());
	t.seekg(0, std::ios::beg);

	str.assign((std::istreambuf_iterator<char>(t)),
				std::istreambuf_iterator<char>());
	return str;
}


cl::Kernel ClUtils::loadKernel(const cl::Context& clCtx, const char* filename, const char* kernelname) {
	auto programText = readFile(filename);
	cl::Program prepProgram(clCtx, programText);
	int clError = prepProgram.build();
	Logger::logOpenClError(clError, "build cl program");
	error_quit_program(clError);

	cl::Kernel kernel(prepProgram, kernelname, &clError);
	Logger::logOpenClError(clError, filename);
	error_quit_program(clError);
	return kernel;
}


cl::Image2D ClUtils::createGrayClImage(const cl::Context& clCtx, unsigned width, unsigned height, cl_channel_type channelType) {
	int clError = 0;
	cl::Image2D clImg(clCtx, CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, channelType), width, height, 0, nullptr, &clError);
	Logger::logOpenClError(clError, "create OpenCL image");
	error_quit_program(clError);
	return clImg;
}


void ClUtils::runKernel(const cl::CommandQueue& queue, const cl::Kernel& kernel, const cl::NDRange& globalRange, const char* progressname) {
		cl::Event ev;
		int clError = queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalRange, cl::NullRange, nullptr, &ev);
		Logger::logOpenClError(clError, "add kernel to command queue");
		error_quit_program(clError);
		queue.finish();
		const cl_ulong duration = ev.getProfilingInfo<CL_PROFILING_COMMAND_END>() - ev.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		std::cout << "OpenCL process: " << progressname << " finished in: " << duration / 1e6f << "ms" << std::endl;
}


std::vector<uint8_t> ClUtils::loadImage(const char* filename, unsigned& width, unsigned& height) {
	std::vector<uint8_t> pixels;
	unsigned error = lodepng::decode(pixels, width, height, filename, LCT_RGBA);
	Logger::logLoad(error, filename);
	error_quit_program(error);
	return pixels;
}


ClUtils::PrecalcImage ClUtils::precalcImage(const cl::Context& clCtx, const cl::CommandQueue& queue, std::vector<uint8_t>& pixels, unsigned width, unsigned height) {
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
		runKernel(queue, stdDevKernel, cl::NDRange(outWidth, outHeight), "std dev kernel");
	}

	// assemble output
	return {outWidth, outHeight, clPrepImg, clMeansImg, clStdImg};
}


cl::Image2D ClUtils::calculateDisparityMap(const cl::Context& clCtx, const cl::CommandQueue& queue, const PrecalcImage& left, const PrecalcImage& right, bool invertD) {
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
		runKernel(queue, dispKernel, cl::NDRange(left.width, left.height), "disparity kernel");
	}
	return outImg;
}
