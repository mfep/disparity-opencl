#ifndef CLUTILS_HPP
#define CLUTILS_HPP

#include <cstdint>
#include "CL/cl.hpp"


namespace ClUtils {

struct PrecalcImage {
	const unsigned width, height;
	cl::Image2D grayImg;
	cl::Image2D means;
	cl::Image2D stdDev;
};


cl::Context initCl();


std::string readFile(const char* filename);


template<typename T>
void error_quit_program(T error) {
	if (error != 0) {
		getchar();
		exit(1);
	}
}


cl::Kernel loadKernel(const cl::Context& clCtx, const char* filename, const char* kernelname);


cl::Image2D createGrayClImage(const cl::Context& clCtx, unsigned width, unsigned height, cl_channel_type channelType = CL_FLOAT);


void runKernel(const cl::CommandQueue& queue, const cl::Kernel& kernel, const cl::NDRange& globalRange, const char* progressname);


std::vector<uint8_t> loadImage(const char* filename, unsigned& width, unsigned& height);


PrecalcImage precalcImage(const cl::Context& clCtx, const cl::CommandQueue& queue, std::vector<uint8_t>& pixels, unsigned width, unsigned height);


cl::Image2D calculateDisparityMap(const cl::Context& clCtx, const cl::CommandQueue& queue, const PrecalcImage& left, const PrecalcImage& right, bool invertD);

}	// namespace ClUtils

#endif