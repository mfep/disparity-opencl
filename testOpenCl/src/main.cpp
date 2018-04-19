#include <iostream>
#include "ClUtils.hpp"
#include "lodepng.h"
#include "Logger.hpp"


int main() {
	using namespace ClUtils;

	// initialize OpenCL
	int clError = 0;
	auto clCtx = initCl();
	cl::CommandQueue queue(clCtx, CL_QUEUE_PROFILING_ENABLE);

	// load images
	unsigned widthL, heightL, widthR, heightR;
	auto pixelsL = loadImage("im0.png", widthL, heightL);
	auto pixelsR = loadImage("im1.png", widthR, heightR);
	if (widthL != widthR || heightL != heightR) {
		std::cout << "input image dimensions should match" << std::endl;
		error_quit_program(1);
	}

	// calculate image mean&stddev
	auto imDataL = precalcImage(clCtx, queue, pixelsL, widthL, heightL);
	auto imDataR = precalcImage(clCtx, queue, pixelsR, widthL, heightL);

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
		runKernel(queue, crossCheckKernel, cl::NDRange(imDataL.width, imDataL.height), "cross check kernel");
	}

	// postprocess (occlusion fill)
	auto outImg = createGrayClImage(clCtx, imDataL.width, imDataL.height, CL_UNSIGNED_INT8);
	{
		auto occlusionKernel = loadKernel(clCtx, "occlusionFill.cl", "occlusionFill");
		occlusionKernel.setArg(0, outImg);
		occlusionKernel.setArg(1, crossCheckImg);
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