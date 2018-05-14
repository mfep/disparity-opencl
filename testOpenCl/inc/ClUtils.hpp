#ifndef CLUTILS_HPP
#define CLUTILS_HPP

#include <cstdint>
#include "CL/cl.hpp"


/// A collection of helper functions implementing the disparity algorithm with OpenCL.
namespace ClUtils {

/// Contains the result of the function `precalcImage`. It contains the precalculated downscaled 
/// grayscale image of the original, the image with the window standard deviations and the image
/// with the window means.
struct PrecalcImage {
	const unsigned width, height;
	cl::Image2D grayImg;
	cl::Image2D means;
	cl::Image2D stdDev;
};

/// If the error is not zero, waits for user input then quits the program.
/// \param error The error code to check.
template<typename T>
void		error_quit_program(T error) {
	if (error != 0) {
		getchar();
		exit(1);
	}
}

/// Selects the platform and device to run the OpenCL kernels on.
/// \return The cl::Context containing the device settings.
cl::Context	initCl();

/// Reads the text file on the given path to an std::string.
/// \param filename The location of the text file.
/// \return The content of the text file.
std::string	readFile(const char* filename);

/// Loads and builds an OpenCL kernel read from a text file.
/// \param clCtx The OpenCL context to use.
/// \param filename The path of the .cl file.
/// \param kernelname The name of the kernel function.
/// \return The loaded and built OpenCL kernel.
cl::Kernel	loadKernel(const cl::Context& clCtx, const char* filename, const char* kernelname);

/// Creates an OpenCL image buffer object on the device.
/// \param clCtx The OpenCL context to use.
/// \param width The width of the image in pixels.
/// \param height The height of the image in pixels.
/// \param channelType The data type of the image. Defaults to float.
/// \return The OpenCL image handle object.
cl::Image2D	createGrayClImage(const cl::Context& clCtx, unsigned width, unsigned height, cl_channel_type channelType = CL_FLOAT);

/// Adds the given kernel to the given command queue. The kernel arguments need to be preset. Logs the execution time as well.
/// \param queue The OpenCL command queue to use.
/// \param kernel The OpenCL kernel to use.
/// \param globalRange The global NDRange to use for the kernel.
/// \param progressname The string used in logging messages.
void		runKernel(const cl::CommandQueue& queue, const cl::Kernel& kernel, const cl::NDRange& globalRange, const char* progressname, const cl::NDRange& localRange = cl::NullRange);

/// Decodes a png image on the disk and loads it to the memory.
/// \param filename The path of the image file to load.
/// \param width Outputs the width of the loaded image.
/// \param height Outputs the height of the loaded image.
/// \return The image data vector.
std::vector<uint8_t>	loadImage(const char* filename, unsigned& width, unsigned& height);

/// From an input RGB pixel data, creates a downscaled grayscale, a mean filtered and a standard deviation OpenCL image.
/// \param clCtx The OpenCL context to use.
/// \param queue The OpenCL command queue to use.
/// \param pixels The RGB pixel data to process.
/// \param width The width of the input image.
/// \param height The height of the input image.
PrecalcImage	precalcImage(const cl::Context& clCtx, const cl::CommandQueue& queue, std::vector<uint8_t>& pixels, unsigned width, unsigned height);

/// Runs the disparity map calculation kernel on pair of `ClUtils::PrecalcImage`-s.
/// \param clCtx The OpenCL context to use.
/// \param queue The OpenCL command queue to use.
/// \param left The left image and preprocessing data.
/// \param right The right image and preprocessing data.
/// \param invertD When the left and right image are mixed up for post-processing purposes, this has to be set `true`.
/// \return The result disparity map.
cl::Image2D		calculateDisparityMap(const cl::Context& clCtx, const cl::CommandQueue& queue, const PrecalcImage& left, const PrecalcImage& right, bool invertD);

}	// namespace ClUtils

#endif