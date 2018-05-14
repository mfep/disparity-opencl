#include "clIncludes.h"

__constant const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;


inline float sample(__read_only image2d_t in, int col, int row) {
	return read_imagef(in, sampler, (int2)(col, row)).x;
}

__kernel void localTest(__read_only image2d_t input, __write_only image2d_t output) {
	const int cx = get_global_id(0);
	const int cy = get_global_id(1);
	const int gx = get_local_id(0);
	const int gy = get_local_id(1);

	__local float pixelBuffer[GW * GH];
	pixelBuffer[gy * GW + gx] = sample(input, cx, cy);

	barrier(CLK_LOCAL_MEM_FENCE);

	const int index = GW * GH - (gy * GW + gx) - 1;
	write_imageui(output, (int2)(cx, cy), convert_uint(pixelBuffer[index]));
}
