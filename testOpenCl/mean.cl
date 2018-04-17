#include "clIncludes.h"

const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void mean(__read_only image2d_t input, __write_only image2d_t output) {
	const int2 coord = (int2)(get_global_id(0), get_global_id(1));
	float sum = 0.f;
	for (int row = coord.y - D; row <= coord.y + D; ++row) {
		for (int col = coord.x - D; col <= coord.x + D; ++col) {
			sum += read_imagef(input, sampler, (int2)(col, row)).x;
		}
	}
	write_imagef(output, coord, sum / (float)(WINDOW * WINDOW));
}