const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void testKernel(__read_only image2d_t input, __write_only image2d_t output) { 
	int2 coord = (int2)(get_global_id(0), get_global_id(1));
	uint4 sample = read_imageui(input, sampler, coord);
	write_imagef(output, coord, convert_float(sample.x));
}