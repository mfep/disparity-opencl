const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

// convert to float grayscale
__kernel void preprocess(__read_only image2d_t input, __write_only image2d_t output) {
	int2 coord = (int2)(get_global_id(0), get_global_id(1));
	const float4 rgb2gray = { 0.2126f, 0.7152f, 0.0722f, 0.f };
	float4 sample = convert_float(read_imageui(input, sampler, coord));
	write_imagef(output, coord, dot(rgb2gray, sample));
}
