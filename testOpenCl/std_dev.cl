const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void stdDev(__read_only image2d_t input, __read_only image2d_t means, __write_only image2d_t output, int WINDOW) {
	const int2 coord = (int2)(get_global_id(0), get_global_id(1));
	const int D = WINDOW / 2;
	float sum = 0.f;
	for (int row = coord.y - D; row <= coord.y + D; ++row) {
		for (int col = coord.x - D; col <= coord.x + D; ++col) {
			const float value = read_imagef(input, sampler, (int2)(col, row)).x;
			const float mean = read_imagef(means, sampler, (int2)(col, row)).x;
			sum += (value - mean) * (value - mean);
		}
	}
	write_imagef(output, coord, sqrt(sum));
}