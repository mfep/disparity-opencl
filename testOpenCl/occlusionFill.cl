const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;


inline float sample(__read_only image2d_t in, int col, int row) {
	return read_imageui(in, sampler, (int2)(col, row)).x;
}


__kernel void occlusionFill(__write_only image2d_t output, __read_only image2d_t input, int maxOffset) {
	const int cx = get_global_id(0);
	const int cy = get_global_id(1);
	const int value = sample(input, cx, cy);
	if (value != 0) {
		write_imageui(output, (int2)(cx, cy), value);
		return;
	}
	for (int offset = 1; offset <= maxOffset; ++offset) {
		for (int row = cy - offset; row <= cy + offset; ++row) {
			for (int col = cx - offset; col <= cx + offset; ++col) {
				const int val = sample(input, col, row);
				if (val != 0) {
					write_imageui(output, (int2)(cx, cy), val);
					return;
				}
			}
		}
	}
}