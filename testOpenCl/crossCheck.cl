const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void crossCheck(__write_only image2d_t output, __read_only image2d_t leftDisp, __read_only image2d_t rightDisp, int threshold) {
	const int2 coord = (int2)(get_global_id(0), get_global_id(1));
	const int leftVal = read_imageui(leftDisp, sampler, coord).x;
	const int rightVal = read_imageui(rightDisp, sampler, coord).x;
	if (abs_diff(leftVal, rightVal) > threshold) {
		write_imageui(output, coord, 0);
	} else {
		write_imageui(output, coord, (leftVal + rightVal) / 2);
	}
}