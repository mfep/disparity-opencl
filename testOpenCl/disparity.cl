const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;


inline float sample(__read_only image2d_t in, int col, int row) {
	return read_imagef(in, sampler, (int2)(col, row)).x;
}


inline float mean(__read_only image2d_t in, int cx, int cy, int window) {
	const int D = window / 2;
	float sum = 0.f;
	for (int row = cy - D; row <= cy + D; ++row) {
		for (int col = cx - D; col <= cx + D; ++col) {
			sum += sample(in, col, row);
		}
	}
	return sum / convert_float(window) / convert_float(window);
}


inline float stdDev(__read_only image2d_t in, int cx, int cy, int window) {
	const int D = window / 2;
	const float m = mean(in, cx, cy, window);
	float sum = 0.f;
	for (int row = cy - D; row <= cy + D; ++row) {
		for (int col = cx - D; col <= cx + D; ++col) {
			const float value = sample(in, col, row);
			sum += (value - m) * (value - m);
		}
	}
	return sqrt(sum);
}


__kernel void disparity(
	__write_only image2d_t output, __read_only image2d_t left, __read_only image2d_t right,
	__read_only image2d_t leftMeans, __read_only image2d_t rightMeans,
	__read_only image2d_t leftStd, __read_only image2d_t rightStd, int invertD, int window, int maxDisp)
{
	const int cx = get_global_id(0);
	const int cy = get_global_id(1);
	const int D = window / 2;
	const float meanL = sample(leftMeans, cx, cy);

	float bestZncc = 0.f;
	int bestDisp = 0;
	for (int disp = 0; disp < maxDisp; ++disp) {
		const float d = invertD ? -disp : disp;
		const float meanR = sample(rightMeans, cx - d, cy);
		float sum = 0.f;
		for (int row = cy - D; row <= cy + D; ++row) {
			for (int col = cx - D; col <= cx + D; ++col) {
				sum += (sample(left, col, row) - meanL) * (sample(right, col - d, row) - meanR);
			}
		}
		const float zncc = sum / sample(leftStd, cx, cy) / stdDev(right, cx - d, cy, window);
		if (zncc > bestZncc) {
			bestZncc = zncc;
			bestDisp = disp;
		}
	}
	write_imageui(output, (int2)(cx, cy), convert_uint((float)bestDisp / maxDisp * 255.f));
}