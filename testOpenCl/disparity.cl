const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;


inline float sample(__read_only image2d_t in, int col, int row) {
	return read_imagef(in, sampler, (int2)(col, row)).x;
}


inline float mean(__read_only image2d_t in, int cx, int cy, int WINDOW) {
	const int D = WINDOW / 2;
	float sum = 0.f;
	for (int row = cy - D; row <= cy + D; ++row) {
		for (int col = cx - D; col <= cx + D; ++col) {
			sum += sample(in, col, row);
		}
	}
	return sum / convert_float(WINDOW) / convert_float(WINDOW);
}


inline float stdDev(__read_only image2d_t in, int cx, int cy, int WINDOW) {
	const int D = WINDOW / 2;
	const float m = mean(in, cx, cy, WINDOW);
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
	__read_only image2d_t leftStd, __read_only image2d_t rightStd, int WINDOW, int MAX_DISP)
{
	const int cx = get_global_id(0);
	const int cy = get_global_id(1);
	const int D = WINDOW / 2;
	const float meanL = sample(leftMeans, cx, cy);

	float bestZncc = 0.f;
	int bestDisp = 0;
	for (int disp = 0; disp < MAX_DISP; ++disp) {
		const float meanR = sample(rightMeans, cx - disp, cy);
		float sum = 0.f;
		for (int row = cy - D; row <= cy + D; ++row) {
			for (int col = cx - D; col <= cx + D; ++col) {
				sum += (sample(left, col, row) - meanL) * (sample(right, col - disp, row) - meanR);
			}
		}
		const float zncc = sum / sample(leftStd, cx, cy) / stdDev(right, cx - disp, cy, WINDOW);
		if (zncc > bestZncc) {
			bestZncc = zncc;
			bestDisp = disp;
		}
	}
	write_imagef(output, (int2)(cx, cy), convert_float(bestDisp / convert_float(MAX_DISP) * 255.f));
}