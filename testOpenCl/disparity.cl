#include "clIncludes.h"

__constant const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;


inline float sample(__read_only image2d_t in, int col, int row) {
	return read_imagef(in, sampler, (int2)(col, row)).x;
}


__kernel void disparity(
	__write_only image2d_t output, __read_only image2d_t left, __read_only image2d_t right,
	__read_only image2d_t leftMeans, __read_only image2d_t rightMeans,
	__read_only image2d_t leftStd, __read_only image2d_t rightStd, int invertD)
{
	const int cx = get_global_id(0);
	const int cy = get_global_id(1);
	const int gx = get_local_id(0);
	const int gy = get_local_id(1);
	const float meanL = sample(leftMeans, cx, cy);

	// cache left samples
	const int bw = GW + 2 * D;
	const int bh = GH + 2 * D;
	__local float leftBuffer[bw*bh];
	const int xiter = (bw - gx) / GW + 1;
	const int yiter = (bh - gy) / GH + 1;
	for (int j = 0; j < yiter; ++j) {
		for (int i = 0; i < xiter; ++i) {
			const int globalx = cx - D + i*GW;
			const int globaly = cy - D + j*GH;
			const float smpl = sample(left, globalx, globaly);

			const int bufferx = gx + i*GW;
			const int buffery = gy + j*GH;
			const int bufferi = buffery * bw + bufferx;
			leftBuffer[bufferi] = smpl;
		}
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);

	float bestZncc = 0.f;
	int bestDisp = 0;
	for (int disp = 0; disp < MAX_DISP; ++disp) {
		const float d = invertD ? -disp : disp;
		const float meanR = sample(rightMeans, cx - d, cy);
		float sum = 0.f;
		for (int row = cy - D; row <= cy + D; ++row) {
			for (int col = cx - D; col <= cx + D; ++col) {
				const int bufferx = col - cx + D + gx;
				const int buffery = row - cy + D + gy;
				const int bufferi = buffery * bw + bufferx;
				sum += (leftBuffer[bufferi] - meanL) * (sample(right, col - d, row) - meanR);
			}
		}
		const float zncc = sum / sample(leftStd, cx, cy) / sample(rightStd, cx - d, cy);
		if (zncc > bestZncc) {
			bestZncc = zncc;
			bestDisp = disp;
		}
	}
	write_imageui(output, (int2)(cx, cy), convert_uchar((float)bestDisp / MAX_DISP * 255.f));
}