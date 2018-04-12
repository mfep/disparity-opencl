#include <algorithm>
#include <sstream>
#include <memory>
#include <iostream>
#include "Logger.hpp"
#include "PixelUtils.hpp"
#include "lodepng.h"


namespace {

Pixelsf preprocessPixels(const std::vector<unsigned char> &pixels, unsigned width, unsigned height) {
    const float
            R = 0.2126f,
            G = 0.7152f,
            B = 0.0722f;

    unsigned index = 0;
    std::vector<float> resized(pixels.size() / 16 / 3);
    for (unsigned row = 0; row < height; row += 4) {
        for (unsigned col = 0; col < width; col += 4) {
            const unsigned char
                    r = pixels[(row * width + col) * 3],
                    g = pixels[(row * width + col) * 3 + 1],
                    b = pixels[(row * width + col) * 3 + 2];
            resized[index++] = r * R + g * G + b * B;
        }
    }
    width /= 4;
    height /= 4;
    return Pixelsf(std::move(resized), width, height);
}


template<typename T, typename U>
std::vector<T> convertPixels(const std::vector<U>& in) {
    std::vector<T> result(in.size());
    unsigned index = 0;
    for (auto& v : result) {
        v = static_cast<T>(in[index++]);
    }
    return result;
}

}


Pixelsf PixelUtils::loadGrey(const char *filename) {
    unsigned width, height;
    std::vector<unsigned char> pixels;
    unsigned error = lodepng::decode(pixels, width, height, filename, LCT_RGB);
    Logger::logLoad(error, filename);
    return preprocessPixels(pixels, width, height);
}


void PixelUtils::save(const Pixelsi& pixels, const char* filename) {
    unsigned error = lodepng::encode(filename, convertPixels<unsigned char, int>(pixels.getData()), pixels.getWidth(), pixels.getHeight(), LCT_GREY, 8);
    Logger::logSave(error, filename);
}
