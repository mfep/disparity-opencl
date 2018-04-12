#ifndef DISPARITY_CPU_PIXELUTILS_HPP
#define DISPARITY_CPU_PIXELUTILS_HPP


#include "Pixels.hpp"

/// Contains basic PNG image loading and saving functions. Curretly implemented using lodePng.
namespace PixelUtils {

/// Loads a PNG file from disk and resizes it to 1/4 of its original dimensions.
/// \param filename The path of the image file to read.
/// \return 0-255 valued floating point array of the image.
Pixelsf loadGrey    (const char* filename);

/// Saves a Pixel array to disk in PNG format.
/// \param pixels 0-255 valued pixel data to save.
/// \param filename Output file path.
void    save        (const Pixelsi &pixels, const char* filename);

};  // namespace PixelUtils


#endif //DISPARITY_CPU_PIXELUTILS_HPP
