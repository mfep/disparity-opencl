#ifndef DISPARITY_CPU_PIXELS_HPP
#define DISPARITY_CPU_PIXELS_HPP


#include <vector>
#include <functional>
#include <thread>
#include <future>
#include <algorithm>
#include <iostream>
#include "CliOptions.hpp"


/// Provides helper functionality to linearly stored pixel arrays.
/// \tparam T Type of the data format. Should be an arithmetic type.
template<typename T>
class Pixels {
public:
    /// Constructs a `Pixels` object from given data.
    /// \param data Source data array.
    /// \param width Width of the pixel image.
    /// \param height Height of the pixel image.
    Pixels(std::vector<T>&& data, unsigned width, unsigned height) noexcept :
            m_data      (data),
            m_width     (width),
            m_height    (height)
    {
        static_assert(std::is_arithmetic<T>::value, "arithmetic type required");
    }

    /// Gets the width of the pixel image.
    /// \return The width of the pixel image.
    unsigned getWidth() const noexcept { return m_width; }

    /// Gets the height of the pixel image.
    /// \return The height of the pixel image.
    unsigned getHeight() const noexcept { return m_height; }

    /// Gets the underlying data container.
    /// \return The data array containing the pixel information.
    const std::vector<T>& getData() const noexcept { return m_data; }

    /// Maps the pixel data to a 2d matrix, and returns the value at a particular position.
    /// Overflow is treated by returning the edge values.
    /// \param row The row in the matrix to get.
    /// \param col The column in the matrix to get.
    /// \return The data value specified by `row` and `col`.
    T get(int row, int col) const noexcept {
        if (row < 0) {
            row = 0;
        } else if ((unsigned)row >= m_height) {
            row = m_height - 1;
        }
        if (col < 0) {
            col = 0;
        } else if ((unsigned)col >= m_width) {
            col = m_width - 1;
        }
        return m_data[m_width * row + col];
    }

    /// Enumerates the data values in a window around a specified row and column.
    /// The argument `fun` is called with every data value in the window.
    /// \tparam Tfun The type of the enumerator function.
    /// \param cx The center column of the window.
    /// \param cy The center row of the column.
    /// \param window Window size.
    /// \param fun Enumerator function.
    template<typename Tfun>
    void enumerateWindow(int cx, int cy, int window, const Tfun& fun) const {
        const int d = window / 2;
        for (int row = cy - d; row <= cy + d; ++row) {
            for (int col = cx - d; col <= cx + d; ++col) {
                fun(get(row,col));
            }
        }
    }

    // TODO docs
    std::vector<T> getWindowData(int cx, int cy, int window, int arraySize = 0) const {
        const auto size = static_cast<unsigned>(std::max(arraySize, window * window));
        std::vector<T> windowData(size);
        const int d = window / 2;
        int index = 0;
        for (int row = cy - d; row <= cy + d; ++row) {
            for (int col = cx - d; col <= cx + d; ++col) {
                windowData[index++] = get(row, col);
            }
        }
        return windowData;
    }

    /// Enumerates two `Pixels` instances against each other in a multithreaded way.
    /// Throws an `std::exception` if the two input `Pixels` is different in size.
    /// \tparam U The type of the output `Pixels` object.
    /// \param leftPixels Input `Pixels` object.
    /// \param rightPixels Input `Pixels` object.
    /// \param fun The enumerator function. Should return the `U` typed value at the location given to it.
    /// \return The zipped `Pixels` object.
    template<typename U>
    static
    Pixels<U> pixelZip(const Pixels<T> &leftPixels, const Pixels<T> &rightPixels, const std::function<U(int, int)> &fun) {
        if (leftPixels.getHeight() != rightPixels.getHeight() || leftPixels.getWidth() != rightPixels.getWidth()) {
            throw std::exception();
        }
        const unsigned threads = static_cast<unsigned>(CliOptions::getThreads());
        const int width = leftPixels.getWidth();
        const int height = leftPixels.getHeight();
        const int rows = height / threads;

        std::vector<std::future<std::vector<U>>> futures;
        for (int i = 0; i < threads; ++i) {
            futures.push_back(std::async(std::launch::async, [width, height, threads, rows, i, &fun](){
                std::vector<U> subRes(width*height/threads);
                unsigned index = 0;
                for (int row = i*rows; row < (i+1)*rows; ++row) {
                    for (int col = 0; col < width; ++col) {
                        subRes[index++] = fun(row, col);
                    }
                }
                return subRes;
            }));
        }

        std::vector<U> result;
        for (auto& future : futures) {
            auto subRes = future.get();
            result.insert(result.end(), subRes.begin(), subRes.end());
        }

        return Pixels<U>(std::move(result), leftPixels.getWidth(), leftPixels.getHeight());
    }

private:
    const std::vector<T> m_data;
    const unsigned m_width;
    const unsigned m_height;
};

using Pixelsi = Pixels<int>;
using Pixelsf = Pixels<float>;


#ifdef DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
TEST_CASE("testing Pixels accessor") {
    Pixels pw ({1, 2, 3, 4, 5, 6, 7, 8, 9}, 3, 3);
    CHECK_EQ(pw.get(0, 0), 1);
    CHECK_EQ(pw.get(0, -1), 1);
    CHECK_EQ(pw.get(-1, 0), 1);
    CHECK_EQ(pw.get(1, -1), 4);
    CHECK_EQ(pw.get(1, 3), 6);
    CHECK_EQ(pw.get(3, 3), 9);
}
#endif


#endif //DISPARITY_CPU_PIXELS_HPP
