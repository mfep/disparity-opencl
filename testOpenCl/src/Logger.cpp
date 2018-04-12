#include "Logger.hpp"
#include <iostream>
#include "lodepng.h"
#include "CliOptions.hpp"


int Logger::m_lastBars = 0;
const char* Logger::m_progressText = nullptr;
std::chrono::system_clock::time_point Logger::m_startTime;


void Logger::logInit() {
    std::cout <<
        "Disparity algorithm CPU implementation started." << std::endl <<
        "Number of worker threads = " << CliOptions::getThreads() << std::endl <<
        "Window size = " << CliOptions::getWindow() << std::endl;
}


void Logger::logLoad(unsigned code, const char *filename) {
    if (code) {
        std::cout << "decoder error " << code << ": " << lodepng_error_text(code) << std::endl;
    } else {
        std::cout << "loading '" << filename << "' was successful" << std::endl;
    }
}


void Logger::logSave(unsigned code, const char *filename) {
    if (code) {
        std::cout << "encoder error " << code << ": " << lodepng_error_text(code) << std::endl;
    } else {
        std::cout << "successfully saved file: " << filename << std::endl;
    }
}


void Logger::startProgress(const char* text) {
    m_progressText = text;
    std::cout << "=== starting " << text << std::endl;
    m_startTime = std::chrono::system_clock::now();
}


// TODO multithreaded logging
void Logger::logProgress(float percent) {
    const int BARS = 40;

    const auto currentBars = static_cast<int>(percent * BARS);
    if (currentBars == m_lastBars) {
        return;
    }
    m_lastBars = currentBars;
    std::cout << m_progressText << "\t[";
    for (int i = 0; i < BARS; ++i) {
        if (i < currentBars) {
            std::cout << "=";
        } else if (i == currentBars) {
            std::cout << ">";
        } else {
            std::cout << " ";
        }
    }
    std::cout << "]\r";
    std::cout.flush();
}


void Logger::endProgress() {
    std::chrono::duration<float> elapsed = std::chrono::system_clock::now() - m_startTime;
    m_lastBars = 0;
    std::cout << "ended " << m_progressText << " in " << elapsed.count() << "s" << std::endl;
}
