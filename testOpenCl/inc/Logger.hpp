#ifndef DISPARITY_CPU_LOGGER_HPP
#define DISPARITY_CPU_LOGGER_HPP


#include <chrono>


/// Provides global logging and time-measurement functionality. Outputs to `stdout`.
class Logger {
public:
    /// Logs global program settings : window size, worker thread count
    static void logInit         ();

    /// Logs about file loading.
    /// \param code The error code returned by image loading method.
    /// \param filename The filename used in the message.
    static void logLoad         (unsigned code, const char *filename);

    /// Logs about file saving.
    /// \param code The error code returned by image saving method.
    /// \param filename
    static void logSave         (unsigned code, const char *filename);

    /// Logs a message about the process started and starts the stopwatch.
    /// \param text Process description.
    static void startProgress   (const char* text);

    /// Logs a percent indicator about the current process. Call after `startProgress`.
    /// \param percent The progress in 0-1 interval.
    static void logProgress     (float percent);

    /// Logs a message about a process end, also an execution time since `startProgress`.
    static void endProgress     ();

private:
    static int m_lastBars;
    static const char* m_progressText;
    static std::chrono::system_clock::time_point m_startTime;
};


#endif //DISPARITY_CPU_LOGGER_HPP
