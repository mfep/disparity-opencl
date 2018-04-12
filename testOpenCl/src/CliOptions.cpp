#include "CliOptions.hpp"
#include "cxxopts.hpp"


namespace {

const char* PROG_NAME = "Disparity Calculator";
const char* PROG_DESC = "";

}


int CliOptions::threads = 0;
int CliOptions::window = 0;


void CliOptions::parse(int argc, const char* argv[]) {
    cxxopts::Options options(PROG_NAME, PROG_DESC);
    options.add_options()
            ("t,threads", "Set number of threads to use", cxxopts::value<int>()->default_value("1"))
    // TODO window from cli
            ("w,window", "Set window size", cxxopts::value<int>()->default_value("9"));
    auto result = options.parse(argc, argv);

    threads = result["threads"].as<int>();
    window = result["window"].as<int>();

    if (threads <= 0 || window <= 0) {
        throw std::exception();
    }
}


int CliOptions::getThreads() {
    return threads;
}


int CliOptions::getWindow() {
    return window;
}
