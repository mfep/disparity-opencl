#ifndef DISPARITY_CPU_CLIOPTIONS_HPP
#define DISPARITY_CPU_CLIOPTIONS_HPP


#include <memory>


// TODO documentation
class CliOptions {
public:
    static void parse       (int argc, const char* argv[]);
    static int  getThreads  ();
    static int  getWindow   ();

private:
    static int threads;
    static int window;
};


#endif //DISPARITY_CPU_CLIOPTIONS_HPP
