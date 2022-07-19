#ifndef TIMER_H
#define TIMER_H

#include <chrono>

class timer
{
public:
    timer() : start_(), end_()
    {
    }

    void start()
    {
        start_ = std::chrono::system_clock::now();
    }

    void stop()
    {
        end_ = std::chrono::system_clock::now();
    }

    double get_elapsed_seconds() const
    {
        return (double)std::chrono::duration_cast<std::chrono::seconds>(end_ - start_).count();
    }

    double get_elapsed_milli_seconds() const
    {
        return (double)std::chrono::duration_cast<std::chrono::milliseconds>(end_ - start_).count();
    }

    double get_elapsed_micro_seconds() const
    {
        return (double)std::chrono::duration_cast<std::chrono::microseconds>(end_ - start_).count();
    }

    double get_elapsed_nano_seconds() const
    {
        return (double)std::chrono::duration_cast<std::chrono::nanoseconds>(end_ - start_).count();
    }

private:
    std::chrono::time_point<std::chrono::system_clock> start_;
    std::chrono::time_point<std::chrono::system_clock> end_;
};

#endif