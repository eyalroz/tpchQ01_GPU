/*
 * This file contains definitions for some general-purpose
 * code, which is inspecific to the TPC-H Q1 experiments,
 * and didn't fit elsewhere
 */
#ifndef UTIL_HELPER_H_
#define UTIL_HELPER_H_
#pragma once

#include "extra_pointer_traits.hpp"

#include <cuda_runtime.h>
#include <cuda/api_wrappers.h>
#include <iostream>
#include <string>
#include <cassert>
#include <unistd.h>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <chrono>


template <typename T>
using plugged_unique_ptr = std::unique_ptr<T>;

template <typename T>
using plain_ptr = std::conditional_t<std::is_array<T>::value, std::decay_t<T>, std::decay_t<T>*>;

inline void assert_always(bool a) {
    assert(a);
    if (!a) {
        fprintf(stderr, "Assert always failed!");
        exit(EXIT_FAILURE);
    }
}


template <typename T>
struct extra_pointer_traits<cuda::memory::host::unique_ptr<T[]>>
{
    using ptr_type = cuda::memory::host::unique_ptr<T[]>;
    using element_type = T;

    static T* release(ptr_type& ptr) { return ptr.release(); }
    static ptr_type make(size_t count)
    {
        return cuda::memory::host::make_unique<T[]>(count);
    }
};

template <typename T>
struct extra_pointer_traits<cuda::memory::host::unique_ptr<T>>
{
    using ptr_type = cuda::memory::host::unique_ptr<T>;
    using element_type = T;

    static T* release(ptr_type& ptr) { return ptr.release(); }
    static ptr_type make()
    {
        return cuda::memory::host::make_unique<T>();
    }
};

void list_device_properties();

template <typename F, typename... Args>
void for_each_argument(F f, Args&&... args) {
    [](...){}((f(std::forward<Args>(args)), 0)...);
}

// Note: This will force casts to int. It's not a problem
// the way our code is written, but otherwise it needs to be generalized
constexpr inline int div_rounding_up(const int& dividend, const int& divisor)
{
    // This is not the fastest implementation, but it's safe, in that there's never overflow
#if __cplusplus >= 201402L
    std::div_t div_result = std::div(dividend, divisor);
    return div_result.quot + !(!div_result.rem);
#else
    // Hopefully the compiler will optimize the two calls away.
    return std::div(dividend, divisor).quot + !(!std::div(dividend, divisor).rem);
#endif
}


void make_sure_we_are_on_cpu_core_0();
std::string host_name();
std::string timestamp();
std::pair<std::string,std::string> split_once(std::string delimited, char delimiter);


#endif // UTIL_HELPER_H_
