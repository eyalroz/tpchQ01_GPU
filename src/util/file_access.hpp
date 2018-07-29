#pragma once
#ifndef FILE_ACCESS_HPP_
#define FILE_ACCESS_HPP_

#if __cplusplus >= 201703L
#include <filesystem>
using namespace filesystem = std::filesystem;
#elif __cplusplus >= 201402L
#include <experimental/filesystem>
namespace filesystem = std::experimental::filesystem;
#else
#error This code must be compiled using the C++14 language started or later
#endif

#include <iostream>
#include <fstream>
#include <ios>

// TODO: Use spans
template <typename T>
void load_column_from_binary_file(
    T* __restrict__          buffer,
    std::size_t              count,
    const filesystem::path&  directory,
    const std::string&       base_filename)
{
    auto file_path = directory / base_filename;
    std::cout << "Loading a column from " << file_path << " ... " << std::flush;
    FILE* pFile = fopen(file_path.c_str(), "rb");
    if (pFile == nullptr) {
        throw std::runtime_error("Failed opening file " + file_path.string());
    }
    auto num_elements_read = fread(buffer, sizeof(T), count, pFile);
    if (num_elements_read != count) {
        throw std::runtime_error("Failed reading sufficient data from " +
            file_path.string() + " : expected " + std::to_string(count) + " elements but read only " + std::to_string(num_elements_read) + "."); }
    fclose(pFile);
    std::cout << "done." << std::endl;
}

template <typename T>
void write_column_to_binary_file(
    const T* __restrict__   buffer,
    std::size_t             count,
    const filesystem::path& directory,
    const std::string&      base_filename)
{
    auto file_path = directory / base_filename;
    std::cout << "Writing a column to " << file_path << " ... " << std::flush;
    FILE* pFile = fopen(file_path.c_str(), "wb+");
    if (pFile == nullptr) { throw std::runtime_error("Failed opening file " + file_path.string()); }
    auto num_elements_written = fwrite(buffer, sizeof(T), count, pFile);
    fclose(pFile);
    if (num_elements_written != count) {
        remove(file_path.c_str());
        throw std::runtime_error("Failed writing all elements to the file - only " +
            std::to_string(num_elements_written) + " written: " + strerror(errno));
    }
    std::cout << "done." << std::endl;
}

#endif // FILE_ACCESS_HPP_
