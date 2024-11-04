#pragma once
#include <fstream>
#include <string>
#include <iostream>
using namespace std;

template <typename scalar_t>
int cuda_read_array(scalar_t **arr, string file) {
    std::ifstream input(file, ios::in | ios::binary);

    // Determine the file length
    input.seekg(0, input.end);
    int length = input.tellg();

    
    int count = length / sizeof(scalar_t);
    
    // Allocate Unified Memory
    cudaMallocManaged(arr, count * sizeof(scalar_t));
    
    // Read data from the file into memory
    input.read((char *)*arr, length);
    input.close();
    return count;
}
