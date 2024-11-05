#include <iostream>
#include "data.h"
#include <random>
#include "spmm_maxk.h"
#include <algorithm>

string BASE_DIR = "../graphs/";

using namespace std;

#define DIM_MUL_N 1
#define DIM_MUL(x) ((x + DIM_MUL_N - 1) / DIM_MUL_N) * DIM_MUL_N

// Define the TRACE macro
#ifdef DEBUG
    #define TRACE(msg) \
            std::cerr << "[TRACE] " << __FILE__ << ":" << __LINE__ << " (" << __func__ << ") - " << msg << std::endl;
#else
    #define TRACE(msg) // No operation
#endif


void test_graph(string graph) {
    int dim_origin = 256;  // dimension of the input embedding
    int dim_k_list[] = {16, 32, 64, 96, 128}; 
    int dim_k_limit = 64;

    // Allocate memory 
    int *indptr_d, *indices_d;
    int v_num = cuda_read_array(&indptr_d, BASE_DIR + graph + ".indptr") - 1;
    int e_num = cuda_read_array(&indices_d, BASE_DIR + graph + ".indices");

    TRACE("v_num: " << v_num);
    TRACE("e_num: " << e_num);

    float *cu_val;
    cudaMallocManaged(&cu_val, e_num * sizeof(float));
    
    
    float *vin_sparse_d, *vin_sparse_data_d;
    float *vout_maxk_d;

    cudaMallocManaged(&vin_sparse_d, v_num * dim_origin * sizeof(float));
    cudaMallocManaged(&vin_sparse_data_d, v_num * DIM_MUL(dim_k_limit) * sizeof(float));
    cudaMallocManaged(&vout_maxk_d, v_num * dim_origin * sizeof(float));
   

    // Data Initialization

    default_random_engine engine;
    engine.seed(123);
    uniform_real_distribution<float> rd(0, 1);

    generate(cu_val, cu_val + e_num, [&](){ return rd(engine); });
    generate(vin_sparse_data_d, vin_sparse_data_d + v_num * dim_k_limit, [&]() { return rd(engine); });
    generate(vin_sparse_d, vin_sparse_d + v_num * dim_origin, [&]() { return rd(engine); });


    // Free the memory
    cudaFree(indptr_d);
    cudaFree(indices_d);
    cudaFree(cu_val);
    cudaFree(vin_sparse_d);
    cudaFree(vin_sparse_data_d);
    cudaFree(vout_maxk_d);
}


int main(int argc, char *argv[]){
    if (argc > 1){ 
        string arg_graph(argv[1]);
        TRACE("graph : " << arg_graph);
        test_graph(arg_graph);
    } else {
        cout << "Please specify the graph." << endl;
    }   

    return 0;
}
