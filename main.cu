#include <algorithm>
#include <iostream>
#include <random>
#include "data.h"
#include "spmm_maxk.h"

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
    int dim_origin = 256;  // dimension of the original input embedding
    int dim_k_list[] = {16, 32, 64, 96, 128}; 
    int dim_k_limit = 64;

    // Read graph to device
    int *indptr_d, *indices_d;
    int v_num = cuda_read_array(&indptr_d, BASE_DIR + graph + ".indptr") - 1;
    int e_num = cuda_read_array(&indices_d, BASE_DIR + graph + ".indices");

    TRACE("v_num: " << v_num);
    TRACE("e_num: " << e_num);


    // Allocate memory on device
    float *e_val_d;  
    float *vin_sparse_d; 
    float *vin_sparse_k_d;
    float *vout_maxk_d;

    cudaMallocManaged(&e_val_d, e_num * sizeof(float));
    cudaMallocManaged(&vin_sparse_d, v_num * dim_origin * sizeof(float));
    cudaMallocManaged(&vin_sparse_k_d, v_num * DIM_MUL(dim_k_limit) * sizeof(float));
    cudaMallocManaged(&vout_maxk_d, v_num * dim_origin * sizeof(float));
   
    // Data Initialization
    default_random_engine engine;
    engine.seed(123);
    uniform_real_distribution<float> rd(0, 1);

    generate(e_val_d, e_val_d + e_num, [&](){ return rd(engine); });
    generate(vin_sparse_k_d, vin_sparse_k_d + v_num * dim_k_limit, [&]() { return rd(engine); });
    generate(vin_sparse_d, vin_sparse_d + v_num * dim_origin, [&]() { return rd(engine); });


    // Initiate forward propagation instance
    SPMM_MAXK maxk(graph, indptr_d, indices_d, e_val_d, vin_sparse_k_d, vout_maxk_d, v_num, e_num, dim_origin);
    
    // Free the memory
    cudaFree(indptr_d);
    cudaFree(indices_d);
    cudaFree(e_val_d);
    cudaFree(vin_sparse_d);
    cudaFree(vin_sparse_k_d);
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
