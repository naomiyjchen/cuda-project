#include <algorithm>
#include <iostream>
#include <random>
#include "data.h"
#include "spmm_maxk.h"
#include "trace.h"

string BASE_DIR = "../graphs/";

using namespace std;

#define DIM_MUL_N 1
#define DIM_MUL(x) ((x + DIM_MUL_N - 1) / DIM_MUL_N) * DIM_MUL_N



void test_graph(string graph) {
    int dim_origin = 256;  // dimension of the original input embedding
    int dim_k_limit = 64;  

    
    /*
     * Read graph to device
     */

    int *indptr_d, *indices_d;
    int v_num = cuda_read_array(&indptr_d, BASE_DIR + graph + ".indptr") - 1;
    int e_num = cuda_read_array(&indices_d, BASE_DIR + graph + ".indices");

    TRACE("v_num: " << v_num);
    TRACE("e_num: " << e_num);


    /*
     * Allocate memory on device
     */
    
    float *e_val_d;  
    float *vin_sparse_full_d; 
    float *vin_sparse_k_values_d;
    u_int8_t *vin_sparse_selector_d;
    float *vout_maxk_d;

    cudaMallocManaged(&e_val_d, e_num * sizeof(float));
    cudaMallocManaged(&vin_sparse_full_d, v_num * dim_origin * sizeof(float));
    cudaMallocManaged(&vin_sparse_k_values_d, v_num * DIM_MUL(dim_k_limit) * sizeof(float));
    cudaMallocManaged(&vin_sparse_selector_d, v_num * DIM_MUL(dim_k_limit) * sizeof(u_int8_t));
    cudaMallocManaged(&vout_maxk_d, v_num * dim_origin * sizeof(float));
   
    
    /*
     * Data initialization
     */
    
    default_random_engine engine;
    engine.seed(123);
    uniform_real_distribution<float> rd(0, 1);

    generate(e_val_d, e_val_d + e_num, [&](){ return rd(engine); });
    generate(vin_sparse_k_values_d, vin_sparse_k_values_d + v_num * dim_k_limit, [&]() { return rd(engine); });
    generate(vin_sparse_full_d, vin_sparse_full_d + v_num * dim_origin, [&]() { return rd(engine); });


    // Initiate forward propagation instance
    SPMM_MAXK maxk(graph, indptr_d, indices_d, e_val_d, vin_sparse_k_values_d, vout_maxk_d, v_num, e_num, dim_origin);
   

    /*
     * Sampling and selecting sparse features
     */

    int dim_k = 16;  // Set k
    vector<int> sampled_indices(dim_k);  // Stores the sampled indices

    // Create a sequence for eampling
    vector<int> sequence(dim_origin);
    iota(sequence.begin(), sequence.end(), 0);
    

    // Sample the vin_sparse_k_values, and select its corresponding indices in vin_sparse_full
    for (int i = 0; i < v_num; ++i){
        std::sample(sequence.begin(), sequence.end(), sampled_indices.begin(), dim_k, engine);
        
        for (int j = 0; j < dim_k; ++j) {
            float v = rd(engine);
            int idx = i * DIM_MUL(dim_k) + j;
            vin_sparse_k_values_d[idx] = v; 
            vin_sparse_selector_d[idx] = sampled_indices[j];

        }
    }

    // Populating sparse original input embedding
    // with the sparse k features
    for (int i = 0; i < v_num; ++i) {
        
        // Initialize all value to zero
        for (int j = 0; j < dim_origin; ++j) {
            vin_sparse_full_d[i * dim_origin + j] = 0.0;
        }

        // Fill in the values in vin_sparse_k_values_d
        for (int j = 0; j < dim_k; ++j) {
            int col = vin_sparse_selector_d[i * DIM_MUL(dim_k) + j];
            vin_sparse_full_d[i * dim_origin + col] = vin_sparse_k_values_d[i * DIM_MUL(dim_k) + j];
        }

    }


    maxk.vin_sparse_selector = vin_sparse_selector_d;
    maxk.dim_sparse = dim_k;
    

    /*
     * Forward Pass
     */

    double time_maxk = maxk.do_test(true, dim_origin);
    cout << graph << " maxk " << time_maxk * 1000 << endl;
    
    
    // Free the memory
    cudaFree(indptr_d);
    cudaFree(indices_d);
    cudaFree(e_val_d);
    cudaFree(vin_sparse_full_d);
    cudaFree(vin_sparse_k_values_d);
    cudaFree(vin_sparse_selector_d);
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
