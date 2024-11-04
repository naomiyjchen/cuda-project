#include <iostream>
#include "data.h"

string BASE_DIR = "../graphs/";

using namespace std;

// Define the TRACE macro
#ifdef DEBUG
    #define TRACE(msg) \
            std::cerr << "[TRACE] " << __FILE__ << ":" << __LINE__ << " (" << __func__ << ") - " << msg << std::endl;
#else
    #define TRACE(msg) // No operation
#endif


void test_graph(string graph) {
    int dim_origin = 256;  // dimension of the input embedding
    int k = 16; 

    // Allocate memory 
    int *indptr_d, *indices_d;


    int v_num = cuda_read_array(&indptr_d, BASE_DIR + graph + ".indptr") - 1;
    int e_num = cuda_read_array(&indices_d, BASE_DIR + graph + ".indices");

    TRACE("v_num: " << v_num);
    TRACE("e_num: " << e_num);

}

int main(int argc, char *argv[]){
    if (argc > 1){ 
        string arg_graph(argv[1]);
        test_graph(arg_graph);
    } else {
        cout << "Please specify the graph." << endl;
    }   

    return 0;
}
