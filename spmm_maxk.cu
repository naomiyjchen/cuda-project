#include "spmm_maxk.h"
#include "data.h"
#include <string>
#include <iostream>
#define CONSTINT const int

using namespace std;

extern string base_dir, graph;

const int WARPS_PER_BLOCK = 12;
const int EXT_WARP_DIM = 32;

#define DIM_MUL_N 1
#define DIM_MUL(x) ((x + DIM_MUL_N - 1) / DIM_MUL_N) * DIM_MUL_N



void SPMM_MAXK::run(int dim) {
    int shared_size = WARPS_PER_BLOCK * dim * sizeof(float);

}

double SPMM_MAXK::do_test(bool timing, int dim) {
    double ret = 0.0;
    return ret;
}
