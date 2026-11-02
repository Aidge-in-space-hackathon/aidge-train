#define MODE_FIXQ
#define ACC_QUAL static __attribute__((always_inline)) inline
#define LAYER_DIMS "small_dims.h"
#define LAYER_WEIGHTS "small_weights.h"
#include "conv_c.c"
#include "input.h"
#include "expected_MODE_FIXQ.h"
int check(int dummy) {
    static struct param_conv p;
    static struct result_conv r;
    int c, y, x, k = 0, bad = 0;
    for (c = 0; c < C_IN; c++) for (y = 0; y < H_IN; y++) for (x = 0; x < W_IN; x++) p.in[c][y][x] = INPUT[c][y][x];
    conv_acc(&p, &r);
    for (c = 0; c < C_OUT; c++) for (y = 0; y < H_OUT; y++) for (x = 0; x < W_OUT; x++) {
        if (r.out[c][y][x] != EXPECTED[k]) bad++;
        k++;
    }
    return bad + dummy;
}
