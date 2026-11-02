/* Spike LoCod : une couche de convolution INT8 en C pur, au format FPGA() (structures de taille fixe).
 * -DLAYER_DIMS="..." -DLAYER_WEIGHTS="..." choisissent la couche ; le mode choisit la sortie et la place des poids :
 *   MODE_INT32   sortie int32 de la convolution, sans Quantizer
 *   MODE_FLOATQ  Quantizer flottant d'Aidge (Mul, Round, Clip, Cast) vers int8
 *   MODE_FIXQ    Quantizer en virgule fixe (M / 2^n) vers int8
 * WEIGHTS_IN_PARAM : poids dans la structure param (DDR), sinon static const (BRAM avec la politique LSS).
 */
#ifndef LOCOD_FPGA
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#endif
#include LAYER_DIMS
#ifndef WEIGHTS_IN_PARAM
#include LAYER_WEIGHTS
#endif

struct param_conv {
    signed char in[C_IN][H_IN][W_IN];
#ifdef WEIGHTS_IN_PARAM
    signed char weights[C_OUT][C_IN][KH][KW];
    int biases[C_OUT];
#endif
};

struct result_conv {
#ifdef MODE_INT32
    int out[C_OUT][H_OUT][W_OUT];
#else
    signed char out[C_OUT][H_OUT][W_OUT];
#endif
};

#ifdef WEIGHTS_IN_PARAM
#define W_AT(c, i, y, x) param->weights[c][i][y][x]
#define B_AT(c) param->biases[c]
#else
#define W_AT(c, i, y, x) WEIGHTS[c][i][y][x]
#define B_AT(c) BIASES[c]
#endif

#ifndef ACC_QUAL
#define ACC_QUAL
#endif
ACC_QUAL void conv_acc(struct param_conv *param, struct result_conv *result) {
    int c, oy, ox, i, ky, kx;
    for (c = 0; c < C_OUT; c++) {
        for (oy = 0; oy < H_OUT; oy++) {
            for (ox = 0; ox < W_OUT; ox++) {
                int sum = B_AT(c);
                for (i = 0; i < C_IN; i++) {
                    for (ky = 0; ky < KH; ky++) {
                        int iy = oy * STRIDE + ky - PAD;
                        if (iy < 0 || iy >= H_IN) continue;
                        for (kx = 0; kx < KW; kx++) {
                            int ix = ox * STRIDE + kx - PAD;
                            if (ix < 0 || ix >= W_IN) continue;
                            sum += (int)W_AT(c, i, ky, kx) * (int)param->in[i][iy][ix];
                        }
                    }
                }
#if defined(MODE_INT32)
                result->out[c][oy][ox] = sum;
#elif defined(MODE_FLOATQ)
                {
                    float v = (float)sum * OUT_SCALE;
                    int r = (int)(v >= 0.0f ? v + 0.5f : v - 0.5f);
                    result->out[c][oy][ox] = (signed char)(r > 127 ? 127 : (r < -128 ? -128 : r));
                }
#else
                {
                    long long p = (long long)sum * FIX_M;
                    int r = (int)((p + (1LL << (FIX_N - 1))) >> FIX_N);
                    result->out[c][oy][ox] = (signed char)(r > 127 ? 127 : (r < -128 ? -128 : r));
                }
#endif
            }
        }
    }
}

#ifdef NATIVE_TEST
#ifdef WEIGHTS_IN_PARAM
#include LAYER_WEIGHTS
#endif
int main(int argc, char **argv) {
    static struct param_conv p;
    static struct result_conv r;
    FILE *f = fopen(argv[1], "rb");
    if (!f || fread(p.in, 1, sizeof(p.in), f) != sizeof(p.in)) return 2;
    fclose(f);
#ifdef WEIGHTS_IN_PARAM
    memcpy(p.weights, WEIGHTS_INIT, sizeof(p.weights));
    memcpy(p.biases, BIASES_INIT, sizeof(p.biases));
#endif
    conv_acc(&p, &r);
    f = fopen(argv[2], "wb");
    fwrite(r.out, 1, sizeof(r.out), f);
    fclose(f);
    return 0;
}
#endif
