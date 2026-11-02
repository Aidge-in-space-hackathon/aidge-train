/* Spike LoCod : coût de la partie flottante de la GDN (une division et une racine par valeur). */
#ifndef LOCOD_FPGA
#include <stdio.h>
#endif
#include <math.h>
#define N_VAL 64
struct param_gdn { float x[N_VAL]; float norm[N_VAL]; };
struct result_gdn { float y[N_VAL]; };
void gdn_acc(struct param_gdn *param, struct result_gdn *result) {
    int i;
    for (i = 0; i < N_VAL; i++) {
        result->y[i] = param->x[i] / sqrtf(param->norm[i]);
    }
}
