/* Racine sans libm : estimation par les bits de l'exposant, puis 3 itérations de Newton. */
#define N_VAL 64
struct param_gdn { float x[N_VAL]; float norm[N_VAL]; };
struct result_gdn { float y[N_VAL]; };
union f2i { float f; unsigned int i; };
void gdn_acc(struct param_gdn *param, struct result_gdn *result) {
    int i, k;
    for (i = 0; i < N_VAL; i++) {
        union f2i u;
        float s;
        u.f = param->norm[i];
        u.i = (u.i >> 1) + 0x1fbd1df5u;
        s = u.f;
        for (k = 0; k < 3; k++) s = 0.5f * (s + param->norm[i] / s);
        result->y[i] = param->x[i] / s;
    }
}
