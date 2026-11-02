#!/bin/bash
# GDN flottante : avec -lm (absent des options de LoCod), division seule, racine de Newton écrite en C.
source <(sed -n '/^A=/,/^mkdir -p \$RES/p' /out/campaign.sh)
source <(sed -n '/^run() {/,/^}/p' /out/campaign.sh)
cat > /out/gdn_div.c <<'E'
#define N_VAL 64
struct param_gdn { float x[N_VAL]; float norm[N_VAL]; };
struct result_gdn { float y[N_VAL]; };
void gdn_acc(struct param_gdn *param, struct result_gdn *result) {
    int i;
    for (i = 0; i < N_VAL; i++) result->y[i] = param->x[i] / param->norm[i];
}
E
cat > /out/gdn_newton.c <<'E'
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
E
run C5b_gdn_lm 1200 gdn_acc /out/gdn.c -I/out -lm
run C6_gdn_div_seule 1200 gdn_acc /out/gdn_div.c -I/out
run C7_gdn_newton 1200 gdn_acc /out/gdn_newton.c -I/out
for n in C5b_gdn_lm C7_gdn_newton; do echo "== $n"; grep -a -E 'error ->|does not exist' /out/res/$n/log.txt | cut -c1-170 | head -4; done
