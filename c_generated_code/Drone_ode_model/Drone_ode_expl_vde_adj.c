/* This file was automatically generated by CasADi.
   The CasADi copyright holders make no ownership claim of its contents. */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CASADI_CODEGEN_PREFIX
  #define CASADI_NAMESPACE_CONCAT(NS, ID) _CASADI_NAMESPACE_CONCAT(NS, ID)
  #define _CASADI_NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) CASADI_NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) Drone_ode_expl_vde_adj_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int int
#endif

/* Add prefix to internal symbols */
#define casadi_clear CASADI_PREFIX(clear)
#define casadi_copy CASADI_PREFIX(copy)
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_s3 CASADI_PREFIX(s3)

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

void casadi_clear(casadi_real* x, casadi_int n) {
  casadi_int i;
  if (x) {
    for (i=0; i<n; ++i) *x++ = 0;
  }
}

void casadi_copy(const casadi_real* x, casadi_int n, casadi_real* y) {
  casadi_int i;
  if (y) {
    if (x) {
      for (i=0; i<n; ++i) *y++ = *x++;
    } else {
      for (i=0; i<n; ++i) *y++ = 0.;
    }
  }
}

static const casadi_int casadi_s0[10] = {6, 1, 0, 6, 0, 1, 2, 3, 4, 5};
static const casadi_int casadi_s1[6] = {2, 1, 0, 2, 0, 1};
static const casadi_int casadi_s2[17] = {13, 1, 0, 13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
static const casadi_int casadi_s3[10] = {8, 1, 0, 6, 2, 3, 4, 5, 6, 7};

/* Drone_ode_expl_vde_adj:(i0[6],i1[6],i2[2],i3[13])->(o0[8x1,6nz]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_int i, j, k;
  casadi_real *rr, *ss, *tt;
  const casadi_real *cs;
  casadi_real w0, w1, *w2=w+8, *w3=w+20, w4, w5, *w6=w+28, *w7=w+30, *w8=w+42, *w9=w+48, *w10=w+54;
  /* #0: @0 = input[0][2] */
  w0 = arg[0] ? arg[0][2] : 0;
  /* #1: @1 = cos(@0) */
  w1 = cos( w0 );
  /* #2: @2 = zeros(6x2) */
  casadi_clear(w2, 12);
  /* #3: @3 = input[1][0] */
  casadi_copy(arg[1], 6, w3);
  /* #4: @4 = input[2][0] */
  w4 = arg[2] ? arg[2][0] : 0;
  /* #5: @5 = input[2][1] */
  w5 = arg[2] ? arg[2][1] : 0;
  /* #6: @6 = vertcat(@4, @5) */
  rr=w6;
  *rr++ = w4;
  *rr++ = w5;
  /* #7: @6 = @6' */
  /* #8: @2 = mac(@3,@6,@2) */
  for (i=0, rr=w2; i<2; ++i) for (j=0; j<6; ++j, ++rr) for (k=0, ss=w3+j, tt=w6+i*1; k<1; ++k) *rr += ss[k*6]**tt++;
  /* #9: @4 = 0 */
  w4 = 0.;
  /* #10: (@2[11] = @4) */
  for (rr=w2+11, ss=(&w4); rr!=w2+12; rr+=1) *rr = *ss++;
  /* #11: @4 = 0 */
  w4 = 0.;
  /* #12: (@2[10] = @4) */
  for (rr=w2+10, ss=(&w4); rr!=w2+11; rr+=1) *rr = *ss++;
  /* #13: @4 = 0 */
  w4 = 0.;
  /* #14: (@2[9] = @4) */
  for (rr=w2+9, ss=(&w4); rr!=w2+10; rr+=1) *rr = *ss++;
  /* #15: @4 = 0 */
  w4 = 0.;
  /* #16: (@2[5] = @4) */
  for (rr=w2+5, ss=(&w4); rr!=w2+6; rr+=1) *rr = *ss++;
  /* #17: @4 = 0 */
  w4 = 0.;
  /* #18: @7 = @2; (@7[4] = @4) */
  casadi_copy(w2, 12, w7);
  for (rr=w7+4, ss=(&w4); rr!=w7+5; rr+=1) *rr = *ss++;
  /* #19: @4 = @7[3] */
  for (rr=(&w4), ss=w7+3; ss!=w7+4; ss+=1) *rr++ = *ss;
  /* #20: @1 = (@1*@4) */
  w1 *= w4;
  /* #21: @1 = (-@1) */
  w1 = (- w1 );
  /* #22: @4 = sin(@0) */
  w4 = sin( w0 );
  /* #23: @5 = @2[4] */
  for (rr=(&w5), ss=w2+4; ss!=w2+5; ss+=1) *rr++ = *ss;
  /* #24: @4 = (@4*@5) */
  w4 *= w5;
  /* #25: @1 = (@1-@4) */
  w1 -= w4;
  /* #26: output[0][0] = @1 */
  if (res[0]) res[0][0] = w1;
  /* #27: @1 = 0 */
  w1 = 0.;
  /* #28: @8 = @3; (@8[5] = @1) */
  casadi_copy(w3, 6, w8);
  for (rr=w8+5, ss=(&w1); rr!=w8+6; rr+=1) *rr = *ss++;
  /* #29: @1 = 0 */
  w1 = 0.;
  /* #30: (@8[4] = @1) */
  for (rr=w8+4, ss=(&w1); rr!=w8+5; rr+=1) *rr = *ss++;
  /* #31: @1 = 0 */
  w1 = 0.;
  /* #32: (@8[3] = @1) */
  for (rr=w8+3, ss=(&w1); rr!=w8+4; rr+=1) *rr = *ss++;
  /* #33: @1 = 0 */
  w1 = 0.;
  /* #34: @9 = @8; (@9[2] = @1) */
  casadi_copy(w8, 6, w9);
  for (rr=w9+2, ss=(&w1); rr!=w9+3; rr+=1) *rr = *ss++;
  /* #35: @1 = 0 */
  w1 = 0.;
  /* #36: @10 = @9; (@10[1] = @1) */
  casadi_copy(w9, 6, w10);
  for (rr=w10+1, ss=(&w1); rr!=w10+2; rr+=1) *rr = *ss++;
  /* #37: @1 = @10[0] */
  for (rr=(&w1), ss=w10+0; ss!=w10+1; ss+=1) *rr++ = *ss;
  /* #38: output[0][1] = @1 */
  if (res[0]) res[0][1] = w1;
  /* #39: @1 = @9[1] */
  for (rr=(&w1), ss=w9+1; ss!=w9+2; ss+=1) *rr++ = *ss;
  /* #40: output[0][2] = @1 */
  if (res[0]) res[0][2] = w1;
  /* #41: @1 = @8[2] */
  for (rr=(&w1), ss=w8+2; ss!=w8+3; ss+=1) *rr++ = *ss;
  /* #42: output[0][3] = @1 */
  if (res[0]) res[0][3] = w1;
  /* #43: @6 = zeros(2x1) */
  casadi_clear(w6, 2);
  /* #44: @2 = zeros(6x2) */
  casadi_clear(w2, 12);
  /* #45: @1 = sin(@0) */
  w1 = sin( w0 );
  /* #46: @1 = (-@1) */
  w1 = (- w1 );
  /* #47: (@2[3] = @1) */
  for (rr=w2+3, ss=(&w1); rr!=w2+4; rr+=1) *rr = *ss++;
  /* #48: @0 = cos(@0) */
  w0 = cos( w0 );
  /* #49: (@2[4] = @0) */
  for (rr=w2+4, ss=(&w0); rr!=w2+5; rr+=1) *rr = *ss++;
  /* #50: @0 = 0 */
  w0 = 0.;
  /* #51: (@2[5] = @0) */
  for (rr=w2+5, ss=(&w0); rr!=w2+6; rr+=1) *rr = *ss++;
  /* #52: @0 = 0 */
  w0 = 0.;
  /* #53: (@2[9] = @0) */
  for (rr=w2+9, ss=(&w0); rr!=w2+10; rr+=1) *rr = *ss++;
  /* #54: @0 = 0 */
  w0 = 0.;
  /* #55: (@2[10] = @0) */
  for (rr=w2+10, ss=(&w0); rr!=w2+11; rr+=1) *rr = *ss++;
  /* #56: @0 = 50 */
  w0 = 50.;
  /* #57: (@2[11] = @0) */
  for (rr=w2+11, ss=(&w0); rr!=w2+12; rr+=1) *rr = *ss++;
  /* #58: @7 = @2' */
  for (i=0, rr=w7, cs=w2; i<2; ++i) for (j=0; j<6; ++j) rr[i+j*2] = *cs++;
  /* #59: @6 = mac(@7,@3,@6) */
  for (i=0, rr=w6; i<1; ++i) for (j=0; j<2; ++j, ++rr) for (k=0, ss=w7+j, tt=w3+i*6; k<6; ++k) *rr += ss[k*2]**tt++;
  /* #60: {@0, @1} = vertsplit(@6) */
  w0 = w6[0];
  w1 = w6[1];
  /* #61: output[0][4] = @0 */
  if (res[0]) res[0][4] = w0;
  /* #62: output[0][5] = @1 */
  if (res[0]) res[0][5] = w1;
  return 0;
}

CASADI_SYMBOL_EXPORT int Drone_ode_expl_vde_adj(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int Drone_ode_expl_vde_adj_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int Drone_ode_expl_vde_adj_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void Drone_ode_expl_vde_adj_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int Drone_ode_expl_vde_adj_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void Drone_ode_expl_vde_adj_release(int mem) {
}

CASADI_SYMBOL_EXPORT void Drone_ode_expl_vde_adj_incref(void) {
}

CASADI_SYMBOL_EXPORT void Drone_ode_expl_vde_adj_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int Drone_ode_expl_vde_adj_n_in(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_int Drone_ode_expl_vde_adj_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real Drone_ode_expl_vde_adj_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* Drone_ode_expl_vde_adj_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* Drone_ode_expl_vde_adj_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* Drone_ode_expl_vde_adj_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s0;
    case 2: return casadi_s1;
    case 3: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* Drone_ode_expl_vde_adj_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s3;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int Drone_ode_expl_vde_adj_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 7;
  if (sz_res) *sz_res = 3;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 60;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
