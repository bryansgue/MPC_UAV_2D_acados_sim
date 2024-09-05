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
  #define CASADI_PREFIX(ID) Drone_ode_cost_ext_cost_0_fun_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int int
#endif

/* Add prefix to internal symbols */
#define casadi_c0 CASADI_PREFIX(c0)
#define casadi_c1 CASADI_PREFIX(c1)
#define casadi_c2 CASADI_PREFIX(c2)
#define casadi_clear CASADI_PREFIX(clear)
#define casadi_copy CASADI_PREFIX(copy)
#define casadi_dot CASADI_PREFIX(dot)
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_s3 CASADI_PREFIX(s3)
#define casadi_s4 CASADI_PREFIX(s4)
#define casadi_s5 CASADI_PREFIX(s5)
#define casadi_sq CASADI_PREFIX(sq)

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

casadi_real casadi_sq(casadi_real x) { return x*x;}

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

casadi_real casadi_dot(casadi_int n, const casadi_real* x, const casadi_real* y) {
  casadi_int i;
  casadi_real r = 0;
  for (i=0; i<n; ++i) r += *x++ * *y++;
  return r;
}

static const casadi_int casadi_s0[4] = {0, 0, 0, 0};
static const casadi_int casadi_s1[10] = {6, 1, 0, 6, 0, 1, 2, 3, 4, 5};
static const casadi_int casadi_s2[6] = {2, 1, 0, 2, 0, 1};
static const casadi_int casadi_s3[3] = {0, 0, 0};
static const casadi_int casadi_s4[17] = {13, 1, 0, 13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
static const casadi_int casadi_s5[5] = {1, 1, 0, 1, 0};

static const casadi_real casadi_c0[4] = {1., 0., 0., 1.};
static const casadi_real casadi_c1[4] = {5., 0., 0., 5.};
static const casadi_real casadi_c2[4] = {1.0000000000000000e-02, 0., 0., 5.0000000000000001e-03};

/* Drone_ode_cost_ext_cost_0_fun:(i0[6],i1[2],i2[],i3[13])->(o0) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_int i, j, k;
  casadi_real *rr, *ss, *tt;
  const casadi_int *cii;
  const casadi_real *cs;
  casadi_real w0, w1, w2, *w3=w+5, *w4=w+7, *w5=w+9, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15, w16, w17, w18, w19, *w20=w+27, *w21=w+40, *w22=w+42, *w23=w+44, *w24=w+48, *w25=w+54;
  /* #0: @0 = 1.5 */
  w0 = 1.5000000000000000e+00;
  /* #1: @1 = input[0][2] */
  w1 = arg[0] ? arg[0][2] : 0;
  /* #2: @2 = sq(@1) */
  w2 = casadi_sq( w1 );
  /* #3: @0 = (@0*@2) */
  w0 *= w2;
  /* #4: @2 = 0 */
  w2 = 0.;
  /* #5: @3 = zeros(1x2) */
  casadi_clear(w3, 2);
  /* #6: @4 = zeros(2x1) */
  casadi_clear(w4, 2);
  /* #7: @5 = 
  [[1, 0], 
   [0, 1]] */
  casadi_copy(casadi_c0, 4, w5);
  /* #8: @6 = 0 */
  w6 = 0.;
  /* #9: @7 = input[3][0] */
  w7 = arg[3] ? arg[3][0] : 0;
  /* #10: @8 = input[3][1] */
  w8 = arg[3] ? arg[3][1] : 0;
  /* #11: @9 = input[3][2] */
  w9 = arg[3] ? arg[3][2] : 0;
  /* #12: @10 = input[3][3] */
  w10 = arg[3] ? arg[3][3] : 0;
  /* #13: @11 = input[3][4] */
  w11 = arg[3] ? arg[3][4] : 0;
  /* #14: @12 = input[3][5] */
  w12 = arg[3] ? arg[3][5] : 0;
  /* #15: @13 = input[3][6] */
  w13 = arg[3] ? arg[3][6] : 0;
  /* #16: @14 = input[3][7] */
  w14 = arg[3] ? arg[3][7] : 0;
  /* #17: @15 = input[3][8] */
  w15 = arg[3] ? arg[3][8] : 0;
  /* #18: @16 = input[3][9] */
  w16 = arg[3] ? arg[3][9] : 0;
  /* #19: @17 = input[3][10] */
  w17 = arg[3] ? arg[3][10] : 0;
  /* #20: @18 = input[3][11] */
  w18 = arg[3] ? arg[3][11] : 0;
  /* #21: @19 = input[3][12] */
  w19 = arg[3] ? arg[3][12] : 0;
  /* #22: @20 = vertcat(@7, @8, @9, @10, @11, @12, @13, @14, @15, @16, @17, @18, @19) */
  rr=w20;
  *rr++ = w7;
  *rr++ = w8;
  *rr++ = w9;
  *rr++ = w10;
  *rr++ = w11;
  *rr++ = w12;
  *rr++ = w13;
  *rr++ = w14;
  *rr++ = w15;
  *rr++ = w16;
  *rr++ = w17;
  *rr++ = w18;
  *rr++ = w19;
  /* #23: @21 = @20[3:5] */
  for (rr=w21, ss=w20+3; ss!=w20+5; ss+=1) *rr++ = *ss;
  /* #24: @22 = @21' */
  casadi_copy(w21, 2, w22);
  /* #25: @6 = mac(@22,@21,@6) */
  for (i=0, rr=(&w6); i<1; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w22+j, tt=w21+i*2; k<2; ++k) *rr += ss[k*1]**tt++;
  /* #26: @23 = @6[0, 0, 0, 0] */
  for (cii=casadi_s0, rr=w23, ss=(&w6); cii!=casadi_s0+4; ++cii) *rr++ = *cii>=0 ? ss[*cii] : 0;
  /* #27: @5 = (@5-@23) */
  for (i=0, rr=w5, cs=w23; i<4; ++i) (*rr++) -= (*cs++);
  /* #28: @22 = @20[:2] */
  for (rr=w22, ss=w20+0; ss!=w20+2; ss+=1) *rr++ = *ss;
  /* #29: @6 = input[0][0] */
  w6 = arg[0] ? arg[0][0] : 0;
  /* #30: @7 = input[0][1] */
  w7 = arg[0] ? arg[0][1] : 0;
  /* #31: @8 = input[0][3] */
  w8 = arg[0] ? arg[0][3] : 0;
  /* #32: @9 = input[0][4] */
  w9 = arg[0] ? arg[0][4] : 0;
  /* #33: @10 = input[0][5] */
  w10 = arg[0] ? arg[0][5] : 0;
  /* #34: @24 = vertcat(@6, @7, @1, @8, @9, @10) */
  rr=w24;
  *rr++ = w6;
  *rr++ = w7;
  *rr++ = w1;
  *rr++ = w8;
  *rr++ = w9;
  *rr++ = w10;
  /* #35: @25 = @24[:2] */
  for (rr=w25, ss=w24+0; ss!=w24+2; ss+=1) *rr++ = *ss;
  /* #36: @22 = (@22-@25) */
  for (i=0, rr=w22, cs=w25; i<2; ++i) (*rr++) -= (*cs++);
  /* #37: @4 = mac(@5,@22,@4) */
  for (i=0, rr=w4; i<1; ++i) for (j=0; j<2; ++j, ++rr) for (k=0, ss=w5+j, tt=w22+i*2; k<2; ++k) *rr += ss[k*2]**tt++;
  /* #38: @25 = @4' */
  casadi_copy(w4, 2, w25);
  /* #39: @5 = 
  [[5, 0], 
   [0, 5]] */
  casadi_copy(casadi_c1, 4, w5);
  /* #40: @3 = mac(@25,@5,@3) */
  for (i=0, rr=w3; i<2; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w25+j, tt=w5+i*2; k<2; ++k) *rr += ss[k*1]**tt++;
  /* #41: @2 = mac(@3,@4,@2) */
  for (i=0, rr=(&w2); i<1; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w3+j, tt=w4+i*2; k<2; ++k) *rr += ss[k*1]**tt++;
  /* #42: @6 = 0 */
  w6 = 0.;
  /* #43: @3 = zeros(1x2) */
  casadi_clear(w3, 2);
  /* #44: @7 = dot(@21, @22) */
  w7 = casadi_dot(2, w21, w22);
  /* #45: @22 = (@7*@21) */
  for (i=0, rr=w22, cs=w21; i<2; ++i) (*rr++)  = (w7*(*cs++));
  /* #46: @4 = @22' */
  casadi_copy(w22, 2, w4);
  /* #47: @5 = 
  [[5, 0], 
   [0, 5]] */
  casadi_copy(casadi_c1, 4, w5);
  /* #48: @3 = mac(@4,@5,@3) */
  for (i=0, rr=w3; i<2; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w4+j, tt=w5+i*2; k<2; ++k) *rr += ss[k*1]**tt++;
  /* #49: @6 = mac(@3,@22,@6) */
  for (i=0, rr=(&w6); i<1; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w3+j, tt=w22+i*2; k<2; ++k) *rr += ss[k*1]**tt++;
  /* #50: @2 = (@2+@6) */
  w2 += w6;
  /* #51: @0 = (@0+@2) */
  w0 += w2;
  /* #52: @2 = 0.2 */
  w2 = 2.0000000000000001e-01;
  /* #53: @3 = @24[3:5] */
  for (rr=w3, ss=w24+3; ss!=w24+5; ss+=1) *rr++ = *ss;
  /* #54: @6 = dot(@21, @3) */
  w6 = casadi_dot(2, w21, w3);
  /* #55: @2 = (@2*@6) */
  w2 *= w6;
  /* #56: @0 = (@0-@2) */
  w0 -= w2;
  /* #57: @2 = 0 */
  w2 = 0.;
  /* #58: @21 = zeros(1x2) */
  casadi_clear(w21, 2);
  /* #59: @6 = input[1][0] */
  w6 = arg[1] ? arg[1][0] : 0;
  /* #60: @7 = input[1][1] */
  w7 = arg[1] ? arg[1][1] : 0;
  /* #61: @3 = vertcat(@6, @7) */
  rr=w3;
  *rr++ = w6;
  *rr++ = w7;
  /* #62: @22 = @3' */
  casadi_copy(w3, 2, w22);
  /* #63: @5 = 
  [[0.01, 0], 
   [0, 0.005]] */
  casadi_copy(casadi_c2, 4, w5);
  /* #64: @21 = mac(@22,@5,@21) */
  for (i=0, rr=w21; i<2; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w22+j, tt=w5+i*2; k<2; ++k) *rr += ss[k*1]**tt++;
  /* #65: @2 = mac(@21,@3,@2) */
  for (i=0, rr=(&w2); i<1; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w21+j, tt=w3+i*2; k<2; ++k) *rr += ss[k*1]**tt++;
  /* #66: @0 = (@0+@2) */
  w0 += w2;
  /* #67: output[0][0] = @0 */
  if (res[0]) res[0][0] = w0;
  return 0;
}

CASADI_SYMBOL_EXPORT int Drone_ode_cost_ext_cost_0_fun(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int Drone_ode_cost_ext_cost_0_fun_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int Drone_ode_cost_ext_cost_0_fun_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void Drone_ode_cost_ext_cost_0_fun_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int Drone_ode_cost_ext_cost_0_fun_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void Drone_ode_cost_ext_cost_0_fun_release(int mem) {
}

CASADI_SYMBOL_EXPORT void Drone_ode_cost_ext_cost_0_fun_incref(void) {
}

CASADI_SYMBOL_EXPORT void Drone_ode_cost_ext_cost_0_fun_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int Drone_ode_cost_ext_cost_0_fun_n_in(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_int Drone_ode_cost_ext_cost_0_fun_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real Drone_ode_cost_ext_cost_0_fun_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* Drone_ode_cost_ext_cost_0_fun_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* Drone_ode_cost_ext_cost_0_fun_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* Drone_ode_cost_ext_cost_0_fun_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s1;
    case 1: return casadi_s2;
    case 2: return casadi_s3;
    case 3: return casadi_s4;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* Drone_ode_cost_ext_cost_0_fun_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s5;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int Drone_ode_cost_ext_cost_0_fun_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 17;
  if (sz_res) *sz_res = 2;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 56;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
