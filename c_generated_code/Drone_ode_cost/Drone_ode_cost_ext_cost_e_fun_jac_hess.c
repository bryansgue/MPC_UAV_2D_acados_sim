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
  #define CASADI_PREFIX(ID) Drone_ode_cost_ext_cost_e_fun_jac_hess_ ## ID
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
#define casadi_clear CASADI_PREFIX(clear)
#define casadi_copy CASADI_PREFIX(copy)
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_mtimes CASADI_PREFIX(mtimes)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s10 CASADI_PREFIX(s10)
#define casadi_s11 CASADI_PREFIX(s11)
#define casadi_s12 CASADI_PREFIX(s12)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_s3 CASADI_PREFIX(s3)
#define casadi_s4 CASADI_PREFIX(s4)
#define casadi_s5 CASADI_PREFIX(s5)
#define casadi_s6 CASADI_PREFIX(s6)
#define casadi_s7 CASADI_PREFIX(s7)
#define casadi_s8 CASADI_PREFIX(s8)
#define casadi_s9 CASADI_PREFIX(s9)

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

void casadi_mtimes(const casadi_real* x, const casadi_int* sp_x, const casadi_real* y, const casadi_int* sp_y, casadi_real* z, const casadi_int* sp_z, casadi_real* w, casadi_int tr) {
  casadi_int ncol_x, ncol_y, ncol_z, cc;
  const casadi_int *colind_x, *row_x, *colind_y, *row_y, *colind_z, *row_z;
  ncol_x = sp_x[1];
  colind_x = sp_x+2; row_x = sp_x + 2 + ncol_x+1;
  ncol_y = sp_y[1];
  colind_y = sp_y+2; row_y = sp_y + 2 + ncol_y+1;
  ncol_z = sp_z[1];
  colind_z = sp_z+2; row_z = sp_z + 2 + ncol_z+1;
  if (tr) {
    for (cc=0; cc<ncol_z; ++cc) {
      casadi_int kk;
      for (kk=colind_y[cc]; kk<colind_y[cc+1]; ++kk) {
        w[row_y[kk]] = y[kk];
      }
      for (kk=colind_z[cc]; kk<colind_z[cc+1]; ++kk) {
        casadi_int kk1;
        casadi_int rr = row_z[kk];
        for (kk1=colind_x[rr]; kk1<colind_x[rr+1]; ++kk1) {
          z[kk] += x[kk1] * w[row_x[kk1]];
        }
      }
    }
  } else {
    for (cc=0; cc<ncol_y; ++cc) {
      casadi_int kk;
      for (kk=colind_z[cc]; kk<colind_z[cc+1]; ++kk) {
        w[row_z[kk]] = z[kk];
      }
      for (kk=colind_y[cc]; kk<colind_y[cc+1]; ++kk) {
        casadi_int kk1;
        casadi_int rr = row_y[kk];
        for (kk1=colind_x[rr]; kk1<colind_x[rr+1]; ++kk1) {
          w[row_x[kk1]] += x[kk1]*y[kk];
        }
      }
      for (kk=colind_z[cc]; kk<colind_z[cc+1]; ++kk) {
        z[kk] = w[row_z[kk]];
      }
    }
  }
}

static const casadi_int casadi_s0[15] = {1, 6, 0, 1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0};
static const casadi_int casadi_s1[45] = {6, 6, 0, 6, 12, 18, 24, 30, 36, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5};
static const casadi_int casadi_s2[10] = {1, 6, 0, 1, 1, 1, 1, 1, 1, 0};
static const casadi_int casadi_s3[10] = {1, 6, 0, 0, 1, 1, 1, 1, 1, 0};
static const casadi_int casadi_s4[10] = {1, 6, 0, 0, 0, 1, 1, 1, 1, 0};
static const casadi_int casadi_s5[10] = {1, 6, 0, 0, 0, 0, 1, 1, 1, 0};
static const casadi_int casadi_s6[10] = {1, 6, 0, 0, 0, 0, 0, 1, 1, 0};
static const casadi_int casadi_s7[10] = {1, 6, 0, 0, 0, 0, 0, 0, 1, 0};
static const casadi_int casadi_s8[10] = {6, 1, 0, 6, 0, 1, 2, 3, 4, 5};
static const casadi_int casadi_s9[3] = {0, 0, 0};
static const casadi_int casadi_s10[12] = {8, 1, 0, 8, 0, 1, 2, 3, 4, 5, 6, 7};
static const casadi_int casadi_s11[5] = {1, 1, 0, 1, 0};
static const casadi_int casadi_s12[9] = {0, 6, 0, 0, 0, 0, 0, 0, 0};

static const casadi_real casadi_c0[36] = {4., 0., 0., 0., 0., 0., 0., 4., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.};

/* Drone_ode_cost_ext_cost_e_fun_jac_hess:(i0[6],i1[],i2[],i3[8])->(o0,o1[6],o2[6x6],o3[],o4[0x6]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_int i, j, k;
  casadi_real *rr, *ss, *tt;
  const casadi_real *cs;
  casadi_real w0, *w1=w+2, w2, w3, w4, w5, w6, w7, w8, w9, *w10=w+16, *w11=w+24, *w12=w+30, *w13=w+36, *w14=w+72, *w15=w+108;
  /* #0: @0 = 0 */
  w0 = 0.;
  /* #1: @1 = zeros(1x6) */
  casadi_clear(w1, 6);
  /* #2: @2 = input[3][0] */
  w2 = arg[3] ? arg[3][0] : 0;
  /* #3: @3 = input[3][1] */
  w3 = arg[3] ? arg[3][1] : 0;
  /* #4: @4 = input[3][2] */
  w4 = arg[3] ? arg[3][2] : 0;
  /* #5: @5 = input[3][3] */
  w5 = arg[3] ? arg[3][3] : 0;
  /* #6: @6 = input[3][4] */
  w6 = arg[3] ? arg[3][4] : 0;
  /* #7: @7 = input[3][5] */
  w7 = arg[3] ? arg[3][5] : 0;
  /* #8: @8 = input[3][6] */
  w8 = arg[3] ? arg[3][6] : 0;
  /* #9: @9 = input[3][7] */
  w9 = arg[3] ? arg[3][7] : 0;
  /* #10: @10 = vertcat(@2, @3, @4, @5, @6, @7, @8, @9) */
  rr=w10;
  *rr++ = w2;
  *rr++ = w3;
  *rr++ = w4;
  *rr++ = w5;
  *rr++ = w6;
  *rr++ = w7;
  *rr++ = w8;
  *rr++ = w9;
  /* #11: @11 = @10[:6] */
  for (rr=w11, ss=w10+0; ss!=w10+6; ss+=1) *rr++ = *ss;
  /* #12: @2 = input[0][0] */
  w2 = arg[0] ? arg[0][0] : 0;
  /* #13: @3 = input[0][1] */
  w3 = arg[0] ? arg[0][1] : 0;
  /* #14: @4 = input[0][2] */
  w4 = arg[0] ? arg[0][2] : 0;
  /* #15: @5 = input[0][3] */
  w5 = arg[0] ? arg[0][3] : 0;
  /* #16: @6 = input[0][4] */
  w6 = arg[0] ? arg[0][4] : 0;
  /* #17: @7 = input[0][5] */
  w7 = arg[0] ? arg[0][5] : 0;
  /* #18: @12 = vertcat(@2, @3, @4, @5, @6, @7) */
  rr=w12;
  *rr++ = w2;
  *rr++ = w3;
  *rr++ = w4;
  *rr++ = w5;
  *rr++ = w6;
  *rr++ = w7;
  /* #19: @11 = (@11-@12) */
  for (i=0, rr=w11, cs=w12; i<6; ++i) (*rr++) -= (*cs++);
  /* #20: @12 = @11' */
  casadi_copy(w11, 6, w12);
  /* #21: @13 = 
  [[4, 0, 0, 0, 0, 0], 
   [0, 4, 0, 0, 0, 0], 
   [0, 0, 0, 0, 0, 0], 
   [0, 0, 0, 0, 0, 0], 
   [0, 0, 0, 0, 0, 0], 
   [0, 0, 0, 0, 0, 0]] */
  casadi_copy(casadi_c0, 36, w13);
  /* #22: @1 = mac(@12,@13,@1) */
  for (i=0, rr=w1; i<6; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w12+j, tt=w13+i*6; k<6; ++k) *rr += ss[k*1]**tt++;
  /* #23: @0 = mac(@1,@11,@0) */
  for (i=0, rr=(&w0); i<1; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w1+j, tt=w11+i*6; k<6; ++k) *rr += ss[k*1]**tt++;
  /* #24: output[0][0] = @0 */
  if (res[0]) res[0][0] = w0;
  /* #25: @1 = @1' */
  /* #26: @12 = zeros(1x6) */
  casadi_clear(w12, 6);
  /* #27: @11 = @11' */
  /* #28: @14 = @13' */
  for (i=0, rr=w14, cs=w13; i<6; ++i) for (j=0; j<6; ++j) rr[i+j*6] = *cs++;
  /* #29: @12 = mac(@11,@14,@12) */
  for (i=0, rr=w12; i<6; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w11+j, tt=w14+i*6; k<6; ++k) *rr += ss[k*1]**tt++;
  /* #30: @12 = @12' */
  /* #31: @1 = (@1+@12) */
  for (i=0, rr=w1, cs=w12; i<6; ++i) (*rr++) += (*cs++);
  /* #32: @1 = (-@1) */
  for (i=0, rr=w1, cs=w1; i<6; ++i) *rr++ = (- *cs++ );
  /* #33: output[1][0] = @1 */
  casadi_copy(w1, 6, res[1]);
  /* #34: @15 = zeros(6x6) */
  casadi_clear(w15, 36);
  /* #35: @1 = zeros(1x6) */
  casadi_clear(w1, 6);
  /* #36: @0 = ones(6x1,1nz) */
  w0 = 1.;
  /* #37: {@2, NULL, NULL, NULL, NULL, NULL} = vertsplit(@0) */
  w2 = w0;
  /* #38: @16 = 00 */
  /* #39: @17 = 00 */
  /* #40: @18 = 00 */
  /* #41: @19 = 00 */
  /* #42: @20 = 00 */
  /* #43: @0 = vertcat(@2, @16, @17, @18, @19, @20) */
  rr=(&w0);
  *rr++ = w2;
  /* #44: @0 = (-@0) */
  w0 = (- w0 );
  /* #45: @2 = @0' */
  casadi_copy((&w0), 1, (&w2));
  /* #46: @1 = mac(@2,@13,@1) */
  casadi_mtimes((&w2), casadi_s2, w13, casadi_s1, w1, casadi_s0, w, 0);
  /* #47: @1 = @1' */
  /* #48: @12 = zeros(1x6) */
  casadi_clear(w12, 6);
  /* #49: @0 = @0' */
  /* #50: @12 = mac(@0,@14,@12) */
  casadi_mtimes((&w0), casadi_s2, w14, casadi_s1, w12, casadi_s0, w, 0);
  /* #51: @12 = @12' */
  /* #52: @1 = (@1+@12) */
  for (i=0, rr=w1, cs=w12; i<6; ++i) (*rr++) += (*cs++);
  /* #53: @1 = (-@1) */
  for (i=0, rr=w1, cs=w1; i<6; ++i) *rr++ = (- *cs++ );
  /* #54: (@15[:6] = @1) */
  for (rr=w15+0, ss=w1; rr!=w15+6; rr+=1) *rr = *ss++;
  /* #55: (@15[:36:6] = @1) */
  for (rr=w15+0, ss=w1; rr!=w15+36; rr+=6) *rr = *ss++;
  /* #56: @1 = zeros(1x6) */
  casadi_clear(w1, 6);
  /* #57: @16 = 00 */
  /* #58: @0 = ones(6x1,1nz) */
  w0 = 1.;
  /* #59: {NULL, @2, NULL, NULL, NULL, NULL} = vertsplit(@0) */
  w2 = w0;
  /* #60: @17 = 00 */
  /* #61: @18 = 00 */
  /* #62: @19 = 00 */
  /* #63: @20 = 00 */
  /* #64: @0 = vertcat(@16, @2, @17, @18, @19, @20) */
  rr=(&w0);
  *rr++ = w2;
  /* #65: @0 = (-@0) */
  w0 = (- w0 );
  /* #66: @2 = @0' */
  casadi_copy((&w0), 1, (&w2));
  /* #67: @1 = mac(@2,@13,@1) */
  casadi_mtimes((&w2), casadi_s3, w13, casadi_s1, w1, casadi_s0, w, 0);
  /* #68: @1 = @1' */
  /* #69: @12 = zeros(1x6) */
  casadi_clear(w12, 6);
  /* #70: @0 = @0' */
  /* #71: @12 = mac(@0,@14,@12) */
  casadi_mtimes((&w0), casadi_s3, w14, casadi_s1, w12, casadi_s0, w, 0);
  /* #72: @12 = @12' */
  /* #73: @1 = (@1+@12) */
  for (i=0, rr=w1, cs=w12; i<6; ++i) (*rr++) += (*cs++);
  /* #74: @1 = (-@1) */
  for (i=0, rr=w1, cs=w1; i<6; ++i) *rr++ = (- *cs++ );
  /* #75: (@15[6:12] = @1) */
  for (rr=w15+6, ss=w1; rr!=w15+12; rr+=1) *rr = *ss++;
  /* #76: (@15[1:37:6] = @1) */
  for (rr=w15+1, ss=w1; rr!=w15+37; rr+=6) *rr = *ss++;
  /* #77: @1 = zeros(1x6) */
  casadi_clear(w1, 6);
  /* #78: @16 = 00 */
  /* #79: @17 = 00 */
  /* #80: @0 = ones(6x1,1nz) */
  w0 = 1.;
  /* #81: {NULL, NULL, @2, NULL, NULL, NULL} = vertsplit(@0) */
  w2 = w0;
  /* #82: @18 = 00 */
  /* #83: @19 = 00 */
  /* #84: @20 = 00 */
  /* #85: @0 = vertcat(@16, @17, @2, @18, @19, @20) */
  rr=(&w0);
  *rr++ = w2;
  /* #86: @0 = (-@0) */
  w0 = (- w0 );
  /* #87: @2 = @0' */
  casadi_copy((&w0), 1, (&w2));
  /* #88: @1 = mac(@2,@13,@1) */
  casadi_mtimes((&w2), casadi_s4, w13, casadi_s1, w1, casadi_s0, w, 0);
  /* #89: @1 = @1' */
  /* #90: @12 = zeros(1x6) */
  casadi_clear(w12, 6);
  /* #91: @0 = @0' */
  /* #92: @12 = mac(@0,@14,@12) */
  casadi_mtimes((&w0), casadi_s4, w14, casadi_s1, w12, casadi_s0, w, 0);
  /* #93: @12 = @12' */
  /* #94: @1 = (@1+@12) */
  for (i=0, rr=w1, cs=w12; i<6; ++i) (*rr++) += (*cs++);
  /* #95: @1 = (-@1) */
  for (i=0, rr=w1, cs=w1; i<6; ++i) *rr++ = (- *cs++ );
  /* #96: (@15[12:18] = @1) */
  for (rr=w15+12, ss=w1; rr!=w15+18; rr+=1) *rr = *ss++;
  /* #97: (@15[2:38:6] = @1) */
  for (rr=w15+2, ss=w1; rr!=w15+38; rr+=6) *rr = *ss++;
  /* #98: @1 = zeros(1x6) */
  casadi_clear(w1, 6);
  /* #99: @16 = 00 */
  /* #100: @17 = 00 */
  /* #101: @18 = 00 */
  /* #102: @0 = ones(6x1,1nz) */
  w0 = 1.;
  /* #103: {NULL, NULL, NULL, @2, NULL, NULL} = vertsplit(@0) */
  w2 = w0;
  /* #104: @19 = 00 */
  /* #105: @20 = 00 */
  /* #106: @0 = vertcat(@16, @17, @18, @2, @19, @20) */
  rr=(&w0);
  *rr++ = w2;
  /* #107: @0 = (-@0) */
  w0 = (- w0 );
  /* #108: @2 = @0' */
  casadi_copy((&w0), 1, (&w2));
  /* #109: @1 = mac(@2,@13,@1) */
  casadi_mtimes((&w2), casadi_s5, w13, casadi_s1, w1, casadi_s0, w, 0);
  /* #110: @1 = @1' */
  /* #111: @12 = zeros(1x6) */
  casadi_clear(w12, 6);
  /* #112: @0 = @0' */
  /* #113: @12 = mac(@0,@14,@12) */
  casadi_mtimes((&w0), casadi_s5, w14, casadi_s1, w12, casadi_s0, w, 0);
  /* #114: @12 = @12' */
  /* #115: @1 = (@1+@12) */
  for (i=0, rr=w1, cs=w12; i<6; ++i) (*rr++) += (*cs++);
  /* #116: @1 = (-@1) */
  for (i=0, rr=w1, cs=w1; i<6; ++i) *rr++ = (- *cs++ );
  /* #117: (@15[18:24] = @1) */
  for (rr=w15+18, ss=w1; rr!=w15+24; rr+=1) *rr = *ss++;
  /* #118: (@15[3:39:6] = @1) */
  for (rr=w15+3, ss=w1; rr!=w15+39; rr+=6) *rr = *ss++;
  /* #119: @1 = zeros(1x6) */
  casadi_clear(w1, 6);
  /* #120: @16 = 00 */
  /* #121: @17 = 00 */
  /* #122: @18 = 00 */
  /* #123: @19 = 00 */
  /* #124: @0 = ones(6x1,1nz) */
  w0 = 1.;
  /* #125: {NULL, NULL, NULL, NULL, @2, NULL} = vertsplit(@0) */
  w2 = w0;
  /* #126: @20 = 00 */
  /* #127: @0 = vertcat(@16, @17, @18, @19, @2, @20) */
  rr=(&w0);
  *rr++ = w2;
  /* #128: @0 = (-@0) */
  w0 = (- w0 );
  /* #129: @2 = @0' */
  casadi_copy((&w0), 1, (&w2));
  /* #130: @1 = mac(@2,@13,@1) */
  casadi_mtimes((&w2), casadi_s6, w13, casadi_s1, w1, casadi_s0, w, 0);
  /* #131: @1 = @1' */
  /* #132: @12 = zeros(1x6) */
  casadi_clear(w12, 6);
  /* #133: @0 = @0' */
  /* #134: @12 = mac(@0,@14,@12) */
  casadi_mtimes((&w0), casadi_s6, w14, casadi_s1, w12, casadi_s0, w, 0);
  /* #135: @12 = @12' */
  /* #136: @1 = (@1+@12) */
  for (i=0, rr=w1, cs=w12; i<6; ++i) (*rr++) += (*cs++);
  /* #137: @1 = (-@1) */
  for (i=0, rr=w1, cs=w1; i<6; ++i) *rr++ = (- *cs++ );
  /* #138: (@15[24:30] = @1) */
  for (rr=w15+24, ss=w1; rr!=w15+30; rr+=1) *rr = *ss++;
  /* #139: (@15[4:40:6] = @1) */
  for (rr=w15+4, ss=w1; rr!=w15+40; rr+=6) *rr = *ss++;
  /* #140: @1 = zeros(1x6) */
  casadi_clear(w1, 6);
  /* #141: @16 = 00 */
  /* #142: @17 = 00 */
  /* #143: @18 = 00 */
  /* #144: @19 = 00 */
  /* #145: @20 = 00 */
  /* #146: @0 = ones(6x1,1nz) */
  w0 = 1.;
  /* #147: {NULL, NULL, NULL, NULL, NULL, @2} = vertsplit(@0) */
  w2 = w0;
  /* #148: @0 = vertcat(@16, @17, @18, @19, @20, @2) */
  rr=(&w0);
  *rr++ = w2;
  /* #149: @0 = (-@0) */
  w0 = (- w0 );
  /* #150: @2 = @0' */
  casadi_copy((&w0), 1, (&w2));
  /* #151: @1 = mac(@2,@13,@1) */
  casadi_mtimes((&w2), casadi_s7, w13, casadi_s1, w1, casadi_s0, w, 0);
  /* #152: @1 = @1' */
  /* #153: @12 = zeros(1x6) */
  casadi_clear(w12, 6);
  /* #154: @0 = @0' */
  /* #155: @12 = mac(@0,@14,@12) */
  casadi_mtimes((&w0), casadi_s7, w14, casadi_s1, w12, casadi_s0, w, 0);
  /* #156: @12 = @12' */
  /* #157: @1 = (@1+@12) */
  for (i=0, rr=w1, cs=w12; i<6; ++i) (*rr++) += (*cs++);
  /* #158: @1 = (-@1) */
  for (i=0, rr=w1, cs=w1; i<6; ++i) *rr++ = (- *cs++ );
  /* #159: (@15[30:36] = @1) */
  for (rr=w15+30, ss=w1; rr!=w15+36; rr+=1) *rr = *ss++;
  /* #160: (@15[5:41:6] = @1) */
  for (rr=w15+5, ss=w1; rr!=w15+41; rr+=6) *rr = *ss++;
  /* #161: @14 = @15' */
  for (i=0, rr=w14, cs=w15; i<6; ++i) for (j=0; j<6; ++j) rr[i+j*6] = *cs++;
  /* #162: output[2][0] = @14 */
  casadi_copy(w14, 36, res[2]);
  return 0;
}

CASADI_SYMBOL_EXPORT int Drone_ode_cost_ext_cost_e_fun_jac_hess(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int Drone_ode_cost_ext_cost_e_fun_jac_hess_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int Drone_ode_cost_ext_cost_e_fun_jac_hess_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void Drone_ode_cost_ext_cost_e_fun_jac_hess_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int Drone_ode_cost_ext_cost_e_fun_jac_hess_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void Drone_ode_cost_ext_cost_e_fun_jac_hess_release(int mem) {
}

CASADI_SYMBOL_EXPORT void Drone_ode_cost_ext_cost_e_fun_jac_hess_incref(void) {
}

CASADI_SYMBOL_EXPORT void Drone_ode_cost_ext_cost_e_fun_jac_hess_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int Drone_ode_cost_ext_cost_e_fun_jac_hess_n_in(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_int Drone_ode_cost_ext_cost_e_fun_jac_hess_n_out(void) { return 5;}

CASADI_SYMBOL_EXPORT casadi_real Drone_ode_cost_ext_cost_e_fun_jac_hess_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* Drone_ode_cost_ext_cost_e_fun_jac_hess_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* Drone_ode_cost_ext_cost_e_fun_jac_hess_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    case 2: return "o2";
    case 3: return "o3";
    case 4: return "o4";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* Drone_ode_cost_ext_cost_e_fun_jac_hess_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s8;
    case 1: return casadi_s9;
    case 2: return casadi_s9;
    case 3: return casadi_s10;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* Drone_ode_cost_ext_cost_e_fun_jac_hess_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s11;
    case 1: return casadi_s8;
    case 2: return casadi_s1;
    case 3: return casadi_s9;
    case 4: return casadi_s12;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int Drone_ode_cost_ext_cost_e_fun_jac_hess_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 12;
  if (sz_res) *sz_res = 11;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 144;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
