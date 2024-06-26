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
static const casadi_int casadi_s10[18] = {14, 1, 0, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
static const casadi_int casadi_s11[5] = {1, 1, 0, 1, 0};
static const casadi_int casadi_s12[9] = {0, 6, 0, 0, 0, 0, 0, 0, 0};

static const casadi_real casadi_c0[36] = {4., 0., 0., 0., 0., 0., 0., 4., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.};

/* Drone_ode_cost_ext_cost_e_fun_jac_hess:(i0[6],i1[],i2[],i3[14])->(o0,o1[6],o2[6x6],o3[],o4[0x6]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_int i, j, k;
  casadi_real *rr, *ss, *tt;
  const casadi_real *cs;
  casadi_real w0, *w1=w+2, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15, *w16=w+22, *w17=w+36, *w18=w+42, *w19=w+48, *w20=w+84, *w21=w+120;
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
  /* #10: @10 = input[3][8] */
  w10 = arg[3] ? arg[3][8] : 0;
  /* #11: @11 = input[3][9] */
  w11 = arg[3] ? arg[3][9] : 0;
  /* #12: @12 = input[3][10] */
  w12 = arg[3] ? arg[3][10] : 0;
  /* #13: @13 = input[3][11] */
  w13 = arg[3] ? arg[3][11] : 0;
  /* #14: @14 = input[3][12] */
  w14 = arg[3] ? arg[3][12] : 0;
  /* #15: @15 = input[3][13] */
  w15 = arg[3] ? arg[3][13] : 0;
  /* #16: @16 = vertcat(@2, @3, @4, @5, @6, @7, @8, @9, @10, @11, @12, @13, @14, @15) */
  rr=w16;
  *rr++ = w2;
  *rr++ = w3;
  *rr++ = w4;
  *rr++ = w5;
  *rr++ = w6;
  *rr++ = w7;
  *rr++ = w8;
  *rr++ = w9;
  *rr++ = w10;
  *rr++ = w11;
  *rr++ = w12;
  *rr++ = w13;
  *rr++ = w14;
  *rr++ = w15;
  /* #17: @17 = @16[:6] */
  for (rr=w17, ss=w16+0; ss!=w16+6; ss+=1) *rr++ = *ss;
  /* #18: @2 = input[0][0] */
  w2 = arg[0] ? arg[0][0] : 0;
  /* #19: @3 = input[0][1] */
  w3 = arg[0] ? arg[0][1] : 0;
  /* #20: @4 = input[0][2] */
  w4 = arg[0] ? arg[0][2] : 0;
  /* #21: @5 = input[0][3] */
  w5 = arg[0] ? arg[0][3] : 0;
  /* #22: @6 = input[0][4] */
  w6 = arg[0] ? arg[0][4] : 0;
  /* #23: @7 = input[0][5] */
  w7 = arg[0] ? arg[0][5] : 0;
  /* #24: @18 = vertcat(@2, @3, @4, @5, @6, @7) */
  rr=w18;
  *rr++ = w2;
  *rr++ = w3;
  *rr++ = w4;
  *rr++ = w5;
  *rr++ = w6;
  *rr++ = w7;
  /* #25: @17 = (@17-@18) */
  for (i=0, rr=w17, cs=w18; i<6; ++i) (*rr++) -= (*cs++);
  /* #26: @18 = @17' */
  casadi_copy(w17, 6, w18);
  /* #27: @19 = 
  [[4, 0, 0, 0, 0, 0], 
   [0, 4, 0, 0, 0, 0], 
   [0, 0, 0, 0, 0, 0], 
   [0, 0, 0, 0, 0, 0], 
   [0, 0, 0, 0, 0, 0], 
   [0, 0, 0, 0, 0, 0]] */
  casadi_copy(casadi_c0, 36, w19);
  /* #28: @1 = mac(@18,@19,@1) */
  for (i=0, rr=w1; i<6; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w18+j, tt=w19+i*6; k<6; ++k) *rr += ss[k*1]**tt++;
  /* #29: @0 = mac(@1,@17,@0) */
  for (i=0, rr=(&w0); i<1; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w1+j, tt=w17+i*6; k<6; ++k) *rr += ss[k*1]**tt++;
  /* #30: output[0][0] = @0 */
  if (res[0]) res[0][0] = w0;
  /* #31: @1 = @1' */
  /* #32: @18 = zeros(1x6) */
  casadi_clear(w18, 6);
  /* #33: @17 = @17' */
  /* #34: @20 = @19' */
  for (i=0, rr=w20, cs=w19; i<6; ++i) for (j=0; j<6; ++j) rr[i+j*6] = *cs++;
  /* #35: @18 = mac(@17,@20,@18) */
  for (i=0, rr=w18; i<6; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w17+j, tt=w20+i*6; k<6; ++k) *rr += ss[k*1]**tt++;
  /* #36: @18 = @18' */
  /* #37: @1 = (@1+@18) */
  for (i=0, rr=w1, cs=w18; i<6; ++i) (*rr++) += (*cs++);
  /* #38: @1 = (-@1) */
  for (i=0, rr=w1, cs=w1; i<6; ++i) *rr++ = (- *cs++ );
  /* #39: output[1][0] = @1 */
  casadi_copy(w1, 6, res[1]);
  /* #40: @21 = zeros(6x6) */
  casadi_clear(w21, 36);
  /* #41: @1 = zeros(1x6) */
  casadi_clear(w1, 6);
  /* #42: @0 = ones(6x1,1nz) */
  w0 = 1.;
  /* #43: {@2, NULL, NULL, NULL, NULL, NULL} = vertsplit(@0) */
  w2 = w0;
  /* #44: @22 = 00 */
  /* #45: @23 = 00 */
  /* #46: @24 = 00 */
  /* #47: @25 = 00 */
  /* #48: @26 = 00 */
  /* #49: @0 = vertcat(@2, @22, @23, @24, @25, @26) */
  rr=(&w0);
  *rr++ = w2;
  /* #50: @0 = (-@0) */
  w0 = (- w0 );
  /* #51: @2 = @0' */
  casadi_copy((&w0), 1, (&w2));
  /* #52: @1 = mac(@2,@19,@1) */
  casadi_mtimes((&w2), casadi_s2, w19, casadi_s1, w1, casadi_s0, w, 0);
  /* #53: @1 = @1' */
  /* #54: @18 = zeros(1x6) */
  casadi_clear(w18, 6);
  /* #55: @0 = @0' */
  /* #56: @18 = mac(@0,@20,@18) */
  casadi_mtimes((&w0), casadi_s2, w20, casadi_s1, w18, casadi_s0, w, 0);
  /* #57: @18 = @18' */
  /* #58: @1 = (@1+@18) */
  for (i=0, rr=w1, cs=w18; i<6; ++i) (*rr++) += (*cs++);
  /* #59: @1 = (-@1) */
  for (i=0, rr=w1, cs=w1; i<6; ++i) *rr++ = (- *cs++ );
  /* #60: (@21[:6] = @1) */
  for (rr=w21+0, ss=w1; rr!=w21+6; rr+=1) *rr = *ss++;
  /* #61: (@21[:36:6] = @1) */
  for (rr=w21+0, ss=w1; rr!=w21+36; rr+=6) *rr = *ss++;
  /* #62: @1 = zeros(1x6) */
  casadi_clear(w1, 6);
  /* #63: @22 = 00 */
  /* #64: @0 = ones(6x1,1nz) */
  w0 = 1.;
  /* #65: {NULL, @2, NULL, NULL, NULL, NULL} = vertsplit(@0) */
  w2 = w0;
  /* #66: @23 = 00 */
  /* #67: @24 = 00 */
  /* #68: @25 = 00 */
  /* #69: @26 = 00 */
  /* #70: @0 = vertcat(@22, @2, @23, @24, @25, @26) */
  rr=(&w0);
  *rr++ = w2;
  /* #71: @0 = (-@0) */
  w0 = (- w0 );
  /* #72: @2 = @0' */
  casadi_copy((&w0), 1, (&w2));
  /* #73: @1 = mac(@2,@19,@1) */
  casadi_mtimes((&w2), casadi_s3, w19, casadi_s1, w1, casadi_s0, w, 0);
  /* #74: @1 = @1' */
  /* #75: @18 = zeros(1x6) */
  casadi_clear(w18, 6);
  /* #76: @0 = @0' */
  /* #77: @18 = mac(@0,@20,@18) */
  casadi_mtimes((&w0), casadi_s3, w20, casadi_s1, w18, casadi_s0, w, 0);
  /* #78: @18 = @18' */
  /* #79: @1 = (@1+@18) */
  for (i=0, rr=w1, cs=w18; i<6; ++i) (*rr++) += (*cs++);
  /* #80: @1 = (-@1) */
  for (i=0, rr=w1, cs=w1; i<6; ++i) *rr++ = (- *cs++ );
  /* #81: (@21[6:12] = @1) */
  for (rr=w21+6, ss=w1; rr!=w21+12; rr+=1) *rr = *ss++;
  /* #82: (@21[1:37:6] = @1) */
  for (rr=w21+1, ss=w1; rr!=w21+37; rr+=6) *rr = *ss++;
  /* #83: @1 = zeros(1x6) */
  casadi_clear(w1, 6);
  /* #84: @22 = 00 */
  /* #85: @23 = 00 */
  /* #86: @0 = ones(6x1,1nz) */
  w0 = 1.;
  /* #87: {NULL, NULL, @2, NULL, NULL, NULL} = vertsplit(@0) */
  w2 = w0;
  /* #88: @24 = 00 */
  /* #89: @25 = 00 */
  /* #90: @26 = 00 */
  /* #91: @0 = vertcat(@22, @23, @2, @24, @25, @26) */
  rr=(&w0);
  *rr++ = w2;
  /* #92: @0 = (-@0) */
  w0 = (- w0 );
  /* #93: @2 = @0' */
  casadi_copy((&w0), 1, (&w2));
  /* #94: @1 = mac(@2,@19,@1) */
  casadi_mtimes((&w2), casadi_s4, w19, casadi_s1, w1, casadi_s0, w, 0);
  /* #95: @1 = @1' */
  /* #96: @18 = zeros(1x6) */
  casadi_clear(w18, 6);
  /* #97: @0 = @0' */
  /* #98: @18 = mac(@0,@20,@18) */
  casadi_mtimes((&w0), casadi_s4, w20, casadi_s1, w18, casadi_s0, w, 0);
  /* #99: @18 = @18' */
  /* #100: @1 = (@1+@18) */
  for (i=0, rr=w1, cs=w18; i<6; ++i) (*rr++) += (*cs++);
  /* #101: @1 = (-@1) */
  for (i=0, rr=w1, cs=w1; i<6; ++i) *rr++ = (- *cs++ );
  /* #102: (@21[12:18] = @1) */
  for (rr=w21+12, ss=w1; rr!=w21+18; rr+=1) *rr = *ss++;
  /* #103: (@21[2:38:6] = @1) */
  for (rr=w21+2, ss=w1; rr!=w21+38; rr+=6) *rr = *ss++;
  /* #104: @1 = zeros(1x6) */
  casadi_clear(w1, 6);
  /* #105: @22 = 00 */
  /* #106: @23 = 00 */
  /* #107: @24 = 00 */
  /* #108: @0 = ones(6x1,1nz) */
  w0 = 1.;
  /* #109: {NULL, NULL, NULL, @2, NULL, NULL} = vertsplit(@0) */
  w2 = w0;
  /* #110: @25 = 00 */
  /* #111: @26 = 00 */
  /* #112: @0 = vertcat(@22, @23, @24, @2, @25, @26) */
  rr=(&w0);
  *rr++ = w2;
  /* #113: @0 = (-@0) */
  w0 = (- w0 );
  /* #114: @2 = @0' */
  casadi_copy((&w0), 1, (&w2));
  /* #115: @1 = mac(@2,@19,@1) */
  casadi_mtimes((&w2), casadi_s5, w19, casadi_s1, w1, casadi_s0, w, 0);
  /* #116: @1 = @1' */
  /* #117: @18 = zeros(1x6) */
  casadi_clear(w18, 6);
  /* #118: @0 = @0' */
  /* #119: @18 = mac(@0,@20,@18) */
  casadi_mtimes((&w0), casadi_s5, w20, casadi_s1, w18, casadi_s0, w, 0);
  /* #120: @18 = @18' */
  /* #121: @1 = (@1+@18) */
  for (i=0, rr=w1, cs=w18; i<6; ++i) (*rr++) += (*cs++);
  /* #122: @1 = (-@1) */
  for (i=0, rr=w1, cs=w1; i<6; ++i) *rr++ = (- *cs++ );
  /* #123: (@21[18:24] = @1) */
  for (rr=w21+18, ss=w1; rr!=w21+24; rr+=1) *rr = *ss++;
  /* #124: (@21[3:39:6] = @1) */
  for (rr=w21+3, ss=w1; rr!=w21+39; rr+=6) *rr = *ss++;
  /* #125: @1 = zeros(1x6) */
  casadi_clear(w1, 6);
  /* #126: @22 = 00 */
  /* #127: @23 = 00 */
  /* #128: @24 = 00 */
  /* #129: @25 = 00 */
  /* #130: @0 = ones(6x1,1nz) */
  w0 = 1.;
  /* #131: {NULL, NULL, NULL, NULL, @2, NULL} = vertsplit(@0) */
  w2 = w0;
  /* #132: @26 = 00 */
  /* #133: @0 = vertcat(@22, @23, @24, @25, @2, @26) */
  rr=(&w0);
  *rr++ = w2;
  /* #134: @0 = (-@0) */
  w0 = (- w0 );
  /* #135: @2 = @0' */
  casadi_copy((&w0), 1, (&w2));
  /* #136: @1 = mac(@2,@19,@1) */
  casadi_mtimes((&w2), casadi_s6, w19, casadi_s1, w1, casadi_s0, w, 0);
  /* #137: @1 = @1' */
  /* #138: @18 = zeros(1x6) */
  casadi_clear(w18, 6);
  /* #139: @0 = @0' */
  /* #140: @18 = mac(@0,@20,@18) */
  casadi_mtimes((&w0), casadi_s6, w20, casadi_s1, w18, casadi_s0, w, 0);
  /* #141: @18 = @18' */
  /* #142: @1 = (@1+@18) */
  for (i=0, rr=w1, cs=w18; i<6; ++i) (*rr++) += (*cs++);
  /* #143: @1 = (-@1) */
  for (i=0, rr=w1, cs=w1; i<6; ++i) *rr++ = (- *cs++ );
  /* #144: (@21[24:30] = @1) */
  for (rr=w21+24, ss=w1; rr!=w21+30; rr+=1) *rr = *ss++;
  /* #145: (@21[4:40:6] = @1) */
  for (rr=w21+4, ss=w1; rr!=w21+40; rr+=6) *rr = *ss++;
  /* #146: @1 = zeros(1x6) */
  casadi_clear(w1, 6);
  /* #147: @22 = 00 */
  /* #148: @23 = 00 */
  /* #149: @24 = 00 */
  /* #150: @25 = 00 */
  /* #151: @26 = 00 */
  /* #152: @0 = ones(6x1,1nz) */
  w0 = 1.;
  /* #153: {NULL, NULL, NULL, NULL, NULL, @2} = vertsplit(@0) */
  w2 = w0;
  /* #154: @0 = vertcat(@22, @23, @24, @25, @26, @2) */
  rr=(&w0);
  *rr++ = w2;
  /* #155: @0 = (-@0) */
  w0 = (- w0 );
  /* #156: @2 = @0' */
  casadi_copy((&w0), 1, (&w2));
  /* #157: @1 = mac(@2,@19,@1) */
  casadi_mtimes((&w2), casadi_s7, w19, casadi_s1, w1, casadi_s0, w, 0);
  /* #158: @1 = @1' */
  /* #159: @18 = zeros(1x6) */
  casadi_clear(w18, 6);
  /* #160: @0 = @0' */
  /* #161: @18 = mac(@0,@20,@18) */
  casadi_mtimes((&w0), casadi_s7, w20, casadi_s1, w18, casadi_s0, w, 0);
  /* #162: @18 = @18' */
  /* #163: @1 = (@1+@18) */
  for (i=0, rr=w1, cs=w18; i<6; ++i) (*rr++) += (*cs++);
  /* #164: @1 = (-@1) */
  for (i=0, rr=w1, cs=w1; i<6; ++i) *rr++ = (- *cs++ );
  /* #165: (@21[30:36] = @1) */
  for (rr=w21+30, ss=w1; rr!=w21+36; rr+=1) *rr = *ss++;
  /* #166: (@21[5:41:6] = @1) */
  for (rr=w21+5, ss=w1; rr!=w21+41; rr+=6) *rr = *ss++;
  /* #167: @20 = @21' */
  for (i=0, rr=w20, cs=w21; i<6; ++i) for (j=0; j<6; ++j) rr[i+j*6] = *cs++;
  /* #168: output[2][0] = @20 */
  casadi_copy(w20, 36, res[2]);
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
  if (sz_arg) *sz_arg = 18;
  if (sz_res) *sz_res = 11;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 156;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
