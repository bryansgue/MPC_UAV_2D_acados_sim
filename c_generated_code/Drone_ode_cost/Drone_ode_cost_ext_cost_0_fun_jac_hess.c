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
  #define CASADI_PREFIX(ID) Drone_ode_cost_ext_cost_0_fun_jac_hess_ ## ID
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
#define casadi_densify CASADI_PREFIX(densify)
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_fill CASADI_PREFIX(fill)
#define casadi_mtimes CASADI_PREFIX(mtimes)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s10 CASADI_PREFIX(s10)
#define casadi_s11 CASADI_PREFIX(s11)
#define casadi_s12 CASADI_PREFIX(s12)
#define casadi_s13 CASADI_PREFIX(s13)
#define casadi_s14 CASADI_PREFIX(s14)
#define casadi_s15 CASADI_PREFIX(s15)
#define casadi_s16 CASADI_PREFIX(s16)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_s3 CASADI_PREFIX(s3)
#define casadi_s4 CASADI_PREFIX(s4)
#define casadi_s5 CASADI_PREFIX(s5)
#define casadi_s6 CASADI_PREFIX(s6)
#define casadi_s7 CASADI_PREFIX(s7)
#define casadi_s8 CASADI_PREFIX(s8)
#define casadi_s9 CASADI_PREFIX(s9)
#define casadi_trans CASADI_PREFIX(trans)

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

#define CASADI_CAST(x,y) ((x) y)

void casadi_densify(const casadi_real* x, const casadi_int* sp_x, casadi_real* y, casadi_int tr) {
  casadi_int nrow_x, ncol_x, i, el;
  const casadi_int *colind_x, *row_x;
  if (!y) return;
  nrow_x = sp_x[0]; ncol_x = sp_x[1];
  colind_x = sp_x+2; row_x = sp_x+ncol_x+3;
  casadi_clear(y, nrow_x*ncol_x);
  if (!x) return;
  if (tr) {
    for (i=0; i<ncol_x; ++i) {
      for (el=colind_x[i]; el!=colind_x[i+1]; ++el) {
        y[i + row_x[el]*ncol_x] = CASADI_CAST(casadi_real, *x++);
      }
    }
  } else {
    for (i=0; i<ncol_x; ++i) {
      for (el=colind_x[i]; el!=colind_x[i+1]; ++el) {
        y[row_x[el]] = CASADI_CAST(casadi_real, *x++);
      }
      y += nrow_x;
    }
  }
}

void casadi_fill(casadi_real* x, casadi_int n, casadi_real alpha) {
  casadi_int i;
  if (x) {
    for (i=0; i<n; ++i) *x++ = alpha;
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

void casadi_trans(const casadi_real* x, const casadi_int* sp_x, casadi_real* y,
    const casadi_int* sp_y, casadi_int* tmp) {
  casadi_int ncol_x, nnz_x, ncol_y, k;
  const casadi_int* row_x, *colind_y;
  ncol_x = sp_x[1];
  nnz_x = sp_x[2 + ncol_x];
  row_x = sp_x + 2 + ncol_x+1;
  ncol_y = sp_y[1];
  colind_y = sp_y+2;
  for (k=0; k<ncol_y; ++k) tmp[k] = colind_y[k];
  for (k=0; k<nnz_x; ++k) {
    y[tmp[row_x[k]]++] = x[k];
  }
}

static const casadi_int casadi_s0[10] = {8, 1, 0, 6, 2, 3, 4, 5, 6, 7};
static const casadi_int casadi_s1[15] = {1, 6, 0, 1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0};
static const casadi_int casadi_s2[45] = {6, 6, 0, 6, 12, 18, 24, 30, 36, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5};
static const casadi_int casadi_s3[10] = {1, 6, 0, 1, 1, 1, 1, 1, 1, 0};
static const casadi_int casadi_s4[10] = {1, 6, 0, 0, 1, 1, 1, 1, 1, 0};
static const casadi_int casadi_s5[10] = {1, 6, 0, 0, 0, 1, 1, 1, 1, 0};
static const casadi_int casadi_s6[10] = {1, 6, 0, 0, 0, 0, 1, 1, 1, 0};
static const casadi_int casadi_s7[10] = {1, 6, 0, 0, 0, 0, 0, 1, 1, 0};
static const casadi_int casadi_s8[10] = {1, 6, 0, 0, 0, 0, 0, 0, 1, 0};
static const casadi_int casadi_s9[47] = {8, 8, 0, 0, 0, 6, 12, 18, 24, 30, 36, 2, 3, 4, 5, 6, 7, 2, 3, 4, 5, 6, 7, 2, 3, 4, 5, 6, 7, 2, 3, 4, 5, 6, 7, 2, 3, 4, 5, 6, 7, 2, 3, 4, 5, 6, 7};
static const casadi_int casadi_s10[10] = {6, 1, 0, 6, 0, 1, 2, 3, 4, 5};
static const casadi_int casadi_s11[6] = {2, 1, 0, 2, 0, 1};
static const casadi_int casadi_s12[3] = {0, 0, 0};
static const casadi_int casadi_s13[18] = {14, 1, 0, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
static const casadi_int casadi_s14[5] = {1, 1, 0, 1, 0};
static const casadi_int casadi_s15[12] = {8, 1, 0, 8, 0, 1, 2, 3, 4, 5, 6, 7};
static const casadi_int casadi_s16[11] = {0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0};

static const casadi_real casadi_c0[36] = {4., 0., 0., 0., 0., 0., 0., 4., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.};

/* Drone_ode_cost_ext_cost_0_fun_jac_hess:(i0[6],i1[2],i2[],i3[14])->(o0,o1[8],o2[8x8,36nz],o3[],o4[0x8]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_int i, j, k;
  casadi_real *rr, *ss, *tt;
  const casadi_real *cs;
  casadi_real w0, *w1=w+2, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15, *w16=w+22, *w17=w+36, *w18=w+42, *w19=w+48, *w22=w+84, *w23=w+120, *w24=w+128, *w25=w+164;
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
  /* #31: @20 = 00 */
  /* #32: @21 = 00 */
  /* #33: @1 = @1' */
  /* #34: @18 = zeros(1x6) */
  casadi_clear(w18, 6);
  /* #35: @17 = @17' */
  /* #36: @22 = @19' */
  for (i=0, rr=w22, cs=w19; i<6; ++i) for (j=0; j<6; ++j) rr[i+j*6] = *cs++;
  /* #37: @18 = mac(@17,@22,@18) */
  for (i=0, rr=w18; i<6; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w17+j, tt=w22+i*6; k<6; ++k) *rr += ss[k*1]**tt++;
  /* #38: @18 = @18' */
  /* #39: @1 = (@1+@18) */
  for (i=0, rr=w1, cs=w18; i<6; ++i) (*rr++) += (*cs++);
  /* #40: @1 = (-@1) */
  for (i=0, rr=w1, cs=w1; i<6; ++i) *rr++ = (- *cs++ );
  /* #41: {@0, @2, @3, @4, @5, @6} = vertsplit(@1) */
  w0 = w1[0];
  w2 = w1[1];
  w3 = w1[2];
  w4 = w1[3];
  w5 = w1[4];
  w6 = w1[5];
  /* #42: @1 = vertcat(@20, @21, @0, @2, @3, @4, @5, @6) */
  rr=w1;
  *rr++ = w0;
  *rr++ = w2;
  *rr++ = w3;
  *rr++ = w4;
  *rr++ = w5;
  *rr++ = w6;
  /* #43: @23 = dense(@1) */
  casadi_densify(w1, casadi_s0, w23, 0);
  /* #44: output[1][0] = @23 */
  casadi_copy(w23, 8, res[1]);
  /* #45: @24 = zeros(8x8,36nz) */
  casadi_clear(w24, 36);
  /* #46: @20 = 00 */
  /* #47: @21 = 00 */
  /* #48: @1 = zeros(1x6) */
  casadi_clear(w1, 6);
  /* #49: @25 = ones(8x1,3nz) */
  casadi_fill(w25, 3, 1.);
  /* #50: {NULL, NULL, @0, NULL, NULL, NULL, NULL, NULL} = vertsplit(@25) */
  w0 = w25[2];
  /* #51: @26 = 00 */
  /* #52: @27 = 00 */
  /* #53: @28 = 00 */
  /* #54: @29 = 00 */
  /* #55: @30 = 00 */
  /* #56: @2 = vertcat(@0, @26, @27, @28, @29, @30) */
  rr=(&w2);
  *rr++ = w0;
  /* #57: @2 = (-@2) */
  w2 = (- w2 );
  /* #58: @0 = @2' */
  casadi_copy((&w2), 1, (&w0));
  /* #59: @1 = mac(@0,@19,@1) */
  casadi_mtimes((&w0), casadi_s3, w19, casadi_s2, w1, casadi_s1, w, 0);
  /* #60: @1 = @1' */
  /* #61: @18 = zeros(1x6) */
  casadi_clear(w18, 6);
  /* #62: @2 = @2' */
  /* #63: @18 = mac(@2,@22,@18) */
  casadi_mtimes((&w2), casadi_s3, w22, casadi_s2, w18, casadi_s1, w, 0);
  /* #64: @18 = @18' */
  /* #65: @1 = (@1+@18) */
  for (i=0, rr=w1, cs=w18; i<6; ++i) (*rr++) += (*cs++);
  /* #66: @1 = (-@1) */
  for (i=0, rr=w1, cs=w1; i<6; ++i) *rr++ = (- *cs++ );
  /* #67: {@2, @0, @3, @4, @5, @6} = vertsplit(@1) */
  w2 = w1[0];
  w0 = w1[1];
  w3 = w1[2];
  w4 = w1[3];
  w5 = w1[4];
  w6 = w1[5];
  /* #68: @1 = vertcat(@20, @21, @2, @0, @3, @4, @5, @6) */
  rr=w1;
  *rr++ = w2;
  *rr++ = w0;
  *rr++ = w3;
  *rr++ = w4;
  *rr++ = w5;
  *rr++ = w6;
  /* #69: @18 = @1[:6] */
  for (rr=w18, ss=w1+0; ss!=w1+6; ss+=1) *rr++ = *ss;
  /* #70: (@24[:6] = @18) */
  for (rr=w24+0, ss=w18; rr!=w24+6; rr+=1) *rr = *ss++;
  /* #71: @18 = @1[:6] */
  for (rr=w18, ss=w1+0; ss!=w1+6; ss+=1) *rr++ = *ss;
  /* #72: (@24[:36:6] = @18) */
  for (rr=w24+0, ss=w18; rr!=w24+36; rr+=6) *rr = *ss++;
  /* #73: @20 = 00 */
  /* #74: @21 = 00 */
  /* #75: @18 = zeros(1x6) */
  casadi_clear(w18, 6);
  /* #76: @26 = 00 */
  /* #77: @2 = ones(8x1,1nz) */
  w2 = 1.;
  /* #78: {NULL, NULL, NULL, @0, NULL, NULL, NULL, NULL} = vertsplit(@2) */
  w0 = w2;
  /* #79: @27 = 00 */
  /* #80: @28 = 00 */
  /* #81: @29 = 00 */
  /* #82: @30 = 00 */
  /* #83: @2 = vertcat(@26, @0, @27, @28, @29, @30) */
  rr=(&w2);
  *rr++ = w0;
  /* #84: @2 = (-@2) */
  w2 = (- w2 );
  /* #85: @0 = @2' */
  casadi_copy((&w2), 1, (&w0));
  /* #86: @18 = mac(@0,@19,@18) */
  casadi_mtimes((&w0), casadi_s4, w19, casadi_s2, w18, casadi_s1, w, 0);
  /* #87: @18 = @18' */
  /* #88: @1 = zeros(1x6) */
  casadi_clear(w1, 6);
  /* #89: @2 = @2' */
  /* #90: @1 = mac(@2,@22,@1) */
  casadi_mtimes((&w2), casadi_s4, w22, casadi_s2, w1, casadi_s1, w, 0);
  /* #91: @1 = @1' */
  /* #92: @18 = (@18+@1) */
  for (i=0, rr=w18, cs=w1; i<6; ++i) (*rr++) += (*cs++);
  /* #93: @18 = (-@18) */
  for (i=0, rr=w18, cs=w18; i<6; ++i) *rr++ = (- *cs++ );
  /* #94: {@2, @0, @3, @4, @5, @6} = vertsplit(@18) */
  w2 = w18[0];
  w0 = w18[1];
  w3 = w18[2];
  w4 = w18[3];
  w5 = w18[4];
  w6 = w18[5];
  /* #95: @18 = vertcat(@20, @21, @2, @0, @3, @4, @5, @6) */
  rr=w18;
  *rr++ = w2;
  *rr++ = w0;
  *rr++ = w3;
  *rr++ = w4;
  *rr++ = w5;
  *rr++ = w6;
  /* #96: @1 = @18[:6] */
  for (rr=w1, ss=w18+0; ss!=w18+6; ss+=1) *rr++ = *ss;
  /* #97: (@24[6:12] = @1) */
  for (rr=w24+6, ss=w1; rr!=w24+12; rr+=1) *rr = *ss++;
  /* #98: @1 = @18[:6] */
  for (rr=w1, ss=w18+0; ss!=w18+6; ss+=1) *rr++ = *ss;
  /* #99: (@24[1:37:6] = @1) */
  for (rr=w24+1, ss=w1; rr!=w24+37; rr+=6) *rr = *ss++;
  /* #100: @20 = 00 */
  /* #101: @21 = 00 */
  /* #102: @1 = zeros(1x6) */
  casadi_clear(w1, 6);
  /* #103: @26 = 00 */
  /* #104: @27 = 00 */
  /* #105: @2 = ones(8x1,1nz) */
  w2 = 1.;
  /* #106: {NULL, NULL, NULL, NULL, @0, NULL, NULL, NULL} = vertsplit(@2) */
  w0 = w2;
  /* #107: @28 = 00 */
  /* #108: @29 = 00 */
  /* #109: @30 = 00 */
  /* #110: @2 = vertcat(@26, @27, @0, @28, @29, @30) */
  rr=(&w2);
  *rr++ = w0;
  /* #111: @2 = (-@2) */
  w2 = (- w2 );
  /* #112: @0 = @2' */
  casadi_copy((&w2), 1, (&w0));
  /* #113: @1 = mac(@0,@19,@1) */
  casadi_mtimes((&w0), casadi_s5, w19, casadi_s2, w1, casadi_s1, w, 0);
  /* #114: @1 = @1' */
  /* #115: @18 = zeros(1x6) */
  casadi_clear(w18, 6);
  /* #116: @2 = @2' */
  /* #117: @18 = mac(@2,@22,@18) */
  casadi_mtimes((&w2), casadi_s5, w22, casadi_s2, w18, casadi_s1, w, 0);
  /* #118: @18 = @18' */
  /* #119: @1 = (@1+@18) */
  for (i=0, rr=w1, cs=w18; i<6; ++i) (*rr++) += (*cs++);
  /* #120: @1 = (-@1) */
  for (i=0, rr=w1, cs=w1; i<6; ++i) *rr++ = (- *cs++ );
  /* #121: {@2, @0, @3, @4, @5, @6} = vertsplit(@1) */
  w2 = w1[0];
  w0 = w1[1];
  w3 = w1[2];
  w4 = w1[3];
  w5 = w1[4];
  w6 = w1[5];
  /* #122: @1 = vertcat(@20, @21, @2, @0, @3, @4, @5, @6) */
  rr=w1;
  *rr++ = w2;
  *rr++ = w0;
  *rr++ = w3;
  *rr++ = w4;
  *rr++ = w5;
  *rr++ = w6;
  /* #123: @18 = @1[:6] */
  for (rr=w18, ss=w1+0; ss!=w1+6; ss+=1) *rr++ = *ss;
  /* #124: (@24[12:18] = @18) */
  for (rr=w24+12, ss=w18; rr!=w24+18; rr+=1) *rr = *ss++;
  /* #125: @18 = @1[:6] */
  for (rr=w18, ss=w1+0; ss!=w1+6; ss+=1) *rr++ = *ss;
  /* #126: (@24[2:38:6] = @18) */
  for (rr=w24+2, ss=w18; rr!=w24+38; rr+=6) *rr = *ss++;
  /* #127: @20 = 00 */
  /* #128: @21 = 00 */
  /* #129: @18 = zeros(1x6) */
  casadi_clear(w18, 6);
  /* #130: @26 = 00 */
  /* #131: @27 = 00 */
  /* #132: @28 = 00 */
  /* #133: @2 = ones(8x1,1nz) */
  w2 = 1.;
  /* #134: {NULL, NULL, NULL, NULL, NULL, @0, NULL, NULL} = vertsplit(@2) */
  w0 = w2;
  /* #135: @29 = 00 */
  /* #136: @30 = 00 */
  /* #137: @2 = vertcat(@26, @27, @28, @0, @29, @30) */
  rr=(&w2);
  *rr++ = w0;
  /* #138: @2 = (-@2) */
  w2 = (- w2 );
  /* #139: @0 = @2' */
  casadi_copy((&w2), 1, (&w0));
  /* #140: @18 = mac(@0,@19,@18) */
  casadi_mtimes((&w0), casadi_s6, w19, casadi_s2, w18, casadi_s1, w, 0);
  /* #141: @18 = @18' */
  /* #142: @1 = zeros(1x6) */
  casadi_clear(w1, 6);
  /* #143: @2 = @2' */
  /* #144: @1 = mac(@2,@22,@1) */
  casadi_mtimes((&w2), casadi_s6, w22, casadi_s2, w1, casadi_s1, w, 0);
  /* #145: @1 = @1' */
  /* #146: @18 = (@18+@1) */
  for (i=0, rr=w18, cs=w1; i<6; ++i) (*rr++) += (*cs++);
  /* #147: @18 = (-@18) */
  for (i=0, rr=w18, cs=w18; i<6; ++i) *rr++ = (- *cs++ );
  /* #148: {@2, @0, @3, @4, @5, @6} = vertsplit(@18) */
  w2 = w18[0];
  w0 = w18[1];
  w3 = w18[2];
  w4 = w18[3];
  w5 = w18[4];
  w6 = w18[5];
  /* #149: @18 = vertcat(@20, @21, @2, @0, @3, @4, @5, @6) */
  rr=w18;
  *rr++ = w2;
  *rr++ = w0;
  *rr++ = w3;
  *rr++ = w4;
  *rr++ = w5;
  *rr++ = w6;
  /* #150: @1 = @18[:6] */
  for (rr=w1, ss=w18+0; ss!=w18+6; ss+=1) *rr++ = *ss;
  /* #151: (@24[18:24] = @1) */
  for (rr=w24+18, ss=w1; rr!=w24+24; rr+=1) *rr = *ss++;
  /* #152: @1 = @18[:6] */
  for (rr=w1, ss=w18+0; ss!=w18+6; ss+=1) *rr++ = *ss;
  /* #153: (@24[3:39:6] = @1) */
  for (rr=w24+3, ss=w1; rr!=w24+39; rr+=6) *rr = *ss++;
  /* #154: @20 = 00 */
  /* #155: @21 = 00 */
  /* #156: @1 = zeros(1x6) */
  casadi_clear(w1, 6);
  /* #157: @26 = 00 */
  /* #158: @27 = 00 */
  /* #159: @28 = 00 */
  /* #160: @29 = 00 */
  /* #161: @2 = ones(8x1,1nz) */
  w2 = 1.;
  /* #162: {NULL, NULL, NULL, NULL, NULL, NULL, @0, NULL} = vertsplit(@2) */
  w0 = w2;
  /* #163: @30 = 00 */
  /* #164: @2 = vertcat(@26, @27, @28, @29, @0, @30) */
  rr=(&w2);
  *rr++ = w0;
  /* #165: @2 = (-@2) */
  w2 = (- w2 );
  /* #166: @0 = @2' */
  casadi_copy((&w2), 1, (&w0));
  /* #167: @1 = mac(@0,@19,@1) */
  casadi_mtimes((&w0), casadi_s7, w19, casadi_s2, w1, casadi_s1, w, 0);
  /* #168: @1 = @1' */
  /* #169: @18 = zeros(1x6) */
  casadi_clear(w18, 6);
  /* #170: @2 = @2' */
  /* #171: @18 = mac(@2,@22,@18) */
  casadi_mtimes((&w2), casadi_s7, w22, casadi_s2, w18, casadi_s1, w, 0);
  /* #172: @18 = @18' */
  /* #173: @1 = (@1+@18) */
  for (i=0, rr=w1, cs=w18; i<6; ++i) (*rr++) += (*cs++);
  /* #174: @1 = (-@1) */
  for (i=0, rr=w1, cs=w1; i<6; ++i) *rr++ = (- *cs++ );
  /* #175: {@2, @0, @3, @4, @5, @6} = vertsplit(@1) */
  w2 = w1[0];
  w0 = w1[1];
  w3 = w1[2];
  w4 = w1[3];
  w5 = w1[4];
  w6 = w1[5];
  /* #176: @1 = vertcat(@20, @21, @2, @0, @3, @4, @5, @6) */
  rr=w1;
  *rr++ = w2;
  *rr++ = w0;
  *rr++ = w3;
  *rr++ = w4;
  *rr++ = w5;
  *rr++ = w6;
  /* #177: @18 = @1[:6] */
  for (rr=w18, ss=w1+0; ss!=w1+6; ss+=1) *rr++ = *ss;
  /* #178: (@24[24:30] = @18) */
  for (rr=w24+24, ss=w18; rr!=w24+30; rr+=1) *rr = *ss++;
  /* #179: @18 = @1[:6] */
  for (rr=w18, ss=w1+0; ss!=w1+6; ss+=1) *rr++ = *ss;
  /* #180: (@24[4:40:6] = @18) */
  for (rr=w24+4, ss=w18; rr!=w24+40; rr+=6) *rr = *ss++;
  /* #181: @20 = 00 */
  /* #182: @21 = 00 */
  /* #183: @18 = zeros(1x6) */
  casadi_clear(w18, 6);
  /* #184: @26 = 00 */
  /* #185: @27 = 00 */
  /* #186: @28 = 00 */
  /* #187: @29 = 00 */
  /* #188: @30 = 00 */
  /* #189: @2 = ones(8x1,1nz) */
  w2 = 1.;
  /* #190: {NULL, NULL, NULL, NULL, NULL, NULL, NULL, @0} = vertsplit(@2) */
  w0 = w2;
  /* #191: @2 = vertcat(@26, @27, @28, @29, @30, @0) */
  rr=(&w2);
  *rr++ = w0;
  /* #192: @2 = (-@2) */
  w2 = (- w2 );
  /* #193: @0 = @2' */
  casadi_copy((&w2), 1, (&w0));
  /* #194: @18 = mac(@0,@19,@18) */
  casadi_mtimes((&w0), casadi_s8, w19, casadi_s2, w18, casadi_s1, w, 0);
  /* #195: @18 = @18' */
  /* #196: @1 = zeros(1x6) */
  casadi_clear(w1, 6);
  /* #197: @2 = @2' */
  /* #198: @1 = mac(@2,@22,@1) */
  casadi_mtimes((&w2), casadi_s8, w22, casadi_s2, w1, casadi_s1, w, 0);
  /* #199: @1 = @1' */
  /* #200: @18 = (@18+@1) */
  for (i=0, rr=w18, cs=w1; i<6; ++i) (*rr++) += (*cs++);
  /* #201: @18 = (-@18) */
  for (i=0, rr=w18, cs=w18; i<6; ++i) *rr++ = (- *cs++ );
  /* #202: {@2, @0, @3, @4, @5, @6} = vertsplit(@18) */
  w2 = w18[0];
  w0 = w18[1];
  w3 = w18[2];
  w4 = w18[3];
  w5 = w18[4];
  w6 = w18[5];
  /* #203: @18 = vertcat(@20, @21, @2, @0, @3, @4, @5, @6) */
  rr=w18;
  *rr++ = w2;
  *rr++ = w0;
  *rr++ = w3;
  *rr++ = w4;
  *rr++ = w5;
  *rr++ = w6;
  /* #204: @1 = @18[:6] */
  for (rr=w1, ss=w18+0; ss!=w18+6; ss+=1) *rr++ = *ss;
  /* #205: (@24[30:36] = @1) */
  for (rr=w24+30, ss=w1; rr!=w24+36; rr+=1) *rr = *ss++;
  /* #206: @1 = @18[:6] */
  for (rr=w1, ss=w18+0; ss!=w18+6; ss+=1) *rr++ = *ss;
  /* #207: (@24[5:41:6] = @1) */
  for (rr=w24+5, ss=w1; rr!=w24+41; rr+=6) *rr = *ss++;
  /* #208: @22 = @24' */
  casadi_trans(w24,casadi_s9, w22, casadi_s9, iw);
  /* #209: output[2][0] = @22 */
  casadi_copy(w22, 36, res[2]);
  return 0;
}

CASADI_SYMBOL_EXPORT int Drone_ode_cost_ext_cost_0_fun_jac_hess(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int Drone_ode_cost_ext_cost_0_fun_jac_hess_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int Drone_ode_cost_ext_cost_0_fun_jac_hess_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void Drone_ode_cost_ext_cost_0_fun_jac_hess_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int Drone_ode_cost_ext_cost_0_fun_jac_hess_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void Drone_ode_cost_ext_cost_0_fun_jac_hess_release(int mem) {
}

CASADI_SYMBOL_EXPORT void Drone_ode_cost_ext_cost_0_fun_jac_hess_incref(void) {
}

CASADI_SYMBOL_EXPORT void Drone_ode_cost_ext_cost_0_fun_jac_hess_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int Drone_ode_cost_ext_cost_0_fun_jac_hess_n_in(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_int Drone_ode_cost_ext_cost_0_fun_jac_hess_n_out(void) { return 5;}

CASADI_SYMBOL_EXPORT casadi_real Drone_ode_cost_ext_cost_0_fun_jac_hess_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* Drone_ode_cost_ext_cost_0_fun_jac_hess_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* Drone_ode_cost_ext_cost_0_fun_jac_hess_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    case 2: return "o2";
    case 3: return "o3";
    case 4: return "o4";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* Drone_ode_cost_ext_cost_0_fun_jac_hess_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s10;
    case 1: return casadi_s11;
    case 2: return casadi_s12;
    case 3: return casadi_s13;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* Drone_ode_cost_ext_cost_0_fun_jac_hess_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s14;
    case 1: return casadi_s15;
    case 2: return casadi_s9;
    case 3: return casadi_s12;
    case 4: return casadi_s16;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int Drone_ode_cost_ext_cost_0_fun_jac_hess_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 18;
  if (sz_res) *sz_res = 13;
  if (sz_iw) *sz_iw = 9;
  if (sz_w) *sz_w = 167;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
