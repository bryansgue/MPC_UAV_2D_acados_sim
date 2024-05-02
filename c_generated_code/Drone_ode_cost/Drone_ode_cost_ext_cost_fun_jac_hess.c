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
  #define CASADI_PREFIX(ID) Drone_ode_cost_ext_cost_fun_jac_hess_ ## ID
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
static const casadi_int casadi_s13[12] = {8, 1, 0, 8, 0, 1, 2, 3, 4, 5, 6, 7};
static const casadi_int casadi_s14[5] = {1, 1, 0, 1, 0};
static const casadi_int casadi_s15[11] = {0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0};

static const casadi_real casadi_c0[36] = {4., 0., 0., 0., 0., 0., 0., 4., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.};

/* Drone_ode_cost_ext_cost_fun_jac_hess:(i0[6],i1[2],i2[],i3[8])->(o0,o1[8],o2[8x8,36nz],o3[],o4[0x8]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_int i, j, k;
  casadi_real *rr, *ss, *tt;
  const casadi_real *cs;
  casadi_real w0, *w1=w+2, w2, w3, w4, w5, w6, w7, w8, w9, *w10=w+16, *w11=w+24, *w12=w+30, *w13=w+36, *w16=w+72, *w17=w+108, *w18=w+144;
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
  /* #25: @14 = 00 */
  /* #26: @15 = 00 */
  /* #27: @1 = @1' */
  /* #28: @12 = zeros(1x6) */
  casadi_clear(w12, 6);
  /* #29: @11 = @11' */
  /* #30: @16 = @13' */
  for (i=0, rr=w16, cs=w13; i<6; ++i) for (j=0; j<6; ++j) rr[i+j*6] = *cs++;
  /* #31: @12 = mac(@11,@16,@12) */
  for (i=0, rr=w12; i<6; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w11+j, tt=w16+i*6; k<6; ++k) *rr += ss[k*1]**tt++;
  /* #32: @12 = @12' */
  /* #33: @1 = (@1+@12) */
  for (i=0, rr=w1, cs=w12; i<6; ++i) (*rr++) += (*cs++);
  /* #34: @1 = (-@1) */
  for (i=0, rr=w1, cs=w1; i<6; ++i) *rr++ = (- *cs++ );
  /* #35: {@0, @2, @3, @4, @5, @6} = vertsplit(@1) */
  w0 = w1[0];
  w2 = w1[1];
  w3 = w1[2];
  w4 = w1[3];
  w5 = w1[4];
  w6 = w1[5];
  /* #36: @1 = vertcat(@14, @15, @0, @2, @3, @4, @5, @6) */
  rr=w1;
  *rr++ = w0;
  *rr++ = w2;
  *rr++ = w3;
  *rr++ = w4;
  *rr++ = w5;
  *rr++ = w6;
  /* #37: @10 = dense(@1) */
  casadi_densify(w1, casadi_s0, w10, 0);
  /* #38: output[1][0] = @10 */
  casadi_copy(w10, 8, res[1]);
  /* #39: @17 = zeros(8x8,36nz) */
  casadi_clear(w17, 36);
  /* #40: @14 = 00 */
  /* #41: @15 = 00 */
  /* #42: @1 = zeros(1x6) */
  casadi_clear(w1, 6);
  /* #43: @18 = ones(8x1,3nz) */
  casadi_fill(w18, 3, 1.);
  /* #44: {NULL, NULL, @0, NULL, NULL, NULL, NULL, NULL} = vertsplit(@18) */
  w0 = w18[2];
  /* #45: @19 = 00 */
  /* #46: @20 = 00 */
  /* #47: @21 = 00 */
  /* #48: @22 = 00 */
  /* #49: @23 = 00 */
  /* #50: @2 = vertcat(@0, @19, @20, @21, @22, @23) */
  rr=(&w2);
  *rr++ = w0;
  /* #51: @2 = (-@2) */
  w2 = (- w2 );
  /* #52: @0 = @2' */
  casadi_copy((&w2), 1, (&w0));
  /* #53: @1 = mac(@0,@13,@1) */
  casadi_mtimes((&w0), casadi_s3, w13, casadi_s2, w1, casadi_s1, w, 0);
  /* #54: @1 = @1' */
  /* #55: @12 = zeros(1x6) */
  casadi_clear(w12, 6);
  /* #56: @2 = @2' */
  /* #57: @12 = mac(@2,@16,@12) */
  casadi_mtimes((&w2), casadi_s3, w16, casadi_s2, w12, casadi_s1, w, 0);
  /* #58: @12 = @12' */
  /* #59: @1 = (@1+@12) */
  for (i=0, rr=w1, cs=w12; i<6; ++i) (*rr++) += (*cs++);
  /* #60: @1 = (-@1) */
  for (i=0, rr=w1, cs=w1; i<6; ++i) *rr++ = (- *cs++ );
  /* #61: {@2, @0, @3, @4, @5, @6} = vertsplit(@1) */
  w2 = w1[0];
  w0 = w1[1];
  w3 = w1[2];
  w4 = w1[3];
  w5 = w1[4];
  w6 = w1[5];
  /* #62: @1 = vertcat(@14, @15, @2, @0, @3, @4, @5, @6) */
  rr=w1;
  *rr++ = w2;
  *rr++ = w0;
  *rr++ = w3;
  *rr++ = w4;
  *rr++ = w5;
  *rr++ = w6;
  /* #63: @12 = @1[:6] */
  for (rr=w12, ss=w1+0; ss!=w1+6; ss+=1) *rr++ = *ss;
  /* #64: (@17[:6] = @12) */
  for (rr=w17+0, ss=w12; rr!=w17+6; rr+=1) *rr = *ss++;
  /* #65: @12 = @1[:6] */
  for (rr=w12, ss=w1+0; ss!=w1+6; ss+=1) *rr++ = *ss;
  /* #66: (@17[:36:6] = @12) */
  for (rr=w17+0, ss=w12; rr!=w17+36; rr+=6) *rr = *ss++;
  /* #67: @14 = 00 */
  /* #68: @15 = 00 */
  /* #69: @12 = zeros(1x6) */
  casadi_clear(w12, 6);
  /* #70: @19 = 00 */
  /* #71: @2 = ones(8x1,1nz) */
  w2 = 1.;
  /* #72: {NULL, NULL, NULL, @0, NULL, NULL, NULL, NULL} = vertsplit(@2) */
  w0 = w2;
  /* #73: @20 = 00 */
  /* #74: @21 = 00 */
  /* #75: @22 = 00 */
  /* #76: @23 = 00 */
  /* #77: @2 = vertcat(@19, @0, @20, @21, @22, @23) */
  rr=(&w2);
  *rr++ = w0;
  /* #78: @2 = (-@2) */
  w2 = (- w2 );
  /* #79: @0 = @2' */
  casadi_copy((&w2), 1, (&w0));
  /* #80: @12 = mac(@0,@13,@12) */
  casadi_mtimes((&w0), casadi_s4, w13, casadi_s2, w12, casadi_s1, w, 0);
  /* #81: @12 = @12' */
  /* #82: @1 = zeros(1x6) */
  casadi_clear(w1, 6);
  /* #83: @2 = @2' */
  /* #84: @1 = mac(@2,@16,@1) */
  casadi_mtimes((&w2), casadi_s4, w16, casadi_s2, w1, casadi_s1, w, 0);
  /* #85: @1 = @1' */
  /* #86: @12 = (@12+@1) */
  for (i=0, rr=w12, cs=w1; i<6; ++i) (*rr++) += (*cs++);
  /* #87: @12 = (-@12) */
  for (i=0, rr=w12, cs=w12; i<6; ++i) *rr++ = (- *cs++ );
  /* #88: {@2, @0, @3, @4, @5, @6} = vertsplit(@12) */
  w2 = w12[0];
  w0 = w12[1];
  w3 = w12[2];
  w4 = w12[3];
  w5 = w12[4];
  w6 = w12[5];
  /* #89: @12 = vertcat(@14, @15, @2, @0, @3, @4, @5, @6) */
  rr=w12;
  *rr++ = w2;
  *rr++ = w0;
  *rr++ = w3;
  *rr++ = w4;
  *rr++ = w5;
  *rr++ = w6;
  /* #90: @1 = @12[:6] */
  for (rr=w1, ss=w12+0; ss!=w12+6; ss+=1) *rr++ = *ss;
  /* #91: (@17[6:12] = @1) */
  for (rr=w17+6, ss=w1; rr!=w17+12; rr+=1) *rr = *ss++;
  /* #92: @1 = @12[:6] */
  for (rr=w1, ss=w12+0; ss!=w12+6; ss+=1) *rr++ = *ss;
  /* #93: (@17[1:37:6] = @1) */
  for (rr=w17+1, ss=w1; rr!=w17+37; rr+=6) *rr = *ss++;
  /* #94: @14 = 00 */
  /* #95: @15 = 00 */
  /* #96: @1 = zeros(1x6) */
  casadi_clear(w1, 6);
  /* #97: @19 = 00 */
  /* #98: @20 = 00 */
  /* #99: @2 = ones(8x1,1nz) */
  w2 = 1.;
  /* #100: {NULL, NULL, NULL, NULL, @0, NULL, NULL, NULL} = vertsplit(@2) */
  w0 = w2;
  /* #101: @21 = 00 */
  /* #102: @22 = 00 */
  /* #103: @23 = 00 */
  /* #104: @2 = vertcat(@19, @20, @0, @21, @22, @23) */
  rr=(&w2);
  *rr++ = w0;
  /* #105: @2 = (-@2) */
  w2 = (- w2 );
  /* #106: @0 = @2' */
  casadi_copy((&w2), 1, (&w0));
  /* #107: @1 = mac(@0,@13,@1) */
  casadi_mtimes((&w0), casadi_s5, w13, casadi_s2, w1, casadi_s1, w, 0);
  /* #108: @1 = @1' */
  /* #109: @12 = zeros(1x6) */
  casadi_clear(w12, 6);
  /* #110: @2 = @2' */
  /* #111: @12 = mac(@2,@16,@12) */
  casadi_mtimes((&w2), casadi_s5, w16, casadi_s2, w12, casadi_s1, w, 0);
  /* #112: @12 = @12' */
  /* #113: @1 = (@1+@12) */
  for (i=0, rr=w1, cs=w12; i<6; ++i) (*rr++) += (*cs++);
  /* #114: @1 = (-@1) */
  for (i=0, rr=w1, cs=w1; i<6; ++i) *rr++ = (- *cs++ );
  /* #115: {@2, @0, @3, @4, @5, @6} = vertsplit(@1) */
  w2 = w1[0];
  w0 = w1[1];
  w3 = w1[2];
  w4 = w1[3];
  w5 = w1[4];
  w6 = w1[5];
  /* #116: @1 = vertcat(@14, @15, @2, @0, @3, @4, @5, @6) */
  rr=w1;
  *rr++ = w2;
  *rr++ = w0;
  *rr++ = w3;
  *rr++ = w4;
  *rr++ = w5;
  *rr++ = w6;
  /* #117: @12 = @1[:6] */
  for (rr=w12, ss=w1+0; ss!=w1+6; ss+=1) *rr++ = *ss;
  /* #118: (@17[12:18] = @12) */
  for (rr=w17+12, ss=w12; rr!=w17+18; rr+=1) *rr = *ss++;
  /* #119: @12 = @1[:6] */
  for (rr=w12, ss=w1+0; ss!=w1+6; ss+=1) *rr++ = *ss;
  /* #120: (@17[2:38:6] = @12) */
  for (rr=w17+2, ss=w12; rr!=w17+38; rr+=6) *rr = *ss++;
  /* #121: @14 = 00 */
  /* #122: @15 = 00 */
  /* #123: @12 = zeros(1x6) */
  casadi_clear(w12, 6);
  /* #124: @19 = 00 */
  /* #125: @20 = 00 */
  /* #126: @21 = 00 */
  /* #127: @2 = ones(8x1,1nz) */
  w2 = 1.;
  /* #128: {NULL, NULL, NULL, NULL, NULL, @0, NULL, NULL} = vertsplit(@2) */
  w0 = w2;
  /* #129: @22 = 00 */
  /* #130: @23 = 00 */
  /* #131: @2 = vertcat(@19, @20, @21, @0, @22, @23) */
  rr=(&w2);
  *rr++ = w0;
  /* #132: @2 = (-@2) */
  w2 = (- w2 );
  /* #133: @0 = @2' */
  casadi_copy((&w2), 1, (&w0));
  /* #134: @12 = mac(@0,@13,@12) */
  casadi_mtimes((&w0), casadi_s6, w13, casadi_s2, w12, casadi_s1, w, 0);
  /* #135: @12 = @12' */
  /* #136: @1 = zeros(1x6) */
  casadi_clear(w1, 6);
  /* #137: @2 = @2' */
  /* #138: @1 = mac(@2,@16,@1) */
  casadi_mtimes((&w2), casadi_s6, w16, casadi_s2, w1, casadi_s1, w, 0);
  /* #139: @1 = @1' */
  /* #140: @12 = (@12+@1) */
  for (i=0, rr=w12, cs=w1; i<6; ++i) (*rr++) += (*cs++);
  /* #141: @12 = (-@12) */
  for (i=0, rr=w12, cs=w12; i<6; ++i) *rr++ = (- *cs++ );
  /* #142: {@2, @0, @3, @4, @5, @6} = vertsplit(@12) */
  w2 = w12[0];
  w0 = w12[1];
  w3 = w12[2];
  w4 = w12[3];
  w5 = w12[4];
  w6 = w12[5];
  /* #143: @12 = vertcat(@14, @15, @2, @0, @3, @4, @5, @6) */
  rr=w12;
  *rr++ = w2;
  *rr++ = w0;
  *rr++ = w3;
  *rr++ = w4;
  *rr++ = w5;
  *rr++ = w6;
  /* #144: @1 = @12[:6] */
  for (rr=w1, ss=w12+0; ss!=w12+6; ss+=1) *rr++ = *ss;
  /* #145: (@17[18:24] = @1) */
  for (rr=w17+18, ss=w1; rr!=w17+24; rr+=1) *rr = *ss++;
  /* #146: @1 = @12[:6] */
  for (rr=w1, ss=w12+0; ss!=w12+6; ss+=1) *rr++ = *ss;
  /* #147: (@17[3:39:6] = @1) */
  for (rr=w17+3, ss=w1; rr!=w17+39; rr+=6) *rr = *ss++;
  /* #148: @14 = 00 */
  /* #149: @15 = 00 */
  /* #150: @1 = zeros(1x6) */
  casadi_clear(w1, 6);
  /* #151: @19 = 00 */
  /* #152: @20 = 00 */
  /* #153: @21 = 00 */
  /* #154: @22 = 00 */
  /* #155: @2 = ones(8x1,1nz) */
  w2 = 1.;
  /* #156: {NULL, NULL, NULL, NULL, NULL, NULL, @0, NULL} = vertsplit(@2) */
  w0 = w2;
  /* #157: @23 = 00 */
  /* #158: @2 = vertcat(@19, @20, @21, @22, @0, @23) */
  rr=(&w2);
  *rr++ = w0;
  /* #159: @2 = (-@2) */
  w2 = (- w2 );
  /* #160: @0 = @2' */
  casadi_copy((&w2), 1, (&w0));
  /* #161: @1 = mac(@0,@13,@1) */
  casadi_mtimes((&w0), casadi_s7, w13, casadi_s2, w1, casadi_s1, w, 0);
  /* #162: @1 = @1' */
  /* #163: @12 = zeros(1x6) */
  casadi_clear(w12, 6);
  /* #164: @2 = @2' */
  /* #165: @12 = mac(@2,@16,@12) */
  casadi_mtimes((&w2), casadi_s7, w16, casadi_s2, w12, casadi_s1, w, 0);
  /* #166: @12 = @12' */
  /* #167: @1 = (@1+@12) */
  for (i=0, rr=w1, cs=w12; i<6; ++i) (*rr++) += (*cs++);
  /* #168: @1 = (-@1) */
  for (i=0, rr=w1, cs=w1; i<6; ++i) *rr++ = (- *cs++ );
  /* #169: {@2, @0, @3, @4, @5, @6} = vertsplit(@1) */
  w2 = w1[0];
  w0 = w1[1];
  w3 = w1[2];
  w4 = w1[3];
  w5 = w1[4];
  w6 = w1[5];
  /* #170: @1 = vertcat(@14, @15, @2, @0, @3, @4, @5, @6) */
  rr=w1;
  *rr++ = w2;
  *rr++ = w0;
  *rr++ = w3;
  *rr++ = w4;
  *rr++ = w5;
  *rr++ = w6;
  /* #171: @12 = @1[:6] */
  for (rr=w12, ss=w1+0; ss!=w1+6; ss+=1) *rr++ = *ss;
  /* #172: (@17[24:30] = @12) */
  for (rr=w17+24, ss=w12; rr!=w17+30; rr+=1) *rr = *ss++;
  /* #173: @12 = @1[:6] */
  for (rr=w12, ss=w1+0; ss!=w1+6; ss+=1) *rr++ = *ss;
  /* #174: (@17[4:40:6] = @12) */
  for (rr=w17+4, ss=w12; rr!=w17+40; rr+=6) *rr = *ss++;
  /* #175: @14 = 00 */
  /* #176: @15 = 00 */
  /* #177: @12 = zeros(1x6) */
  casadi_clear(w12, 6);
  /* #178: @19 = 00 */
  /* #179: @20 = 00 */
  /* #180: @21 = 00 */
  /* #181: @22 = 00 */
  /* #182: @23 = 00 */
  /* #183: @2 = ones(8x1,1nz) */
  w2 = 1.;
  /* #184: {NULL, NULL, NULL, NULL, NULL, NULL, NULL, @0} = vertsplit(@2) */
  w0 = w2;
  /* #185: @2 = vertcat(@19, @20, @21, @22, @23, @0) */
  rr=(&w2);
  *rr++ = w0;
  /* #186: @2 = (-@2) */
  w2 = (- w2 );
  /* #187: @0 = @2' */
  casadi_copy((&w2), 1, (&w0));
  /* #188: @12 = mac(@0,@13,@12) */
  casadi_mtimes((&w0), casadi_s8, w13, casadi_s2, w12, casadi_s1, w, 0);
  /* #189: @12 = @12' */
  /* #190: @1 = zeros(1x6) */
  casadi_clear(w1, 6);
  /* #191: @2 = @2' */
  /* #192: @1 = mac(@2,@16,@1) */
  casadi_mtimes((&w2), casadi_s8, w16, casadi_s2, w1, casadi_s1, w, 0);
  /* #193: @1 = @1' */
  /* #194: @12 = (@12+@1) */
  for (i=0, rr=w12, cs=w1; i<6; ++i) (*rr++) += (*cs++);
  /* #195: @12 = (-@12) */
  for (i=0, rr=w12, cs=w12; i<6; ++i) *rr++ = (- *cs++ );
  /* #196: {@2, @0, @3, @4, @5, @6} = vertsplit(@12) */
  w2 = w12[0];
  w0 = w12[1];
  w3 = w12[2];
  w4 = w12[3];
  w5 = w12[4];
  w6 = w12[5];
  /* #197: @12 = vertcat(@14, @15, @2, @0, @3, @4, @5, @6) */
  rr=w12;
  *rr++ = w2;
  *rr++ = w0;
  *rr++ = w3;
  *rr++ = w4;
  *rr++ = w5;
  *rr++ = w6;
  /* #198: @1 = @12[:6] */
  for (rr=w1, ss=w12+0; ss!=w12+6; ss+=1) *rr++ = *ss;
  /* #199: (@17[30:36] = @1) */
  for (rr=w17+30, ss=w1; rr!=w17+36; rr+=1) *rr = *ss++;
  /* #200: @1 = @12[:6] */
  for (rr=w1, ss=w12+0; ss!=w12+6; ss+=1) *rr++ = *ss;
  /* #201: (@17[5:41:6] = @1) */
  for (rr=w17+5, ss=w1; rr!=w17+41; rr+=6) *rr = *ss++;
  /* #202: @16 = @17' */
  casadi_trans(w17,casadi_s9, w16, casadi_s9, iw);
  /* #203: output[2][0] = @16 */
  casadi_copy(w16, 36, res[2]);
  return 0;
}

CASADI_SYMBOL_EXPORT int Drone_ode_cost_ext_cost_fun_jac_hess(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int Drone_ode_cost_ext_cost_fun_jac_hess_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int Drone_ode_cost_ext_cost_fun_jac_hess_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void Drone_ode_cost_ext_cost_fun_jac_hess_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int Drone_ode_cost_ext_cost_fun_jac_hess_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void Drone_ode_cost_ext_cost_fun_jac_hess_release(int mem) {
}

CASADI_SYMBOL_EXPORT void Drone_ode_cost_ext_cost_fun_jac_hess_incref(void) {
}

CASADI_SYMBOL_EXPORT void Drone_ode_cost_ext_cost_fun_jac_hess_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int Drone_ode_cost_ext_cost_fun_jac_hess_n_in(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_int Drone_ode_cost_ext_cost_fun_jac_hess_n_out(void) { return 5;}

CASADI_SYMBOL_EXPORT casadi_real Drone_ode_cost_ext_cost_fun_jac_hess_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* Drone_ode_cost_ext_cost_fun_jac_hess_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* Drone_ode_cost_ext_cost_fun_jac_hess_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    case 2: return "o2";
    case 3: return "o3";
    case 4: return "o4";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* Drone_ode_cost_ext_cost_fun_jac_hess_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s10;
    case 1: return casadi_s11;
    case 2: return casadi_s12;
    case 3: return casadi_s13;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* Drone_ode_cost_ext_cost_fun_jac_hess_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s14;
    case 1: return casadi_s13;
    case 2: return casadi_s9;
    case 3: return casadi_s12;
    case 4: return casadi_s15;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int Drone_ode_cost_ext_cost_fun_jac_hess_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 12;
  if (sz_res) *sz_res = 13;
  if (sz_iw) *sz_iw = 9;
  if (sz_w) *sz_w = 147;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
