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
  #define CASADI_PREFIX(ID) Drone_ode_expl_vde_forw_ ## ID
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
#define casadi_mtimes CASADI_PREFIX(mtimes)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_s3 CASADI_PREFIX(s3)
#define casadi_s4 CASADI_PREFIX(s4)
#define casadi_s5 CASADI_PREFIX(s5)
#define casadi_s6 CASADI_PREFIX(s6)

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

static const casadi_int casadi_s0[10] = {6, 1, 0, 6, 0, 1, 2, 3, 4, 5};
static const casadi_int casadi_s1[5] = {2, 1, 0, 1, 0};
static const casadi_int casadi_s2[17] = {6, 2, 0, 6, 12, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5};
static const casadi_int casadi_s3[5] = {2, 1, 0, 1, 1};
static const casadi_int casadi_s4[45] = {6, 6, 0, 6, 12, 18, 24, 30, 36, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5};
static const casadi_int casadi_s5[6] = {2, 1, 0, 2, 0, 1};
static const casadi_int casadi_s6[17] = {13, 1, 0, 13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

/* Drone_ode_expl_vde_forw:(i0[6],i1[6x6],i2[6x2],i3[2],i4[13])->(o0[6],o1[6x6],o2[6x2]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_int i, j, k;
  casadi_real *rr, *ss, *tt;
  const casadi_real *cs;
  casadi_real *w0=w+6, w1, *w2=w+13, *w3=w+19, w4, w5, *w6=w+33, *w7=w+35, *w8=w+71, *w9=w+77, *w10=w+83, *w11=w+89, *w12=w+95, w13, w14, *w15=w+103, w16, w17;
  /* #0: @0 = zeros(6x1) */
  casadi_clear(w0, 6);
  /* #1: @1 = input[0][3] */
  w1 = arg[0] ? arg[0][3] : 0;
  /* #2: (@0[0] = @1) */
  for (rr=w0+0, ss=(&w1); rr!=w0+1; rr+=1) *rr = *ss++;
  /* #3: @1 = input[0][4] */
  w1 = arg[0] ? arg[0][4] : 0;
  /* #4: (@0[1] = @1) */
  for (rr=w0+1, ss=(&w1); rr!=w0+2; rr+=1) *rr = *ss++;
  /* #5: @1 = input[0][5] */
  w1 = arg[0] ? arg[0][5] : 0;
  /* #6: (@0[2] = @1) */
  for (rr=w0+2, ss=(&w1); rr!=w0+3; rr+=1) *rr = *ss++;
  /* #7: @1 = 0 */
  w1 = 0.;
  /* #8: (@0[3] = @1) */
  for (rr=w0+3, ss=(&w1); rr!=w0+4; rr+=1) *rr = *ss++;
  /* #9: @1 = -9.81 */
  w1 = -9.8100000000000005e+00;
  /* #10: (@0[4] = @1) */
  for (rr=w0+4, ss=(&w1); rr!=w0+5; rr+=1) *rr = *ss++;
  /* #11: @1 = 0 */
  w1 = 0.;
  /* #12: (@0[5] = @1) */
  for (rr=w0+5, ss=(&w1); rr!=w0+6; rr+=1) *rr = *ss++;
  /* #13: @2 = zeros(6x1) */
  casadi_clear(w2, 6);
  /* #14: @3 = zeros(6x2) */
  casadi_clear(w3, 12);
  /* #15: @1 = input[0][2] */
  w1 = arg[0] ? arg[0][2] : 0;
  /* #16: @4 = sin(@1) */
  w4 = sin( w1 );
  /* #17: @4 = (-@4) */
  w4 = (- w4 );
  /* #18: (@3[3] = @4) */
  for (rr=w3+3, ss=(&w4); rr!=w3+4; rr+=1) *rr = *ss++;
  /* #19: @4 = cos(@1) */
  w4 = cos( w1 );
  /* #20: (@3[4] = @4) */
  for (rr=w3+4, ss=(&w4); rr!=w3+5; rr+=1) *rr = *ss++;
  /* #21: @4 = 0 */
  w4 = 0.;
  /* #22: (@3[5] = @4) */
  for (rr=w3+5, ss=(&w4); rr!=w3+6; rr+=1) *rr = *ss++;
  /* #23: @4 = 0 */
  w4 = 0.;
  /* #24: (@3[9] = @4) */
  for (rr=w3+9, ss=(&w4); rr!=w3+10; rr+=1) *rr = *ss++;
  /* #25: @4 = 0 */
  w4 = 0.;
  /* #26: (@3[10] = @4) */
  for (rr=w3+10, ss=(&w4); rr!=w3+11; rr+=1) *rr = *ss++;
  /* #27: @4 = 50 */
  w4 = 50.;
  /* #28: (@3[11] = @4) */
  for (rr=w3+11, ss=(&w4); rr!=w3+12; rr+=1) *rr = *ss++;
  /* #29: @4 = input[3][0] */
  w4 = arg[3] ? arg[3][0] : 0;
  /* #30: @5 = input[3][1] */
  w5 = arg[3] ? arg[3][1] : 0;
  /* #31: @6 = vertcat(@4, @5) */
  rr=w6;
  *rr++ = w4;
  *rr++ = w5;
  /* #32: @2 = mac(@3,@6,@2) */
  for (i=0, rr=w2; i<1; ++i) for (j=0; j<6; ++j, ++rr) for (k=0, ss=w3+j, tt=w6+i*2; k<2; ++k) *rr += ss[k*6]**tt++;
  /* #33: @0 = (@0+@2) */
  for (i=0, rr=w0, cs=w2; i<6; ++i) (*rr++) += (*cs++);
  /* #34: output[0][0] = @0 */
  casadi_copy(w0, 6, res[0]);
  /* #35: @0 = zeros(6x1) */
  casadi_clear(w0, 6);
  /* #36: @7 = input[1][0] */
  casadi_copy(arg[1], 36, w7);
  /* #37: {@2, @8, @9, @10, @11, @12} = horzsplit(@7) */
  casadi_copy(w7, 6, w2);
  casadi_copy(w7+6, 6, w8);
  casadi_copy(w7+12, 6, w9);
  casadi_copy(w7+18, 6, w10);
  casadi_copy(w7+24, 6, w11);
  casadi_copy(w7+30, 6, w12);
  /* #38: {NULL, NULL, @4, @5, @13, @14} = vertsplit(@2) */
  w4 = w2[2];
  w5 = w2[3];
  w13 = w2[4];
  w14 = w2[5];
  /* #39: (@0[0] += @5) */
  for (rr=w0+0, ss=(&w5); rr!=w0+1; rr+=1) *rr += *ss++;
  /* #40: @5 = 0 */
  w5 = 0.;
  /* #41: (@0[1] = @5) */
  for (rr=w0+1, ss=(&w5); rr!=w0+2; rr+=1) *rr = *ss++;
  /* #42: (@0[1] += @13) */
  for (rr=w0+1, ss=(&w13); rr!=w0+2; rr+=1) *rr += *ss++;
  /* #43: @13 = 0 */
  w13 = 0.;
  /* #44: (@0[2] = @13) */
  for (rr=w0+2, ss=(&w13); rr!=w0+3; rr+=1) *rr = *ss++;
  /* #45: (@0[2] += @14) */
  for (rr=w0+2, ss=(&w14); rr!=w0+3; rr+=1) *rr += *ss++;
  /* #46: @14 = 0 */
  w14 = 0.;
  /* #47: (@0[3] = @14) */
  for (rr=w0+3, ss=(&w14); rr!=w0+4; rr+=1) *rr = *ss++;
  /* #48: @14 = 0 */
  w14 = 0.;
  /* #49: (@0[4] = @14) */
  for (rr=w0+4, ss=(&w14); rr!=w0+5; rr+=1) *rr = *ss++;
  /* #50: @14 = 0 */
  w14 = 0.;
  /* #51: (@0[5] = @14) */
  for (rr=w0+5, ss=(&w14); rr!=w0+6; rr+=1) *rr = *ss++;
  /* #52: @2 = zeros(6x1) */
  casadi_clear(w2, 6);
  /* #53: @15 = zeros(6x2) */
  casadi_clear(w15, 12);
  /* #54: @14 = cos(@1) */
  w14 = cos( w1 );
  /* #55: @13 = (@14*@4) */
  w13  = (w14*w4);
  /* #56: @13 = (-@13) */
  w13 = (- w13 );
  /* #57: (@15[3] += @13) */
  for (rr=w15+3, ss=(&w13); rr!=w15+4; rr+=1) *rr += *ss++;
  /* #58: @13 = 0 */
  w13 = 0.;
  /* #59: (@15[4] = @13) */
  for (rr=w15+4, ss=(&w13); rr!=w15+5; rr+=1) *rr = *ss++;
  /* #60: @13 = sin(@1) */
  w13 = sin( w1 );
  /* #61: @4 = (@13*@4) */
  w4  = (w13*w4);
  /* #62: @4 = (-@4) */
  w4 = (- w4 );
  /* #63: (@15[4] += @4) */
  for (rr=w15+4, ss=(&w4); rr!=w15+5; rr+=1) *rr += *ss++;
  /* #64: @4 = 0 */
  w4 = 0.;
  /* #65: (@15[5] = @4) */
  for (rr=w15+5, ss=(&w4); rr!=w15+6; rr+=1) *rr = *ss++;
  /* #66: @4 = 0 */
  w4 = 0.;
  /* #67: (@15[9] = @4) */
  for (rr=w15+9, ss=(&w4); rr!=w15+10; rr+=1) *rr = *ss++;
  /* #68: @4 = 0 */
  w4 = 0.;
  /* #69: (@15[10] = @4) */
  for (rr=w15+10, ss=(&w4); rr!=w15+11; rr+=1) *rr = *ss++;
  /* #70: @4 = 0 */
  w4 = 0.;
  /* #71: (@15[11] = @4) */
  for (rr=w15+11, ss=(&w4); rr!=w15+12; rr+=1) *rr = *ss++;
  /* #72: @2 = mac(@15,@6,@2) */
  for (i=0, rr=w2; i<1; ++i) for (j=0; j<6; ++j, ++rr) for (k=0, ss=w15+j, tt=w6+i*2; k<2; ++k) *rr += ss[k*6]**tt++;
  /* #73: @0 = (@0+@2) */
  for (i=0, rr=w0, cs=w2; i<6; ++i) (*rr++) += (*cs++);
  /* #74: output[1][0] = @0 */
  casadi_copy(w0, 6, res[1]);
  /* #75: @0 = zeros(6x1) */
  casadi_clear(w0, 6);
  /* #76: {NULL, NULL, @4, @5, @16, @17} = vertsplit(@8) */
  w4 = w8[2];
  w5 = w8[3];
  w16 = w8[4];
  w17 = w8[5];
  /* #77: (@0[0] += @5) */
  for (rr=w0+0, ss=(&w5); rr!=w0+1; rr+=1) *rr += *ss++;
  /* #78: @5 = 0 */
  w5 = 0.;
  /* #79: (@0[1] = @5) */
  for (rr=w0+1, ss=(&w5); rr!=w0+2; rr+=1) *rr = *ss++;
  /* #80: (@0[1] += @16) */
  for (rr=w0+1, ss=(&w16); rr!=w0+2; rr+=1) *rr += *ss++;
  /* #81: @16 = 0 */
  w16 = 0.;
  /* #82: (@0[2] = @16) */
  for (rr=w0+2, ss=(&w16); rr!=w0+3; rr+=1) *rr = *ss++;
  /* #83: (@0[2] += @17) */
  for (rr=w0+2, ss=(&w17); rr!=w0+3; rr+=1) *rr += *ss++;
  /* #84: @17 = 0 */
  w17 = 0.;
  /* #85: (@0[3] = @17) */
  for (rr=w0+3, ss=(&w17); rr!=w0+4; rr+=1) *rr = *ss++;
  /* #86: @17 = 0 */
  w17 = 0.;
  /* #87: (@0[4] = @17) */
  for (rr=w0+4, ss=(&w17); rr!=w0+5; rr+=1) *rr = *ss++;
  /* #88: @17 = 0 */
  w17 = 0.;
  /* #89: (@0[5] = @17) */
  for (rr=w0+5, ss=(&w17); rr!=w0+6; rr+=1) *rr = *ss++;
  /* #90: @8 = zeros(6x1) */
  casadi_clear(w8, 6);
  /* #91: @15 = zeros(6x2) */
  casadi_clear(w15, 12);
  /* #92: @17 = (@14*@4) */
  w17  = (w14*w4);
  /* #93: @17 = (-@17) */
  w17 = (- w17 );
  /* #94: (@15[3] += @17) */
  for (rr=w15+3, ss=(&w17); rr!=w15+4; rr+=1) *rr += *ss++;
  /* #95: @17 = 0 */
  w17 = 0.;
  /* #96: (@15[4] = @17) */
  for (rr=w15+4, ss=(&w17); rr!=w15+5; rr+=1) *rr = *ss++;
  /* #97: @4 = (@13*@4) */
  w4  = (w13*w4);
  /* #98: @4 = (-@4) */
  w4 = (- w4 );
  /* #99: (@15[4] += @4) */
  for (rr=w15+4, ss=(&w4); rr!=w15+5; rr+=1) *rr += *ss++;
  /* #100: @4 = 0 */
  w4 = 0.;
  /* #101: (@15[5] = @4) */
  for (rr=w15+5, ss=(&w4); rr!=w15+6; rr+=1) *rr = *ss++;
  /* #102: @4 = 0 */
  w4 = 0.;
  /* #103: (@15[9] = @4) */
  for (rr=w15+9, ss=(&w4); rr!=w15+10; rr+=1) *rr = *ss++;
  /* #104: @4 = 0 */
  w4 = 0.;
  /* #105: (@15[10] = @4) */
  for (rr=w15+10, ss=(&w4); rr!=w15+11; rr+=1) *rr = *ss++;
  /* #106: @4 = 0 */
  w4 = 0.;
  /* #107: (@15[11] = @4) */
  for (rr=w15+11, ss=(&w4); rr!=w15+12; rr+=1) *rr = *ss++;
  /* #108: @8 = mac(@15,@6,@8) */
  for (i=0, rr=w8; i<1; ++i) for (j=0; j<6; ++j, ++rr) for (k=0, ss=w15+j, tt=w6+i*2; k<2; ++k) *rr += ss[k*6]**tt++;
  /* #109: @0 = (@0+@8) */
  for (i=0, rr=w0, cs=w8; i<6; ++i) (*rr++) += (*cs++);
  /* #110: output[1][1] = @0 */
  if (res[1]) casadi_copy(w0, 6, res[1]+6);
  /* #111: @0 = zeros(6x1) */
  casadi_clear(w0, 6);
  /* #112: {NULL, NULL, @4, @17, @16, @5} = vertsplit(@9) */
  w4 = w9[2];
  w17 = w9[3];
  w16 = w9[4];
  w5 = w9[5];
  /* #113: (@0[0] += @17) */
  for (rr=w0+0, ss=(&w17); rr!=w0+1; rr+=1) *rr += *ss++;
  /* #114: @17 = 0 */
  w17 = 0.;
  /* #115: (@0[1] = @17) */
  for (rr=w0+1, ss=(&w17); rr!=w0+2; rr+=1) *rr = *ss++;
  /* #116: (@0[1] += @16) */
  for (rr=w0+1, ss=(&w16); rr!=w0+2; rr+=1) *rr += *ss++;
  /* #117: @16 = 0 */
  w16 = 0.;
  /* #118: (@0[2] = @16) */
  for (rr=w0+2, ss=(&w16); rr!=w0+3; rr+=1) *rr = *ss++;
  /* #119: (@0[2] += @5) */
  for (rr=w0+2, ss=(&w5); rr!=w0+3; rr+=1) *rr += *ss++;
  /* #120: @5 = 0 */
  w5 = 0.;
  /* #121: (@0[3] = @5) */
  for (rr=w0+3, ss=(&w5); rr!=w0+4; rr+=1) *rr = *ss++;
  /* #122: @5 = 0 */
  w5 = 0.;
  /* #123: (@0[4] = @5) */
  for (rr=w0+4, ss=(&w5); rr!=w0+5; rr+=1) *rr = *ss++;
  /* #124: @5 = 0 */
  w5 = 0.;
  /* #125: (@0[5] = @5) */
  for (rr=w0+5, ss=(&w5); rr!=w0+6; rr+=1) *rr = *ss++;
  /* #126: @9 = zeros(6x1) */
  casadi_clear(w9, 6);
  /* #127: @15 = zeros(6x2) */
  casadi_clear(w15, 12);
  /* #128: @5 = (@14*@4) */
  w5  = (w14*w4);
  /* #129: @5 = (-@5) */
  w5 = (- w5 );
  /* #130: (@15[3] += @5) */
  for (rr=w15+3, ss=(&w5); rr!=w15+4; rr+=1) *rr += *ss++;
  /* #131: @5 = 0 */
  w5 = 0.;
  /* #132: (@15[4] = @5) */
  for (rr=w15+4, ss=(&w5); rr!=w15+5; rr+=1) *rr = *ss++;
  /* #133: @4 = (@13*@4) */
  w4  = (w13*w4);
  /* #134: @4 = (-@4) */
  w4 = (- w4 );
  /* #135: (@15[4] += @4) */
  for (rr=w15+4, ss=(&w4); rr!=w15+5; rr+=1) *rr += *ss++;
  /* #136: @4 = 0 */
  w4 = 0.;
  /* #137: (@15[5] = @4) */
  for (rr=w15+5, ss=(&w4); rr!=w15+6; rr+=1) *rr = *ss++;
  /* #138: @4 = 0 */
  w4 = 0.;
  /* #139: (@15[9] = @4) */
  for (rr=w15+9, ss=(&w4); rr!=w15+10; rr+=1) *rr = *ss++;
  /* #140: @4 = 0 */
  w4 = 0.;
  /* #141: (@15[10] = @4) */
  for (rr=w15+10, ss=(&w4); rr!=w15+11; rr+=1) *rr = *ss++;
  /* #142: @4 = 0 */
  w4 = 0.;
  /* #143: (@15[11] = @4) */
  for (rr=w15+11, ss=(&w4); rr!=w15+12; rr+=1) *rr = *ss++;
  /* #144: @9 = mac(@15,@6,@9) */
  for (i=0, rr=w9; i<1; ++i) for (j=0; j<6; ++j, ++rr) for (k=0, ss=w15+j, tt=w6+i*2; k<2; ++k) *rr += ss[k*6]**tt++;
  /* #145: @0 = (@0+@9) */
  for (i=0, rr=w0, cs=w9; i<6; ++i) (*rr++) += (*cs++);
  /* #146: output[1][2] = @0 */
  if (res[1]) casadi_copy(w0, 6, res[1]+12);
  /* #147: @0 = zeros(6x1) */
  casadi_clear(w0, 6);
  /* #148: {NULL, NULL, @4, @5, @16, @17} = vertsplit(@10) */
  w4 = w10[2];
  w5 = w10[3];
  w16 = w10[4];
  w17 = w10[5];
  /* #149: (@0[0] += @5) */
  for (rr=w0+0, ss=(&w5); rr!=w0+1; rr+=1) *rr += *ss++;
  /* #150: @5 = 0 */
  w5 = 0.;
  /* #151: (@0[1] = @5) */
  for (rr=w0+1, ss=(&w5); rr!=w0+2; rr+=1) *rr = *ss++;
  /* #152: (@0[1] += @16) */
  for (rr=w0+1, ss=(&w16); rr!=w0+2; rr+=1) *rr += *ss++;
  /* #153: @16 = 0 */
  w16 = 0.;
  /* #154: (@0[2] = @16) */
  for (rr=w0+2, ss=(&w16); rr!=w0+3; rr+=1) *rr = *ss++;
  /* #155: (@0[2] += @17) */
  for (rr=w0+2, ss=(&w17); rr!=w0+3; rr+=1) *rr += *ss++;
  /* #156: @17 = 0 */
  w17 = 0.;
  /* #157: (@0[3] = @17) */
  for (rr=w0+3, ss=(&w17); rr!=w0+4; rr+=1) *rr = *ss++;
  /* #158: @17 = 0 */
  w17 = 0.;
  /* #159: (@0[4] = @17) */
  for (rr=w0+4, ss=(&w17); rr!=w0+5; rr+=1) *rr = *ss++;
  /* #160: @17 = 0 */
  w17 = 0.;
  /* #161: (@0[5] = @17) */
  for (rr=w0+5, ss=(&w17); rr!=w0+6; rr+=1) *rr = *ss++;
  /* #162: @10 = zeros(6x1) */
  casadi_clear(w10, 6);
  /* #163: @15 = zeros(6x2) */
  casadi_clear(w15, 12);
  /* #164: @17 = (@14*@4) */
  w17  = (w14*w4);
  /* #165: @17 = (-@17) */
  w17 = (- w17 );
  /* #166: (@15[3] += @17) */
  for (rr=w15+3, ss=(&w17); rr!=w15+4; rr+=1) *rr += *ss++;
  /* #167: @17 = 0 */
  w17 = 0.;
  /* #168: (@15[4] = @17) */
  for (rr=w15+4, ss=(&w17); rr!=w15+5; rr+=1) *rr = *ss++;
  /* #169: @4 = (@13*@4) */
  w4  = (w13*w4);
  /* #170: @4 = (-@4) */
  w4 = (- w4 );
  /* #171: (@15[4] += @4) */
  for (rr=w15+4, ss=(&w4); rr!=w15+5; rr+=1) *rr += *ss++;
  /* #172: @4 = 0 */
  w4 = 0.;
  /* #173: (@15[5] = @4) */
  for (rr=w15+5, ss=(&w4); rr!=w15+6; rr+=1) *rr = *ss++;
  /* #174: @4 = 0 */
  w4 = 0.;
  /* #175: (@15[9] = @4) */
  for (rr=w15+9, ss=(&w4); rr!=w15+10; rr+=1) *rr = *ss++;
  /* #176: @4 = 0 */
  w4 = 0.;
  /* #177: (@15[10] = @4) */
  for (rr=w15+10, ss=(&w4); rr!=w15+11; rr+=1) *rr = *ss++;
  /* #178: @4 = 0 */
  w4 = 0.;
  /* #179: (@15[11] = @4) */
  for (rr=w15+11, ss=(&w4); rr!=w15+12; rr+=1) *rr = *ss++;
  /* #180: @10 = mac(@15,@6,@10) */
  for (i=0, rr=w10; i<1; ++i) for (j=0; j<6; ++j, ++rr) for (k=0, ss=w15+j, tt=w6+i*2; k<2; ++k) *rr += ss[k*6]**tt++;
  /* #181: @0 = (@0+@10) */
  for (i=0, rr=w0, cs=w10; i<6; ++i) (*rr++) += (*cs++);
  /* #182: output[1][3] = @0 */
  if (res[1]) casadi_copy(w0, 6, res[1]+18);
  /* #183: @0 = zeros(6x1) */
  casadi_clear(w0, 6);
  /* #184: {NULL, NULL, @4, @17, @16, @5} = vertsplit(@11) */
  w4 = w11[2];
  w17 = w11[3];
  w16 = w11[4];
  w5 = w11[5];
  /* #185: (@0[0] += @17) */
  for (rr=w0+0, ss=(&w17); rr!=w0+1; rr+=1) *rr += *ss++;
  /* #186: @17 = 0 */
  w17 = 0.;
  /* #187: (@0[1] = @17) */
  for (rr=w0+1, ss=(&w17); rr!=w0+2; rr+=1) *rr = *ss++;
  /* #188: (@0[1] += @16) */
  for (rr=w0+1, ss=(&w16); rr!=w0+2; rr+=1) *rr += *ss++;
  /* #189: @16 = 0 */
  w16 = 0.;
  /* #190: (@0[2] = @16) */
  for (rr=w0+2, ss=(&w16); rr!=w0+3; rr+=1) *rr = *ss++;
  /* #191: (@0[2] += @5) */
  for (rr=w0+2, ss=(&w5); rr!=w0+3; rr+=1) *rr += *ss++;
  /* #192: @5 = 0 */
  w5 = 0.;
  /* #193: (@0[3] = @5) */
  for (rr=w0+3, ss=(&w5); rr!=w0+4; rr+=1) *rr = *ss++;
  /* #194: @5 = 0 */
  w5 = 0.;
  /* #195: (@0[4] = @5) */
  for (rr=w0+4, ss=(&w5); rr!=w0+5; rr+=1) *rr = *ss++;
  /* #196: @5 = 0 */
  w5 = 0.;
  /* #197: (@0[5] = @5) */
  for (rr=w0+5, ss=(&w5); rr!=w0+6; rr+=1) *rr = *ss++;
  /* #198: @11 = zeros(6x1) */
  casadi_clear(w11, 6);
  /* #199: @15 = zeros(6x2) */
  casadi_clear(w15, 12);
  /* #200: @5 = (@14*@4) */
  w5  = (w14*w4);
  /* #201: @5 = (-@5) */
  w5 = (- w5 );
  /* #202: (@15[3] += @5) */
  for (rr=w15+3, ss=(&w5); rr!=w15+4; rr+=1) *rr += *ss++;
  /* #203: @5 = 0 */
  w5 = 0.;
  /* #204: (@15[4] = @5) */
  for (rr=w15+4, ss=(&w5); rr!=w15+5; rr+=1) *rr = *ss++;
  /* #205: @4 = (@13*@4) */
  w4  = (w13*w4);
  /* #206: @4 = (-@4) */
  w4 = (- w4 );
  /* #207: (@15[4] += @4) */
  for (rr=w15+4, ss=(&w4); rr!=w15+5; rr+=1) *rr += *ss++;
  /* #208: @4 = 0 */
  w4 = 0.;
  /* #209: (@15[5] = @4) */
  for (rr=w15+5, ss=(&w4); rr!=w15+6; rr+=1) *rr = *ss++;
  /* #210: @4 = 0 */
  w4 = 0.;
  /* #211: (@15[9] = @4) */
  for (rr=w15+9, ss=(&w4); rr!=w15+10; rr+=1) *rr = *ss++;
  /* #212: @4 = 0 */
  w4 = 0.;
  /* #213: (@15[10] = @4) */
  for (rr=w15+10, ss=(&w4); rr!=w15+11; rr+=1) *rr = *ss++;
  /* #214: @4 = 0 */
  w4 = 0.;
  /* #215: (@15[11] = @4) */
  for (rr=w15+11, ss=(&w4); rr!=w15+12; rr+=1) *rr = *ss++;
  /* #216: @11 = mac(@15,@6,@11) */
  for (i=0, rr=w11; i<1; ++i) for (j=0; j<6; ++j, ++rr) for (k=0, ss=w15+j, tt=w6+i*2; k<2; ++k) *rr += ss[k*6]**tt++;
  /* #217: @0 = (@0+@11) */
  for (i=0, rr=w0, cs=w11; i<6; ++i) (*rr++) += (*cs++);
  /* #218: output[1][4] = @0 */
  if (res[1]) casadi_copy(w0, 6, res[1]+24);
  /* #219: @0 = zeros(6x1) */
  casadi_clear(w0, 6);
  /* #220: {NULL, NULL, @4, @5, @16, @17} = vertsplit(@12) */
  w4 = w12[2];
  w5 = w12[3];
  w16 = w12[4];
  w17 = w12[5];
  /* #221: (@0[0] += @5) */
  for (rr=w0+0, ss=(&w5); rr!=w0+1; rr+=1) *rr += *ss++;
  /* #222: @5 = 0 */
  w5 = 0.;
  /* #223: (@0[1] = @5) */
  for (rr=w0+1, ss=(&w5); rr!=w0+2; rr+=1) *rr = *ss++;
  /* #224: (@0[1] += @16) */
  for (rr=w0+1, ss=(&w16); rr!=w0+2; rr+=1) *rr += *ss++;
  /* #225: @16 = 0 */
  w16 = 0.;
  /* #226: (@0[2] = @16) */
  for (rr=w0+2, ss=(&w16); rr!=w0+3; rr+=1) *rr = *ss++;
  /* #227: (@0[2] += @17) */
  for (rr=w0+2, ss=(&w17); rr!=w0+3; rr+=1) *rr += *ss++;
  /* #228: @17 = 0 */
  w17 = 0.;
  /* #229: (@0[3] = @17) */
  for (rr=w0+3, ss=(&w17); rr!=w0+4; rr+=1) *rr = *ss++;
  /* #230: @17 = 0 */
  w17 = 0.;
  /* #231: (@0[4] = @17) */
  for (rr=w0+4, ss=(&w17); rr!=w0+5; rr+=1) *rr = *ss++;
  /* #232: @17 = 0 */
  w17 = 0.;
  /* #233: (@0[5] = @17) */
  for (rr=w0+5, ss=(&w17); rr!=w0+6; rr+=1) *rr = *ss++;
  /* #234: @12 = zeros(6x1) */
  casadi_clear(w12, 6);
  /* #235: @15 = zeros(6x2) */
  casadi_clear(w15, 12);
  /* #236: @14 = (@14*@4) */
  w14 *= w4;
  /* #237: @14 = (-@14) */
  w14 = (- w14 );
  /* #238: (@15[3] += @14) */
  for (rr=w15+3, ss=(&w14); rr!=w15+4; rr+=1) *rr += *ss++;
  /* #239: @14 = 0 */
  w14 = 0.;
  /* #240: (@15[4] = @14) */
  for (rr=w15+4, ss=(&w14); rr!=w15+5; rr+=1) *rr = *ss++;
  /* #241: @13 = (@13*@4) */
  w13 *= w4;
  /* #242: @13 = (-@13) */
  w13 = (- w13 );
  /* #243: (@15[4] += @13) */
  for (rr=w15+4, ss=(&w13); rr!=w15+5; rr+=1) *rr += *ss++;
  /* #244: @13 = 0 */
  w13 = 0.;
  /* #245: (@15[5] = @13) */
  for (rr=w15+5, ss=(&w13); rr!=w15+6; rr+=1) *rr = *ss++;
  /* #246: @13 = 0 */
  w13 = 0.;
  /* #247: (@15[9] = @13) */
  for (rr=w15+9, ss=(&w13); rr!=w15+10; rr+=1) *rr = *ss++;
  /* #248: @13 = 0 */
  w13 = 0.;
  /* #249: (@15[10] = @13) */
  for (rr=w15+10, ss=(&w13); rr!=w15+11; rr+=1) *rr = *ss++;
  /* #250: @13 = 0 */
  w13 = 0.;
  /* #251: (@15[11] = @13) */
  for (rr=w15+11, ss=(&w13); rr!=w15+12; rr+=1) *rr = *ss++;
  /* #252: @12 = mac(@15,@6,@12) */
  for (i=0, rr=w12; i<1; ++i) for (j=0; j<6; ++j, ++rr) for (k=0, ss=w15+j, tt=w6+i*2; k<2; ++k) *rr += ss[k*6]**tt++;
  /* #253: @0 = (@0+@12) */
  for (i=0, rr=w0, cs=w12; i<6; ++i) (*rr++) += (*cs++);
  /* #254: output[1][5] = @0 */
  if (res[1]) casadi_copy(w0, 6, res[1]+30);
  /* #255: @15 = zeros(2x6) */
  casadi_clear(w15, 12);
  /* #256: @0 = zeros(6x1) */
  casadi_clear(w0, 6);
  /* #257: @13 = ones(2x1,1nz) */
  w13 = 1.;
  /* #258: {@4, NULL} = vertsplit(@13) */
  w4 = w13;
  /* #259: @18 = 00 */
  /* #260: @13 = vertcat(@4, @18) */
  rr=(&w13);
  *rr++ = w4;
  /* #261: @0 = mac(@3,@13,@0) */
  casadi_mtimes(w3, casadi_s2, (&w13), casadi_s1, w0, casadi_s0, w, 0);
  /* #262: (@15[:12:2] = @0) */
  for (rr=w15+0, ss=w0; rr!=w15+12; rr+=2) *rr = *ss++;
  /* #263: @0 = zeros(6x1) */
  casadi_clear(w0, 6);
  /* #264: @18 = 00 */
  /* #265: @13 = ones(2x1,1nz) */
  w13 = 1.;
  /* #266: {NULL, @4} = vertsplit(@13) */
  w4 = w13;
  /* #267: @13 = vertcat(@18, @4) */
  rr=(&w13);
  *rr++ = w4;
  /* #268: @0 = mac(@3,@13,@0) */
  casadi_mtimes(w3, casadi_s2, (&w13), casadi_s3, w0, casadi_s0, w, 0);
  /* #269: (@15[1:13:2] = @0) */
  for (rr=w15+1, ss=w0; rr!=w15+13; rr+=2) *rr = *ss++;
  /* #270: @3 = @15' */
  for (i=0, rr=w3, cs=w15; i<6; ++i) for (j=0; j<2; ++j) rr[i+j*6] = *cs++;
  /* #271: @0 = zeros(6x1) */
  casadi_clear(w0, 6);
  /* #272: @15 = input[2][0] */
  casadi_copy(arg[2], 12, w15);
  /* #273: {@12, @11} = horzsplit(@15) */
  casadi_copy(w15, 6, w12);
  casadi_copy(w15+6, 6, w11);
  /* #274: {NULL, NULL, @13, @4, @14, @17} = vertsplit(@12) */
  w13 = w12[2];
  w4 = w12[3];
  w14 = w12[4];
  w17 = w12[5];
  /* #275: (@0[0] += @4) */
  for (rr=w0+0, ss=(&w4); rr!=w0+1; rr+=1) *rr += *ss++;
  /* #276: @4 = 0 */
  w4 = 0.;
  /* #277: (@0[1] = @4) */
  for (rr=w0+1, ss=(&w4); rr!=w0+2; rr+=1) *rr = *ss++;
  /* #278: (@0[1] += @14) */
  for (rr=w0+1, ss=(&w14); rr!=w0+2; rr+=1) *rr += *ss++;
  /* #279: @14 = 0 */
  w14 = 0.;
  /* #280: (@0[2] = @14) */
  for (rr=w0+2, ss=(&w14); rr!=w0+3; rr+=1) *rr = *ss++;
  /* #281: (@0[2] += @17) */
  for (rr=w0+2, ss=(&w17); rr!=w0+3; rr+=1) *rr += *ss++;
  /* #282: @17 = 0 */
  w17 = 0.;
  /* #283: (@0[3] = @17) */
  for (rr=w0+3, ss=(&w17); rr!=w0+4; rr+=1) *rr = *ss++;
  /* #284: @17 = 0 */
  w17 = 0.;
  /* #285: (@0[4] = @17) */
  for (rr=w0+4, ss=(&w17); rr!=w0+5; rr+=1) *rr = *ss++;
  /* #286: @17 = 0 */
  w17 = 0.;
  /* #287: (@0[5] = @17) */
  for (rr=w0+5, ss=(&w17); rr!=w0+6; rr+=1) *rr = *ss++;
  /* #288: @12 = zeros(6x1) */
  casadi_clear(w12, 6);
  /* #289: @15 = zeros(6x2) */
  casadi_clear(w15, 12);
  /* #290: @17 = cos(@1) */
  w17 = cos( w1 );
  /* #291: @14 = (@17*@13) */
  w14  = (w17*w13);
  /* #292: @14 = (-@14) */
  w14 = (- w14 );
  /* #293: (@15[3] += @14) */
  for (rr=w15+3, ss=(&w14); rr!=w15+4; rr+=1) *rr += *ss++;
  /* #294: @14 = 0 */
  w14 = 0.;
  /* #295: (@15[4] = @14) */
  for (rr=w15+4, ss=(&w14); rr!=w15+5; rr+=1) *rr = *ss++;
  /* #296: @1 = sin(@1) */
  w1 = sin( w1 );
  /* #297: @13 = (@1*@13) */
  w13  = (w1*w13);
  /* #298: @13 = (-@13) */
  w13 = (- w13 );
  /* #299: (@15[4] += @13) */
  for (rr=w15+4, ss=(&w13); rr!=w15+5; rr+=1) *rr += *ss++;
  /* #300: @13 = 0 */
  w13 = 0.;
  /* #301: (@15[5] = @13) */
  for (rr=w15+5, ss=(&w13); rr!=w15+6; rr+=1) *rr = *ss++;
  /* #302: @13 = 0 */
  w13 = 0.;
  /* #303: (@15[9] = @13) */
  for (rr=w15+9, ss=(&w13); rr!=w15+10; rr+=1) *rr = *ss++;
  /* #304: @13 = 0 */
  w13 = 0.;
  /* #305: (@15[10] = @13) */
  for (rr=w15+10, ss=(&w13); rr!=w15+11; rr+=1) *rr = *ss++;
  /* #306: @13 = 0 */
  w13 = 0.;
  /* #307: (@15[11] = @13) */
  for (rr=w15+11, ss=(&w13); rr!=w15+12; rr+=1) *rr = *ss++;
  /* #308: @12 = mac(@15,@6,@12) */
  for (i=0, rr=w12; i<1; ++i) for (j=0; j<6; ++j, ++rr) for (k=0, ss=w15+j, tt=w6+i*2; k<2; ++k) *rr += ss[k*6]**tt++;
  /* #309: @0 = (@0+@12) */
  for (i=0, rr=w0, cs=w12; i<6; ++i) (*rr++) += (*cs++);
  /* #310: @12 = zeros(6x1) */
  casadi_clear(w12, 6);
  /* #311: {NULL, NULL, @13, @14, @4, @16} = vertsplit(@11) */
  w13 = w11[2];
  w14 = w11[3];
  w4 = w11[4];
  w16 = w11[5];
  /* #312: (@12[0] += @14) */
  for (rr=w12+0, ss=(&w14); rr!=w12+1; rr+=1) *rr += *ss++;
  /* #313: @14 = 0 */
  w14 = 0.;
  /* #314: (@12[1] = @14) */
  for (rr=w12+1, ss=(&w14); rr!=w12+2; rr+=1) *rr = *ss++;
  /* #315: (@12[1] += @4) */
  for (rr=w12+1, ss=(&w4); rr!=w12+2; rr+=1) *rr += *ss++;
  /* #316: @4 = 0 */
  w4 = 0.;
  /* #317: (@12[2] = @4) */
  for (rr=w12+2, ss=(&w4); rr!=w12+3; rr+=1) *rr = *ss++;
  /* #318: (@12[2] += @16) */
  for (rr=w12+2, ss=(&w16); rr!=w12+3; rr+=1) *rr += *ss++;
  /* #319: @16 = 0 */
  w16 = 0.;
  /* #320: (@12[3] = @16) */
  for (rr=w12+3, ss=(&w16); rr!=w12+4; rr+=1) *rr = *ss++;
  /* #321: @16 = 0 */
  w16 = 0.;
  /* #322: (@12[4] = @16) */
  for (rr=w12+4, ss=(&w16); rr!=w12+5; rr+=1) *rr = *ss++;
  /* #323: @16 = 0 */
  w16 = 0.;
  /* #324: (@12[5] = @16) */
  for (rr=w12+5, ss=(&w16); rr!=w12+6; rr+=1) *rr = *ss++;
  /* #325: @11 = zeros(6x1) */
  casadi_clear(w11, 6);
  /* #326: @15 = zeros(6x2) */
  casadi_clear(w15, 12);
  /* #327: @17 = (@17*@13) */
  w17 *= w13;
  /* #328: @17 = (-@17) */
  w17 = (- w17 );
  /* #329: (@15[3] += @17) */
  for (rr=w15+3, ss=(&w17); rr!=w15+4; rr+=1) *rr += *ss++;
  /* #330: @17 = 0 */
  w17 = 0.;
  /* #331: (@15[4] = @17) */
  for (rr=w15+4, ss=(&w17); rr!=w15+5; rr+=1) *rr = *ss++;
  /* #332: @1 = (@1*@13) */
  w1 *= w13;
  /* #333: @1 = (-@1) */
  w1 = (- w1 );
  /* #334: (@15[4] += @1) */
  for (rr=w15+4, ss=(&w1); rr!=w15+5; rr+=1) *rr += *ss++;
  /* #335: @1 = 0 */
  w1 = 0.;
  /* #336: (@15[5] = @1) */
  for (rr=w15+5, ss=(&w1); rr!=w15+6; rr+=1) *rr = *ss++;
  /* #337: @1 = 0 */
  w1 = 0.;
  /* #338: (@15[9] = @1) */
  for (rr=w15+9, ss=(&w1); rr!=w15+10; rr+=1) *rr = *ss++;
  /* #339: @1 = 0 */
  w1 = 0.;
  /* #340: (@15[10] = @1) */
  for (rr=w15+10, ss=(&w1); rr!=w15+11; rr+=1) *rr = *ss++;
  /* #341: @1 = 0 */
  w1 = 0.;
  /* #342: (@15[11] = @1) */
  for (rr=w15+11, ss=(&w1); rr!=w15+12; rr+=1) *rr = *ss++;
  /* #343: @11 = mac(@15,@6,@11) */
  for (i=0, rr=w11; i<1; ++i) for (j=0; j<6; ++j, ++rr) for (k=0, ss=w15+j, tt=w6+i*2; k<2; ++k) *rr += ss[k*6]**tt++;
  /* #344: @12 = (@12+@11) */
  for (i=0, rr=w12, cs=w11; i<6; ++i) (*rr++) += (*cs++);
  /* #345: @15 = horzcat(@0, @12) */
  rr=w15;
  for (i=0, cs=w0; i<6; ++i) *rr++ = *cs++;
  for (i=0, cs=w12; i<6; ++i) *rr++ = *cs++;
  /* #346: @3 = (@3+@15) */
  for (i=0, rr=w3, cs=w15; i<12; ++i) (*rr++) += (*cs++);
  /* #347: output[2][0] = @3 */
  casadi_copy(w3, 12, res[2]);
  return 0;
}

CASADI_SYMBOL_EXPORT int Drone_ode_expl_vde_forw(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int Drone_ode_expl_vde_forw_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int Drone_ode_expl_vde_forw_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void Drone_ode_expl_vde_forw_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int Drone_ode_expl_vde_forw_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void Drone_ode_expl_vde_forw_release(int mem) {
}

CASADI_SYMBOL_EXPORT void Drone_ode_expl_vde_forw_incref(void) {
}

CASADI_SYMBOL_EXPORT void Drone_ode_expl_vde_forw_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int Drone_ode_expl_vde_forw_n_in(void) { return 5;}

CASADI_SYMBOL_EXPORT casadi_int Drone_ode_expl_vde_forw_n_out(void) { return 3;}

CASADI_SYMBOL_EXPORT casadi_real Drone_ode_expl_vde_forw_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* Drone_ode_expl_vde_forw_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    case 4: return "i4";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* Drone_ode_expl_vde_forw_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    case 2: return "o2";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* Drone_ode_expl_vde_forw_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s4;
    case 2: return casadi_s2;
    case 3: return casadi_s5;
    case 4: return casadi_s6;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* Drone_ode_expl_vde_forw_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s4;
    case 2: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int Drone_ode_expl_vde_forw_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 8;
  if (sz_res) *sz_res = 9;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 117;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
