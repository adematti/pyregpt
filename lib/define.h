#ifndef _REgammaPT_DEFINE_
#define _REgammaPT_DEFINE_

#define MAX(a, b) (((a) > (b)) ? (a) : (b)) //Maximum of two numbers
#define MIN(a, b) (((a) < (b)) ? (a) : (b)) //Minimum of two numbers
#define CLAMP(x, low, high)  (((x) > (high)) ? (high) : (((x) < (low)) ? (low) : (x))) //min(max(a,low),high)
#define MAX3(a, b, c) MAX(MAX((a), (b)), MAX((b), (c)))
#define MIN3(a, b, c) MIN(MIN((a), (b)), MIN((b), (c)))
//#define ABS(a)   (((a) < 0) ? -(a) : (a)) //Absolute value

#ifdef _FLOAT32
typedef float histo_t;
#else
typedef double histo_t;
#endif //_FLOAT32

typedef enum {DELTA, THETA} FLAG;
typedef enum {LIN, POLY} INTERPOL;

typedef struct {
	size_t n;
	histo_t min;
	histo_t max;
	INTERPOL interpol;
} Precision;

typedef struct {
	size_t nq;
	histo_t k;
	histo_t *x;
	histo_t *q;
	histo_t *w;
	histo_t *pk;
} gammaaussLegendreQ;

typedef struct {
	size_t nmu;
	histo_t *muref;
	histo_t *wref;
	histo_t *mu;
	histo_t *w;
} gammaaussLegendreMu;

typedef struct {
	size_t nk;
	histo_t* k;
	histo_t* pk;
} Pk;

typedef struct {
	size_t nk;
	histo_t* k;
	histo_t* pk_lin;
	histo_t* sigma_d2;
	histo_t* gamma1a_1loop;
	histo_t* gamma1a_2loop;
	histo_t* gamma1b_1loop;
	histo_t* gamma1b_2loop;
	histo_t* pkcorr_gamma2_tree_tree;
	histo_t* pkcorr_gamma2_tree_1loop;
	histo_t* pkcorr_gamma2_1loop_1loop;
	histo_t* pkcorr_gamma3_tree;
} Terms2Loop;

typedef struct {
	size_t nk;
	histo_t* k;
	histo_t* pk_lin;
	histo_t* sigma_d2;
	histo_t* gamma1a_1loop;
	histo_t* gamma1b_1loop;
	histo_t* pkcorr_gamma2_tree_tree;
} Terms1Loop;

typedef struct {
	size_t nk;
	histo_t* k;
	histo_t* pk_lin;
	histo_t* sigma_d2;
	histo_t* pk_b2d;
	histo_t* pk_bs2d;
	histo_t* pk_b2t;
	histo_t* pk_bs2t;
	histo_t* pk_b22;
	histo_t* pk_b2s2;
	histo_t* pk_bs22;
	histo_t* sigma3sq;
} TermsBias;

typedef struct {
	size_t nk;
	histo_t* k;
	histo_t* pk_lin;
	histo_t* sigma_d2;
	histo_t* A;
} TermsA;

typedef struct {
	size_t nk;
	histo_t* k;
	histo_t* pk_lin;
	histo_t* sigma_d2;
	histo_t* B;
} TermsB;

Pk pk_lin;

#endif //_REgammaPT_DEFINE_
