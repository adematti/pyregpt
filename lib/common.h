#ifndef _REGPT_COMMON_
#define _REGPT_COMMON_

#define EPS 3.e-14

inline histo_t my_abs(histo_t x);

histo_t my_log(histo_t x);

histo_t my_exp(histo_t x);

histo_t my_pow(histo_t x,histo_t exp);

histo_t my_sqrt(histo_t x);

histo_t my_sin(histo_t x);

histo_t my_cos(histo_t x);

histo_t power(histo_t x,const size_t n);

void powers(histo_t x,histo_t *tab, const size_t npowers);

void set_interpol(INTERPOL *current,char *new);

void set_precision(Precision *precision,size_t n,histo_t min,histo_t max,char *interpol);

size_t get_dichotomy_index(histo_t x,histo_t *tab,size_t min,size_t max);

histo_t interpol_lin(histo_t x,histo_t xmin,histo_t xmax,histo_t ymin,histo_t ymax);

histo_t interpol_poly(histo_t x,histo_t *tabx,histo_t *taby,size_t nx);

//void interpol_pk_lin(histo_t* k,histo_t* pk,size_t nk);

histo_t extrapol_pk_lin(histo_t k);

void find_pk_lin(histo_t* k,histo_t* pk,size_t nk,INTERPOL interpol);

void nodes_weights_gauss_legendre(histo_t xmin,histo_t xmax,histo_t *x,histo_t *w,size_t n);

void init_gauss_legendre_q(GaussLegendreQ *gauss_legendre,Precision *precision);

void update_gauss_legendre_q(GaussLegendreQ *gauss_legendre,histo_t k);

void free_gauss_legendre_q(GaussLegendreQ *gauss_legendre);

void init_gauss_legendre_mu(GaussLegendreMu *gauss_legendre,Precision *precision);

void update_gauss_legendre_mu(GaussLegendreMu *gauss_legendre,histo_t mumin,histo_t mumax);

void free_gauss_legendre_mu(GaussLegendreMu *gauss_legendre);

void set_precision_gamma1_1loop(size_t n_,histo_t min_,histo_t max_,char* interpol_);

void set_precision_gamma1_2loop(size_t n_,histo_t min_,histo_t max_,char* interpol_);

void init_gamma1();

void free_gamma1();

void calc_pkcorr_from_gamma1(FLAG a,FLAG b,histo_t k,histo_t *G1a_1loop,histo_t *G1a_2loop,histo_t *G1b_1loop,histo_t *G1b_2loop);

void set_precision_gamma2_q(size_t n_,histo_t min_,histo_t max_,char* interpol_);

void set_precision_gamma2_mu(size_t n_,char* interpol_);

void init_gamma2();

void free_gamma2();

void calc_pkcorr_from_gamma2(FLAG a,FLAG b,histo_t k,histo_t *pkcorr_G2_tree_tree,histo_t *pkcorr_G2_tree_1loop,histo_t *pkcorr_G2_1loop_1loop);

void set_precision_gamma2d(size_t n_,histo_t min_,histo_t max_,char* interpol_);

void init_gamma2d();

void free_gamma2d();

histo_t one_loop_Gamma2d(histo_t k1, histo_t k2, histo_t k3);

void set_precision_gamma2v(size_t n_,histo_t min_,histo_t max_,char* interpol_);

void init_gamma2v();

void free_gamma2v();

histo_t one_loop_Gamma2v(histo_t k1, histo_t k2, histo_t k3);

histo_t LFunc(histo_t k, histo_t q);

histo_t WFunc(histo_t k1, histo_t k2, histo_t k3, histo_t q);

histo_t betafunc(size_t i,histo_t z);

histo_t small_beta(histo_t k, histo_t q);

histo_t big_beta(histo_t k1,histo_t k2,histo_t k3,histo_t q);

void set_precision_gamma3(histo_t min_,histo_t max_,char* interpol_);

void init_gamma3();

void free_gamma3();

void calc_pkcorr_from_gamma3(FLAG a_, FLAG b_, histo_t k_, histo_t *pkcorr_G3_tree);

void set_precision_bias_q(size_t n_,histo_t min_,histo_t max_,char* interpol_);

void set_precision_bias_mu(size_t n_,char* interpol_);

histo_t calc_pkcorr_from_bias(FLAG a, histo_t k, kernel_bias kernel, _Bool run_half);

void set_precision_A_B_q(size_t n_,histo_t min_,histo_t max_,char* interpol_);

void set_precision_A_B_mu(size_t n_,char* interpol_);

void calc_pkcorr_from_A(histo_t k, histo_t pk_k, histo_t* A);

void calc_pkcorr_from_B(histo_t k, histo_t* B);

//IOs

void timer(size_t i);

void write_gauss_legendre_q(GaussLegendreQ gauss_legendre,char *fn);

void write_gauss_legendre_mu(GaussLegendreMu gauss_legendre,char *fn);

void write_2loop(Terms2Loop terms_2loop,char *fn);

#endif
