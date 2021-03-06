#ifndef _REGPT_COMMON_
#define _REGPT_COMMON_

#define EPS 3.e-14

// ios

void set_verbosity(char* mode);

void error_open_file(char *fname);

void print_num_threads();

void print_pk_lin();

void print_k(histo_t *k,size_t nk);

void print_flags(FLAG a,FLAG b);

void set_num_threads(size_t num_threads);

FLAG set_flag(char* flag);

void set_pk_lin(histo_t* k,histo_t* pk,size_t nk);

// usual functions

histo_t my_abs(histo_t x);

histo_t my_log(histo_t x);

histo_t my_exp(histo_t x);

histo_t my_pow(histo_t x,histo_t exp);

histo_t my_sqrt(histo_t x);

histo_t my_sin(histo_t x);

histo_t my_cos(histo_t x);

histo_t power(histo_t x,const size_t n);

void powers(histo_t x,histo_t *tab, const size_t npowers);

// interpolation

size_t get_dichotomy_index(histo_t x,histo_t *tab,size_t min,size_t max);

histo_t interpol_lin(histo_t x,histo_t xmin,histo_t xmax,histo_t ymin,histo_t ymax);

histo_t interpol_poly(histo_t x,histo_t *tabx,histo_t *taby,size_t nx);

void find_pk(Pk pkin,histo_t* k,histo_t* pk,size_t nk,INTERPOL interpol);

void find_pk_lin(histo_t* k,histo_t* pk,size_t nk,INTERPOL interpol);

void calc_running_sigmad2(histo_t *k,histo_t *sigmad2,size_t nk,histo_t uvcutoff);

// gauss-legendre integration

void nodes_weights_gauss_legendre(histo_t xmin,histo_t xmax,histo_t *x,histo_t *w,size_t n);

void init_gauss_legendre_q(GaussLegendreQ *gauss_legendre,Precision *precision);

void update_gauss_legendre_q(GaussLegendreQ *gauss_legendre,histo_t k);

void free_gauss_legendre_q(GaussLegendreQ *gauss_legendre);

void init_gauss_legendre_mu(GaussLegendreMu *gauss_legendre,Precision *precision);

void update_gauss_legendre_mu(GaussLegendreMu *gauss_legendre,histo_t mumin,histo_t mumax);

void free_gauss_legendre_mu(GaussLegendreMu *gauss_legendre);

// precision setters

_Bool set_interpol(INTERPOL *current,char *newi);

_Bool set_precision(Precision *precision,size_t n,histo_t min,histo_t max,char *interpol,const Precision *precision_default);

// gamma1

void set_precision_gamma1_1loop(size_t n_,histo_t min_,histo_t max_,char* interpol_);

void init_gamma1_1loop();

void free_gamma1_1loop();

histo_t gamma1_1loop(FLAG a,histo_t k);

void set_precision_gamma1_2loop(size_t n_,histo_t min_,histo_t max_,char* interpol_);

void init_gamma1_2loop();

void free_gamma1_2loop();

histo_t gamma1_2loop(FLAG a,histo_t k);

void calc_pk_gamma1_2loop(FLAG a,FLAG b,histo_t k,histo_t *gamma1a_1loop,histo_t *gamma1a_2loop,histo_t *gamma1b_1loop,histo_t *gamma1b_2loop);

// gamma2

void set_precision_gamma2_tree_q(size_t n_,histo_t min_,histo_t max_,char* interpol_);

void set_precision_gamma2_tree_mu(size_t n_,char* interpol_);

void init_gamma2_tree();

void free_gamma2_tree();

histo_t gamma2_tree(FLAG a, histo_t k1, histo_t k2, histo_t k3);

void calc_pk_gamma2_tree(FLAG a,FLAG b,histo_t k,histo_t *pk_gamma2_tree_tree);

void set_precision_gamma2d_1loop(size_t n_,histo_t min_,histo_t max_,char* interpol_);

void init_gamma2d_1loop();

void free_gamma2d_1loop();

histo_t gamma2d_1loop(histo_t k1, histo_t k2, histo_t k3);

void set_precision_gamma2t_1loop(size_t n_,histo_t min_,histo_t max_,char* interpol_);

void init_gamma2t_1loop();

void free_gamma2t_1loop();

histo_t gamma2t_1loop(histo_t k1, histo_t k2, histo_t k3);

void calc_pk_gamma2_1loop(FLAG a,FLAG b,histo_t k,histo_t *pk_gamma2_tree_tree,histo_t *pk_gamma2_tree_1loop,histo_t *pk_gamma2_1loop_1loop);

// gamma3

void set_precision_gamma3_tree(histo_t min_,histo_t max_,char* interpol_);

void init_gamma3_tree();

void free_gamma3_tree();

void calc_pk_gamma3_tree(FLAG a_, FLAG b_, histo_t k_, histo_t *pk_gamma3_tree);

// bias

void set_precision_bias_1loop_q(size_t n_,histo_t min_,histo_t max_,char* interpol_);

void set_precision_bias_1loop_mu(size_t n_,char* interpol_);

void init_bias_1loop();

void free_bias_1loop();

typedef histo_t (kernel_bias_1loop)(FLAG a, histo_t q, histo_t kq, histo_t mu, histo_t mukkq, histo_t pk_q, histo_t pk_kq); 

histo_t calc_pk_bias_1loop(FLAG a, histo_t k, kernel_bias_1loop kernel, _Bool run_half);

// A 1 loop

void set_precision_A_1loop_q(size_t n_,histo_t min_,histo_t max_,char* interpol_);

void set_precision_A_1loop_mu(size_t n_,char* interpol_);

void init_A_1loop();

void free_A_1loop();

void calc_pk_A_1loop(histo_t k,histo_t* pk_A);

// A 2 loop

void kernel_projection_A_tA(histo_t x, histo_t mu, histo_t* kernel_A, histo_t* kernel_tA);

void set_precision_A_2loop_I_q(size_t n_,histo_t min_,histo_t max_,char* interpol_);

void set_precision_A_2loop_I_mu(size_t n_,char* interpol_);

void init_A_2loop_I();

void free_A_2loop_I();

void calc_pk_A_2loop_I(histo_t k,histo_t* pk_A);

void set_precision_A_2loop_II_III(histo_t min_,histo_t max_,char* interpol_);

void init_A_2loop_II_III();

void free_A_2loop_II_III();

void calc_pk_A_2loop_II_III(histo_t k_,histo_t* pk_A);

// B

void set_precision_B_q(size_t n_,histo_t min_,histo_t max_,char* interpol_);

void set_precision_B_mu(size_t n_,char* interpol_);

void init_B(Pk pk_dt_,Pk pk_tt_);

void free_B();

void calc_pk_B(histo_t k,histo_t* pk_B);

// spectrum 1 loop

void set_spectrum_1loop(size_t nk,histo_t* k,histo_t* pk);

void run_spectrum_1loop(char* a_,char* b_,size_t num_threads);

// bispectrum 1 loop

void set_precision_bispectrum_1loop_pk_lin(char* interpol_);

void set_running_uvcutoff_bispectrum_1loop(histo_t uvcutoff_);

void get_precision_bispectrum_1loop_pk_lin(INTERPOL *interpol_);

void init_bispectrum_1loop_I();

void free_bispectrum_1loop_I();

void get_running_uvcutoff_bispectrum_1loop(histo_t *uvcutoff_);

void bispectrum_1loop_I(histo_t k, histo_t p, histo_t kp, histo_t* b211A, histo_t* b221A, histo_t* b212A, histo_t* b222A, histo_t* b211tA, histo_t* b221tA, histo_t* b212tA, histo_t* b222tA);

void bispectrum_1loop_II_III(histo_t* kk1,histo_t* kk2,histo_t* kk3,histo_t* qq,FLAG a,FLAG b,FLAG c, histo_t *bk222_abc, histo_t *bk321_abc);

/*
// A and B pertub

void set_precision_A_B_q(size_t n_,histo_t min_,histo_t max_,char* interpol_);

void set_precision_A_B_mu(size_t n_,char* interpol_);

void calc_pk_A(histo_t k, histo_t pk_k, histo_t* A);

void calc_pk_B(histo_t k, histo_t* B);
*/

// IOs

void timer(size_t i);

void write_gauss_legendre_q(GaussLegendreQ gauss_legendre,char *fn);

void write_gauss_legendre_mu(GaussLegendreMu gauss_legendre,char *fn);

#endif //_REGPT_COMMON_
