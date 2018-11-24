#ifndef _REgammaPT_KERNELS_
#define _REgammaPT_KERNELS_

histo_t F2_sym_fast(FLAG a, histo_t p2, histo_t q2, histo_t pq);

histo_t F2_sym_full(FLAG a, histo_t *p, histo_t *q);

histo_t F2_sym(FLAG a, histo_t p, histo_t q, histo_t mu);

histo_t S2(histo_t mu);

histo_t D2(histo_t mu);

histo_t LFunc(histo_t k, histo_t q);

histo_t WFunc(histo_t k1, histo_t k2, histo_t k3, histo_t q);

histo_t betafunc(size_t a,histo_t z);

histo_t small_beta(histo_t k, histo_t q);

histo_t big_beta(histo_t k1,histo_t k2,histo_t k3,histo_t q);

histo_t sigmaab(size_t n,size_t a,size_t b);

histo_t F3(FLAG a_, histo_t* p, histo_t* q, histo_t* r);

histo_t F3_sym(FLAG a, histo_t* p, histo_t* q, histo_t* r);

#endif //_REgammaPT_KERNELS_
