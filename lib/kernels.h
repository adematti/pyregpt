#ifndef _REGPT_KERNELS_
#define _REGPT_KERNELS_

histo_t F2_sym(FLAG a, histo_t p2, histo_t q2, histo_t pq);

histo_t S2(histo_t mu);

histo_t D2(histo_t mu);

histo_t LFunc(histo_t k, histo_t q);

histo_t WFunc(histo_t k1, histo_t k2, histo_t k3, histo_t q);

histo_t betafunc(size_t a,histo_t z);

histo_t small_beta(histo_t k, histo_t q);

histo_t big_beta(histo_t k1,histo_t k2,histo_t k3,histo_t q);

#endif //_REGPT_KERNELS_
