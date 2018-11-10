#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include "define.h"
#include "common.h"
#include "kernels.h"

#define STEP_VERBOSE 10

static INTERPOL interpol_pk_lin = POLY;
static histo_t uvcutoff = 0.5;
static TermsBias terms_bias;


histo_t kernel_b2(FLAG a, histo_t q, histo_t kq, histo_t mu, histo_t mukkq, histo_t pk_q, histo_t pk_kq)
{
	return pk_q * pk_kq * F2_sym_fast(a,q*q,kq*kq,mukkq*q*kq);
}

histo_t kernel_bs2(FLAG a, histo_t q, histo_t kq, histo_t mu, histo_t mukkq, histo_t pk_q, histo_t pk_kq)
{
	return pk_q * pk_kq * F2_sym_fast(a,q*q,kq*kq,mukkq*q*kq) * S2(mukkq);
}

histo_t kernel_b22(FLAG a, histo_t q, histo_t kq, histo_t mu, histo_t mukkq, histo_t pk_q, histo_t pk_kq)
{
	return 0.5 * pk_q * (pk_kq - pk_q);
}

histo_t kernel_b2s2(FLAG a, histo_t q, histo_t kq, histo_t mu, histo_t mukkq, histo_t pk_q, histo_t pk_kq)
{
	return -0.5 * pk_q * (2./3. * pk_q - pk_kq * S2(mukkq));
}

histo_t kernel_bs22(FLAG a, histo_t q, histo_t kq, histo_t mu, histo_t mukkq, histo_t pk_q, histo_t pk_kq)
{
	histo_t s2 = S2(mukkq);
	return -0.5 * pk_q * (4./9. * pk_q - pk_kq * s2*s2);
}

histo_t kernel_sigma3sq(FLAG a, histo_t q, histo_t kq, histo_t mu, histo_t mukkq, histo_t pk_q, histo_t pk_kq)
{
	return 105./16. * pk_q * (D2(mu)*S2(mukkq) + 8./63.);
}


void set_precision_bias_pk_lin(char* interpol_)
{
	set_interpol(&interpol_pk_lin,interpol_);
}

void set_running_uvcutoff_bias(histo_t uvcutoff_)
{
	uvcutoff = uvcutoff_;
}


void set_terms_bias(size_t nk,histo_t* k,histo_t* pk_lin,histo_t* sigma_v2,histo_t* pkbias_b2d,histo_t* pkbias_bs2d,histo_t* pkbias_b2t,histo_t* pkbias_bs2t,histo_t* pkbias_b22,histo_t* pkbias_b2s2,histo_t* pkbias_bs22,histo_t* sigma3sq)
{
	terms_bias.nk = nk;
	terms_bias.k = k;
	terms_bias.pk_lin = pk_lin;
	terms_bias.sigma_v2 = sigma_v2;
	terms_bias.pkbias_b2d = pkbias_b2d;
	terms_bias.pkbias_bs2d = pkbias_bs2d;
	terms_bias.pkbias_b2t = pkbias_b2t;
	terms_bias.pkbias_bs2t = pkbias_bs2t;
	terms_bias.pkbias_b22 = pkbias_b22;
	terms_bias.pkbias_b2s2 = pkbias_b2s2;
	terms_bias.pkbias_bs22 = pkbias_bs22;
	terms_bias.sigma3sq = sigma3sq;
	
#ifdef _VERBOSE
	print_k(terms_bias.k,nk);
	printf("\n");
#endif //_VERBOSE
}

void run_terms_bias(size_t num_threads)
{

#ifdef _VERBOSE
	printf("*** Calculation of bias terms\n");
#endif //_VERBOSE
	set_num_threads(num_threads);
	size_t ik=0;
	size_t step_verbose = MAX(terms_bias.nk*STEP_VERBOSE/100,1);

	timer(0);
	find_pk_lin(terms_bias.k,terms_bias.pk_lin,terms_bias.nk,interpol_pk_lin);
	calc_running_sigma_v2(terms_bias.k,terms_bias.sigma_v2,terms_bias.nk,uvcutoff);
#pragma omp parallel default(none) shared(terms_bias,step_verbose) private(ik)
	{
		init_bias();
#pragma omp for nowait schedule(dynamic)
		for (ik=0;ik<terms_bias.nk;ik++) {
			histo_t k = terms_bias.k[ik];
#ifdef _VERBOSE
			if (ik % step_verbose == 0) printf(" - Computation done at %zu percent.\n",ik*STEP_VERBOSE/step_verbose);
#endif //_VERBOSE
			terms_bias.pkbias_b2d[ik] = calc_pkcorr_from_bias(DELTA,k,kernel_b2,1);
			//terms_bias.pkbias_b2d[ik] = calc_pkcorr_from_bias(DELTA,k,kernel_b2,0);
			terms_bias.pkbias_bs2d[ik] = calc_pkcorr_from_bias(DELTA,k,kernel_bs2,1);
			terms_bias.pkbias_b2t[ik] = calc_pkcorr_from_bias(THETA,k,kernel_b2,1);
			terms_bias.pkbias_bs2t[ik] = calc_pkcorr_from_bias(THETA,k,kernel_bs2,1);
			terms_bias.pkbias_b22[ik] = calc_pkcorr_from_bias(DELTA,k,kernel_b22,0);
			terms_bias.pkbias_b2s2[ik] = calc_pkcorr_from_bias(DELTA,k,kernel_b2s2,0);
			terms_bias.pkbias_bs22[ik] = calc_pkcorr_from_bias(DELTA,k,kernel_bs22,0);
			terms_bias.sigma3sq[ik] = calc_pkcorr_from_bias(DELTA,k,kernel_sigma3sq,0);
		}
#pragma omp critical
		{
			free_bias();
		}
	}
	
#ifdef _VERBOSE
	timer(1);
#endif //_VERBOSE
}
