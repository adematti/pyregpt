#ifndef _REGPT_PKCORRGAMMA2_
#define _REGPT_PKCORRGAMMA2_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "define.h"
#include "common.h"
#include "kernels.h"

static GaussLegendreQ gauss_legendre_q;
static GaussLegendreMu gauss_legendre_mu;
#pragma omp threadprivate(gauss_legendre_q,gauss_legendre_mu)
static Precision precision_q = {.n=200,.min=5e-4,.max=10.,.interpol=POLY};
static Precision precision_mu = {.n=10,.min=-1.,.max=1.,.interpol=POLY};

histo_t Gamma2_tree(FLAG a, histo_t k1, histo_t k2, histo_t k3)
{
	histo_t k12 = k1*k1;
	histo_t k22 = k2*k2;
	histo_t k32 = k3*k3;
	histo_t k1dk2 = (k32 - k12 - k22)/2.;
	return F2_sym(a,k12,k22,k1dk2);
}

histo_t Gamma2_stdPT(FLAG a, size_t n_loop, histo_t k1, histo_t k2, histo_t k3)
{
	if (n_loop==0) return Gamma2_tree(a, k1, k2, k3);
	else if (n_loop==1) {
		if (a==DELTA) return one_loop_Gamma2d(k1, k2, k3);
		return one_loop_Gamma2v(k1, k2, k3);
	}
	return 0.;
}

void kernel_pkcorr_G2(FLAG a,FLAG b,size_t iq, histo_t mu, histo_t *kernel1, histo_t *kernel2, histo_t *kernel3)
{
	histo_t x = gauss_legendre_q.x[iq];
	histo_t k = gauss_legendre_q.k;
	histo_t q = gauss_legendre_q.q[iq];
	histo_t kq = gauss_legendre_q.k*my_sqrt(1.+x*x-2.*mu*x);
	histo_t pk_q = gauss_legendre_q.pk[iq];
	histo_t pk_kq;
	find_pk_lin(&kq,&pk_kq,1,precision_mu.interpol);
	histo_t G2a_tree,G2a_1loop,G2b_tree,G2b_1loop;
	
	if (a==b) {
		G2a_tree = Gamma2_stdPT(a, 0, kq, q, k);
		G2a_1loop = Gamma2_stdPT(a, 1, kq, q, k); 
		G2b_tree = G2a_tree;
		G2b_1loop = G2a_1loop;
	}
	else {
		G2a_tree = Gamma2_stdPT(a, 0, kq, q, k);
		G2a_1loop = Gamma2_stdPT(a, 1, kq, q, k);
		G2b_tree = Gamma2_stdPT(b, 0, kq, q, k);
		G2b_1loop = Gamma2_stdPT(b, 1, kq, q, k);
	}
	*kernel1 = 2. * pk_q * pk_kq * G2a_tree * G2b_tree;
	*kernel2 = 2. * pk_q * pk_kq * ( G2a_tree * G2b_1loop + G2a_1loop * G2b_tree );
	*kernel3 = 2. * pk_q * pk_kq * G2a_1loop * G2b_1loop;        
}


void calc_integ_pkcorr_G2(FLAG a,FLAG b,size_t iq,histo_t *integ_pkcorr1,histo_t *integ_pkcorr2,histo_t *integ_pkcorr3)
{
	*integ_pkcorr1 = 0.;
	*integ_pkcorr2 = 0.;
	*integ_pkcorr3 = 0.;
	histo_t x = gauss_legendre_q.x[iq];
	histo_t xmin = gauss_legendre_q.x[0];
	histo_t xmax = gauss_legendre_q.x[gauss_legendre_q.nq-1];
	histo_t mumin = MAX(-1.,(1.+x*x-xmax*xmax)/2./x);
	histo_t mumax = MIN(1.,(1.+x*x-xmin*xmin)/2./x);
	if ((mumin>=1.)||(mumax<=-1.)||(mumax<=mumin)) return;
	if (x>=0.5) mumax = 0.5/x; //symmetric q <-> k-q
	
	update_gauss_legendre_mu(&gauss_legendre_mu,mumin,mumax);
	//nodes_weights_gauss_legendre(mumin,mumax,gauss_legendre_mu.mu,gauss_legendre_mu.w,gauss_legendre_mu.nmu);
	size_t imu,nmu=gauss_legendre_mu.nmu;	
	
	for (imu=0;imu<nmu;imu++) {
		histo_t w = gauss_legendre_mu.w[imu];
		histo_t kernel1,kernel2,kernel3;
		kernel_pkcorr_G2(a, b, iq, gauss_legendre_mu.mu[imu], &kernel1, &kernel2, &kernel3);
		*integ_pkcorr1 += kernel1*w;
		*integ_pkcorr2 += kernel2*w;
		*integ_pkcorr3 += kernel3*w;
	}
	
}

void set_precision_gamma2_q(size_t n_,histo_t min_,histo_t max_,char* interpol_)
{
	set_precision(&precision_q,n_,min_,max_,interpol_);
}

void set_precision_gamma2_mu(size_t n_,char* interpol_)
{
	set_precision(&precision_mu,n_,-1.,1.,interpol_);
}

void init_gamma2()
{
	init_gauss_legendre_q(&gauss_legendre_q,&precision_q);
	init_gauss_legendre_mu(&gauss_legendre_mu,&precision_mu);
}

void free_gamma2()
{
	free_gauss_legendre_q(&gauss_legendre_q);
	free_gauss_legendre_mu(&gauss_legendre_mu);
}

void calc_pkcorr_from_gamma2(FLAG a,FLAG b,histo_t k,histo_t *pkcorr_G2_tree_tree,histo_t *pkcorr_G2_tree_1loop,histo_t *pkcorr_G2_1loop_1loop)
{
	update_gauss_legendre_q(&gauss_legendre_q,k);
	
	size_t iq,nq=gauss_legendre_q.nq;
	*pkcorr_G2_tree_tree = 0;
	*pkcorr_G2_tree_1loop = 0.;
	*pkcorr_G2_1loop_1loop = 0.;
	histo_t integ_pkcorr1,integ_pkcorr2,integ_pkcorr3;
	
	for (iq=0;iq<nq;iq++) {
		histo_t x = gauss_legendre_q.x[iq];
		histo_t x3w = x*x*x*gauss_legendre_q.w[iq];
		calc_integ_pkcorr_G2(a, b, iq, &integ_pkcorr1, &integ_pkcorr2, &integ_pkcorr3);
		printf("%.3f ",integ_pkcorr1);
 		*pkcorr_G2_tree_tree += integ_pkcorr1 * x3w;
		*pkcorr_G2_tree_1loop += integ_pkcorr2 * x3w;
		*pkcorr_G2_1loop_1loop += integ_pkcorr3 * x3w;
	}

	histo_t factor = k*k*k / (2.*M_PI*M_PI);
	*pkcorr_G2_tree_tree *= factor;
	*pkcorr_G2_tree_1loop *= factor;
	*pkcorr_G2_1loop_1loop *= factor;

}

#endif //_REGPT_PKCORRGAMMA2_
