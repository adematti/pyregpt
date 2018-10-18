#ifndef _REGPT_PKCORRBIAS_
#define _REGPT_PKCORRBIAS_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "define.h"
#include "common.h"
#include "kernels.h"

static GaussLegendreQ gauss_legendre_q;
static GaussLegendreMu gauss_legendre_mu;
#pragma omp threadprivate(gauss_legendre_q,gauss_legendre_mu)
static size_t nq_tree = 600;
static size_t nmu_tree = 100;
static INTERPOL interpol_q_tree = POLY;
static INTERPOL interpol_mu_tree = POLY;

histo_t kernel_pkcorr_bias(FLAG a,size_t iq, histo_t mu, kernel_bias kernel)
{
	histo_t x = gauss_legendre_q.x[iq];
	histo_t k = gauss_legendre_q.k;
	histo_t q = gauss_legendre_q.q[iq];
	histo_t kq = gauss_legendre_q.k*my_sqrt(1.+x*x-2.*mu*x);
	histo_t pk_q = gauss_legendre_q.pk[iq];
	histo_t pk_kq;
	find_pk_lin(&kq,&pk_kq,1,interpol_mu_tree);
	histo_t mukkq = (k*k - kq*kq - q*q)/2./q/kq;
	
	return (*kernel)(a,q,kq,mu,mukkq,pk_q,pk_kq); 
}


histo_t calc_integ_pkcorr_bias(FLAG a,size_t iq, kernel_bias kernel)
{
	histo_t integ_bias = 0.;
	histo_t x = gauss_legendre_q.x[iq];
	histo_t xmin = gauss_legendre_q.x[0];
	histo_t xmax = gauss_legendre_q.x[gauss_legendre_q.nq-1];
	histo_t mumin = MAX(-1.,(1.+x*x-xmax*xmax)/2./x);
	histo_t mumax = MIN(1.,(1.+x*x-xmin*xmin)/2./x);
	if (x>=0.5) mumax = 0.5/x;
	
	update_gauss_legendre_mu(&gauss_legendre_mu,mumin,mumax);
	//nodes_weights_gauss_legendre(mumin,mumax,gauss_legendre_mu.mu,gauss_legendre_mu.w,gauss_legendre_mu.nmu);
	size_t imu,nmu=gauss_legendre_mu.nmu;
	
	for (imu=0;imu<nmu;imu++) integ_bias += kernel_pkcorr_bias(a, iq, gauss_legendre_mu.mu[imu], kernel) * gauss_legendre_mu.w[imu];
	
	return integ_bias;
}

void set_precision_bias_q(size_t nq_tree_,char* interpol_q_tree_)
{
	nq_tree = nq_tree_;
	interpol_q_tree = get_interpol(interpol_q_tree_);
}

void set_precision_bias_mu(size_t nmu_tree_,char* interpol_mu_tree_)
{
	nmu_tree = nmu_tree_;
	interpol_mu_tree = get_interpol(interpol_mu_tree_);
}

void init_bias()
{
	init_gauss_legendre_q(&gauss_legendre_q,nq_tree,interpol_q_tree,-1,-1);
	init_gauss_legendre_mu(&gauss_legendre_mu,nmu_tree);
}

void free_bias()
{
	free_gauss_legendre_q(&gauss_legendre_q);
	free_gauss_legendre_mu(&gauss_legendre_mu);
}

histo_t calc_pkcorr_from_bias(FLAG a, histo_t k, kernel_bias kernel)
{

	update_gauss_legendre_q(&gauss_legendre_q,k);
	
	size_t iq,nq=gauss_legendre_q.nq;
	//histo_t integ_b2d,integ_bs2d,integ_b2d,integ_bs2d,integ_b22,integ_b2s2,integ_bs22,integ_sigma3sq;
	histo_t integ_bias = 0.;
	
	for (iq=0;iq<nq;iq++) {
		histo_t x = gauss_legendre_q.x[iq];
		histo_t x3w = x*x*x*gauss_legendre_q.w[iq];
		integ_bias += calc_integ_pkcorr_bias(a,iq,kernel) * x3w;
	}

	return k*k*k / (2.*M_PI*M_PI) * integ_bias;
}


#endif //_PKCORRBIAS_
