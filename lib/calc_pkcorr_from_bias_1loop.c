#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "define.h"
#include "common.h"
#include "kernels.h"

static GaussLegendreQ gauss_legendre_q;
static GaussLegendreMu gauss_legendre_mu;
#pragma omp threadprivate(gauss_legendre_q,gauss_legendre_mu)
static const Precision precision_q_default = {.n=2000,.min=-1.,.max=-1.,.interpol=POLY};
static Precision precision_q = {.n=2000,.min=-1.,.max=-1.,.interpol=POLY};
static const Precision precision_mu_default = {.n=20,.min=-1.,.max=1.,.interpol=POLY};
static Precision precision_mu = {.n=20,.min=-1.,.max=1.,.interpol=POLY};

histo_t kernel_pk_bias_1loop(FLAG a,size_t iq, histo_t mu, kernel_bias_1loop kernel)
{
	histo_t x = gauss_legendre_q.x[iq];
	histo_t k = gauss_legendre_q.k;
	histo_t q = gauss_legendre_q.q[iq];
	histo_t kq = gauss_legendre_q.k*my_sqrt(1.+x*x-2.*mu*x);
	//if ((kq<gauss_legendre_q.q[0])||(kq>gauss_legendre_q.q[gauss_legendre_q.nq-1])) return 0.; //superfluous
	histo_t pk_q = gauss_legendre_q.pk[iq];
	histo_t pk_kq;
	find_pk_lin(&kq,&pk_kq,1,precision_mu.interpol);
	histo_t muqkq = (k*k - kq*kq - q*q)/2./q/kq;
	//if (kq/q>10.) muqkq = 1.;

	return (*kernel)(a,q,kq,mu,muqkq,pk_q,pk_kq);
}


void set_precision_bias_1loop_q(size_t n_,histo_t min_,histo_t max_,char* interpol_)
{
	set_precision(&precision_q,n_,min_,max_,interpol_,&precision_q_default);
}

void set_precision_bias_1loop_mu(size_t n_,char* interpol_)
{
	set_precision(&precision_mu,n_,1.,-1.,interpol_,&precision_mu_default);
}

void init_bias_1loop()
{
	init_gauss_legendre_q(&gauss_legendre_q,&precision_q);
	init_gauss_legendre_mu(&gauss_legendre_mu,&precision_mu);
}

void free_bias_1loop()
{
	free_gauss_legendre_q(&gauss_legendre_q);
	free_gauss_legendre_mu(&gauss_legendre_mu);
}

static _Bool set_mu_range(histo_t x, _Bool run_half)
{
	histo_t xmin = gauss_legendre_q.x[0];
	histo_t xmax = gauss_legendre_q.x[gauss_legendre_q.nq-1];
	histo_t mumin = MAX(-1.,(1.+x*x-xmax*xmax)/2./x);
	histo_t mumax = MIN(1.,(1.+x*x-xmin*xmin)/2./x);
	if ((mumin>=1.)||(mumax<=-1.)||(mumax<=mumin)) return 0;
	if (run_half) {
		if (x>=0.5) mumax = 0.5/x; //symmetric q <-> k-q, shorter and avoids oscillations
		//mumax = MIN(1.,1./2./x);
	}
	update_gauss_legendre_mu(&gauss_legendre_mu,mumin,mumax);
	return 1;
}

histo_t calc_pk_bias_1loop(FLAG a, histo_t k, kernel_bias_1loop kernel, _Bool run_half)
{

	update_gauss_legendre_q(&gauss_legendre_q,k);
	
	size_t iq,nq=gauss_legendre_q.nq;
	histo_t bias = 0.;
	
	for (iq=0;iq<nq;iq++) {
		histo_t x = gauss_legendre_q.x[iq];
		if (!set_mu_range(x,run_half)) continue;
		
		histo_t integ_bias = 0.;
		size_t imu,nmu=gauss_legendre_mu.nmu;
		
		for (imu=0;imu<nmu;imu++) integ_bias += kernel_pk_bias_1loop(a,iq,gauss_legendre_mu.mu[imu],kernel) * gauss_legendre_mu.w[imu];
		
		histo_t x3w = x*x*x*gauss_legendre_q.w[iq];
		if (run_half) x3w *= 2.;
		bias += integ_bias * x3w;
	}

	return k*k*k / (4.*M_PI*M_PI) * bias;

}
