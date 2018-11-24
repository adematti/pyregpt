#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include "define.h"
#include "common.h"
#include "kernels.h"

#define NCOMP 12

static gammaaussLegendreQ gauss_legendre_q;
static gammaaussLegendreMu gauss_legendre_mu;
#pragma omp threadprivate(gauss_legendre_q,gauss_legendre_mu)
static const Precision precision_q_default = {.n=300,.min=5e-4,.max=10.,.interpol=POLY};
static Precision precision_q = {.n=300,.min=5e-4,.max=10.,.interpol=POLY};
static const Precision precision_mu_default = {.n=10,.min=-1.,.max=1.,.interpol=POLY};
static Precision precision_mu = {.n=10,.min=-1.,.max=1.,.interpol=POLY};

void kernel_projection_A_tA(histo_t x, histo_t mu, histo_t* kernel_A, histo_t* kernel_tA)
{
	histo_t xmu = (1.+x*x-2.*mu*x);
	
	kernel_A[0] = x*mu;
	kernel_A[1] = - x*x * (3.*x*mu-2.) * (mu*mu-1.) / xmu /2.;
	kernel_A[2] = x*( 2.*mu + x*(2.-6.*mu*mu) + x*x*mu*(-3.+5.*mu*mu) ) / xmu / 2.;
	kernel_A[3] = kernel_A[0];
	kernel_A[4] = kernel_A[1];
	kernel_A[5] = kernel_A[2];
	
	kernel_tA[0] = - x*x * (x*mu-1.) / xmu;
	kernel_tA[1] = x*x * (3.*x*mu-1.) * (mu*mu-1.) / xmu /2.;
	kernel_tA[2] = x*x * (-1. + 3.*x*mu + 3.*mu*mu - 5.*x*mu*mu*mu ) / xmu /2.;
	kernel_tA[3] = kernel_tA[0];
	kernel_tA[4] = kernel_tA[1];
	kernel_tA[5] = kernel_tA[2];
}


static void kernel_A_tA_2loop_I(size_t iq, histo_t mu, histo_t* kernel_A_tA)
{
	histo_t x = gauss_legendre_q.x[iq];
	histo_t k = gauss_legendre_q.k;
	histo_t q = gauss_legendre_q.q[iq];
	histo_t kq = gauss_legendre_q.k*my_sqrt(1.+x*x-2.*mu*x);

	histo_t kernel_projection_A[6],kernel_projection_tA[6];
	kernel_projection_A_tA(x,mu,kernel_projection_A,kernel_projection_tA);
	
	histo_t b211A, b221A, b212A, b222A, b211tA, b221tA, b212tA, b222tA;
	bispectrum_1loop_I(k, q, kq, &b211A, &b221A, &b212A, &b222A, &b211tA, &b221tA, &b212tA, &b222tA);
	
	kernel_A_tA[0] = kernel_projection_A[0] * b211A * x;
	kernel_A_tA[1] = kernel_projection_A[1] * b221A * x;
	kernel_A_tA[2] = kernel_projection_A[2] * b221A * x;
	kernel_A_tA[3] = kernel_projection_A[3] * b212A * x;
	kernel_A_tA[4] = kernel_projection_A[4] * b222A * x;
	kernel_A_tA[5] = kernel_projection_A[5] * b222A * x;

	kernel_A_tA[6] = kernel_projection_tA[0] * b211tA * x;
	kernel_A_tA[7] = kernel_projection_tA[1] * b221tA * x;
	kernel_A_tA[8] = kernel_projection_tA[2] * b221tA * x;
	kernel_A_tA[9] = kernel_projection_tA[3] * b212tA * x;
	kernel_A_tA[10] = kernel_projection_tA[4] * b222tA * x;
	kernel_A_tA[11] = kernel_projection_tA[5] * b222tA * x;
}

void set_precision_A_2loop_I_q(size_t n_,histo_t min_,histo_t max_,char* interpol_)
{
	set_precision(&precision_q,n_,min_,max_,interpol_,&precision_q_default);
}

void set_precision_A_2loop_I_mu(size_t n_,char* interpol_)
{
	set_precision(&precision_mu,n_,1.,-1.,interpol_,&precision_mu_default);
}

void init_A_2loop_I()
{
	init_gauss_legendre_q(&gauss_legendre_q,&precision_q);
	init_gauss_legendre_mu(&gauss_legendre_mu,&precision_mu);
	init_bispectrum_1loop_I();
}

void free_A_2loop_I()
{
	free_gauss_legendre_q(&gauss_legendre_q);
	free_gauss_legendre_mu(&gauss_legendre_mu);
	free_bispectrum_1loop_I();
}

static _Bool set_mu_range(histo_t x)
{
	histo_t xmin = gauss_legendre_q.x[0];
	histo_t xmax = gauss_legendre_q.x[gauss_legendre_q.nq-1];
	histo_t mumin = MAX(-1.,(1.+x*x-xmax*xmax)/2./x);
	histo_t mumax = MIN(1.,(1.+x*x-xmin*xmin)/2./x);
	if ((mumin>=1.)||(mumax<=-1.)||(mumax<=mumin)) return 0;
	if (x>=0.5) mumax = 0.5/x; //symmetric q <-> k-q
	update_gauss_legendre_mu(&gauss_legendre_mu,mumin,mumax);
	return 1;
}

void calc_pkcorr_A_2loop_I(histo_t k,histo_t* pkcorr_A)
{
	update_gauss_legendre_q(&gauss_legendre_q,k);
	
	size_t ii,iq,nq=gauss_legendre_q.nq;
	for (ii=0;ii<NCOMP;ii++) pkcorr_A[ii] = 0.;
		
	for (iq=0;iq<nq;iq++) {
	
		if (!set_mu_range(gauss_legendre_q.x[iq])) continue;
		
		histo_t integ_A_tA[NCOMP]={0.};
		histo_t kernel_A_tA[NCOMP];
		size_t imu,nmu=gauss_legendre_mu.nmu;	
	
		for (imu=0;imu<nmu;imu++) {
			kernel_A_tA_2loop_I(iq, gauss_legendre_mu.mu[imu], kernel_A_tA);
			histo_t w = gauss_legendre_mu.w[imu];
			for (ii=0;ii<NCOMP;ii++) integ_A_tA[ii] += kernel_A_tA[ii]*w;
		}
		histo_t w = gauss_legendre_q.w[iq];
		for (ii=0;ii<NCOMP;ii++) pkcorr_A[ii] += integ_A_tA[ii]*w; 
	}
	
	histo_t factor = k*k*k / (2.*M_PI*M_PI);
	for (ii=0;ii<NCOMP;ii++) pkcorr_A[ii] *= factor;

}
