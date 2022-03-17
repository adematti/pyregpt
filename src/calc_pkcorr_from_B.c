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

static Pk pk_dt,pk_tt;
static GaussLegendreQ gauss_legendre_q;
static GaussLegendreMu gauss_legendre_mu;
#pragma omp threadprivate(gauss_legendre_q,gauss_legendre_mu)
static const Precision precision_q_default = {.n=600,.min=5e-4,.max=10.,.interpol=POLY};
static Precision precision_q = {.n=600,.min=5e-4,.max=10.,.interpol=POLY};
static const Precision precision_mu_default = {.n=10,.min=-1.,.max=1.,.interpol=POLY};
static Precision precision_mu = {.n=10,.min=-1.,.max=1.,.interpol=POLY};

static void kernel_B(size_t iq,histo_t mu_,histo_t *kernel)
{
	histo_t x[5],mu[5];
	powers(gauss_legendre_q.x[iq],x,5); powers(mu_,mu,5); 
	histo_t xmu = 1.+x[2]-2.*mu[1]*x[1];
	histo_t q = gauss_legendre_q.q[iq];
	histo_t kq = gauss_legendre_q.k*my_sqrt(xmu);

	histo_t pk_dt_q,pk_tt_q,pk_dt_kq,pk_tt_kq;
	find_pk(pk_dt,&q,&pk_dt_q,1,precision_mu.interpol);
	find_pk(pk_dt,&kq,&pk_dt_kq,1,precision_mu.interpol);
	find_pk(pk_tt,&q,&pk_tt_q,1,precision_mu.interpol);
	find_pk(pk_tt,&kq,&pk_tt_kq,1,precision_mu.interpol);

	kernel[0] = x[2] * (mu[2]-1.) / 2. * pk_dt_kq * pk_dt_q; // n,a,b = 1,1,1
	kernel[1] = 3.*x[2] * power(mu[2]-1.,2) / 8. * pk_dt_kq * pk_tt_q; // n,a,b = 1,1,2
	kernel[2] = 3.*x[4] * power(mu[2]-1.,2) / xmu / 8. * pk_tt_kq * pk_dt_q; // n,a,b = 1,2,1
	kernel[3] = 5.*x[4] * power(mu[2]-1.,3) / xmu / 16. * pk_tt_kq * pk_tt_q; // n,a,b = 1,2,2
	
	kernel[4] = x[1] * (x[1]+2.*mu[1]-3.*x[1]*mu[2]) / 2. * pk_dt_kq * pk_dt_q; // n,a,b = 2,1,1
	kernel[5] = - 3.*x[1] * (mu[2]-1.) * (-x[1]-2.*mu[1]+5.*x[1]*mu[2]) / 4. * pk_dt_kq * pk_tt_q; // n,a,b = 2,1,2
	kernel[6] = 3.*x[2] * (mu[2]-1.) * (-2.+x[2]+6.*x[1]*mu[1]-5.*x[2]*mu[2]) / xmu / 4. * pk_tt_kq * pk_dt_q; // n,a,b = 2,2,1
	kernel[7] = - 3.*x[2] * power(mu[2]-1.,2) * (6.-5.*x[2]-30.*x[1]*mu[1]+35.*x[2]*mu[2]) / xmu / 16. * pk_tt_kq * pk_tt_q; // n,a,b = 2,2,2
	
	kernel[8] = x[1] * (4.*mu[1]*(3.-5.*mu[2]) + x[1]*(3.-30.*mu[2]+35.*mu[4])) / 8. * pk_dt_kq * pk_tt_q; // n,a,b = 3,1,2
	kernel[9] = x[1] * (-8.*mu[1] + x[1]*(-12.+36.*mu[2]+12.*x[1]*mu[1]*(3.-5.*mu[2]) + x[2]*(3.-30.*mu[2]+35.*mu[4]))) / xmu / 8. * pk_tt_kq * pk_dt_q; // n,a,b = 3,2,1
	kernel[10] = 3.*x[1] * (mu[2]-1.) * (-8.*mu[1] + x[1]*(-12.+60.*mu[2]+20.*x[1]*mu[1]*(3.-7.*mu[2])+5.*x[2]*(1.-14.*mu[2]+21.*mu[4]))) / xmu / 16. * pk_tt_kq * pk_tt_q; // n,a,b = 3,2,2
	
	kernel[11] = x[1] * (8.*mu[1]*(-3.+5.*mu[2]) - 6.*x[1]*(3.-30.*mu[2]+35.*mu[4]) + 6.*x[2]*mu[1]*(15.-70.*mu[2]+63*mu[4]) + x[3]*(5.-21.*mu[2]*(5.-15.*mu[2]+11.*mu[4]))) / xmu / 16. * pk_tt_kq * pk_tt_q; // n,a,b = 4,2,2
	
	size_t ii;
	for (ii=0;ii<NCOMP;ii++) kernel[ii] /= xmu;
	
}

void set_precision_B_q(size_t n_,histo_t min_,histo_t max_,char* interpol_)
{
	set_precision(&precision_q,n_,min_,max_,interpol_,&precision_q_default);
}

void set_precision_B_mu(size_t n_,char* interpol_)
{
	set_precision(&precision_mu,n_,1.,-1.,interpol_,&precision_mu_default);
}

void init_B(Pk pk_dt_,Pk pk_tt_)
{
	pk_dt = pk_dt_;
	pk_tt = pk_tt_;
	init_gauss_legendre_q(&gauss_legendre_q,&precision_q);
	init_gauss_legendre_mu(&gauss_legendre_mu,&precision_mu);
}

void free_B()
{
	free_gauss_legendre_q(&gauss_legendre_q);
	free_gauss_legendre_mu(&gauss_legendre_mu);
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
	return 1.;
}

void calc_pk_B(histo_t k,histo_t* pk_B)
{
	update_gauss_legendre_q(&gauss_legendre_q,k);
	
	size_t ii,iq,nq=gauss_legendre_q.nq;
	for (ii=0;ii<NCOMP;ii++) pk_B[ii] = 0.;
		
	for (iq=0;iq<nq;iq++) {
	
		histo_t x = gauss_legendre_q.x[iq];
		if (!set_mu_range(x)) continue;
		
		histo_t integ_B[NCOMP]={0.};
		histo_t kernel[NCOMP];
		size_t imu,nmu=gauss_legendre_mu.nmu;	
	
		for (imu=0;imu<nmu;imu++) {
			kernel_B(iq, gauss_legendre_mu.mu[imu], kernel);
			histo_t w = gauss_legendre_mu.w[imu];
			for (ii=0;ii<NCOMP;ii++) integ_B[ii] += kernel[ii]*w;
		}
		histo_t xw = x*gauss_legendre_q.w[iq];
		for (ii=0;ii<NCOMP;ii++) pk_B[ii] += integ_B[ii]*xw; 
	}
	
	histo_t factor = k*k*k / (2.*M_PI*M_PI);
	for (ii=0;ii<NCOMP;ii++) pk_B[ii] *= factor;

}
