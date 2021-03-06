#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "define.h"
#include "common.h"
#include "kernels.h"

#define NCOMP 5

static GaussLegendreQ gauss_legendre_q;
static GaussLegendreMu gauss_legendre_mu;
#pragma omp threadprivate(gauss_legendre_q,gauss_legendre_mu)
static const Precision precision_q_default = {.n=2000,.min=-1.,.max=-1.,.interpol=POLY};
static Precision precision_q = {.n=2000,.min=-1.,.max=-1.,.interpol=POLY};
static const Precision precision_mu_default = {.n=20,.min=-1.,.max=1.,.interpol=POLY};
static Precision precision_mu = {.n=20,.min=-1.,.max=1.,.interpol=POLY};

void kernel_A_tA_1loop(size_t iq, histo_t mu_, histo_t pk_k, histo_t *kernel)
{
	histo_t x_ = gauss_legendre_q.x[iq];
	
	histo_t x[5],mu[5];
	powers(x_,x,5); powers(mu_,mu,5);
	histo_t xmu = 1.+x[2]-2.*mu[1]*x[1];
	histo_t kq = gauss_legendre_q.k*my_sqrt(xmu);
	
	histo_t pk_q = gauss_legendre_q.pk[iq];
	histo_t pk_kq;
	find_pk_lin(&kq,&pk_kq,1,precision_mu.interpol);
	pk_kq /= xmu*xmu;
	
	histo_t kernel_A[NCOMP],kernel_tA[NCOMP];
	
	kernel_A[0] = -x[3]/7.*(mu[1]+6*mu[3]+x[2]*mu[1]*(-3+10*mu[2])+x[1]*(-3+mu[2]-12*mu[4]));			// m,n = 1,1
	kernel_A[1] = x[4]/14.*(mu[2]-1)*(-1+7*x[1]*mu[1]-6*mu[2]);											// m,n = 1,2
	kernel_A[2] = x[3]/14.*(x[2]*mu[1]*(13-41*mu[2])-4*(mu[1]+6*mu[3])+x[1]*(5+9*mu[2]+42*mu[4]));		// m,n = 2,2
	kernel_A[3] = kernel_A[1];																			// m,n = 2,3
	kernel_A[4] = x[3]/14.*(1-7*x[1]*mu[1]+6*mu[2])*(-2*mu[1]+x[1]*(-1+3*mu[2]));						// m,n = 3,3
	
	kernel_tA[0] = 1./7.*(mu[1]+x[1]-2*x[1]*mu[2])*(3*x[1]+7*mu[1]-10*x[1]*mu[2]);
	kernel_tA[1] = x[1]/14.*(mu[2]-1)*(3*x[1]+7*mu[1]-10*x[1]*mu[2]);
	kernel_tA[2] = 1./14.*(28*mu[2]+x[1]*mu[1]*(25-81*mu[2])+x[2]*(1-27*mu[2]+54*mu[4]));
	kernel_tA[3] = x[1]/14.*(1-mu[2])*(x[1]-7*mu[1]+6*x[1]*mu[2]);
	kernel_tA[4] = 1./14.*(x[1]-7*mu[1]+6*x[1]*mu[2])*(-2*mu[1]-x[1]+3*x[1]*mu[2]);
	
	size_t ii;
	//Taruya 2010 (arXiv 1006.0699v1) eq A3
	for (ii=0;ii<NCOMP;ii++) kernel[ii] = (kernel_A[ii] * pk_k + kernel_tA[ii] * pk_q) * pk_kq;
}

void kernel_a_1loop(size_t iq, histo_t pk_k, histo_t* kernel)
{

	histo_t x_ = gauss_legendre_q.x[iq];
	histo_t pk_q = gauss_legendre_q.pk[iq];
	histo_t x[9];
	powers(x_,x,9);
	
	if (x[1]<1e-4) {
		kernel[0] = 8*x[8]/735 + 24*x[6]/245 - 24*x[4]/35 + 8*x[2]/7 - 2./3;
		kernel[1] = -16*x[8]/8085 - 16*x[6]/735 + 48*x[4]/245 - 16*x[2]/35;
		kernel[2] = 32*x[8]/1617 + 128*x[6]/735 - 288*x[4]/245 + 64*x[2]/35 - 4./3;
		kernel[4] = 24*x[8]/2695 + 8*x[6]/105 - 24*x[4]/49 + 24*x[2]/35 - 2./3;
	}
	else if (x[1]>1e2) {
		kernel[0] = 2./105 - 24/(245*x[2]) - 8/(735*x[4]) - 8/(2695*x[6]) - 8/(7007*x[8]);
		kernel[1] = -16./35 + 48/(245*x[2]) - 16/(735*x[4]) - 16/(8085*x[6]) - 16/(35035*x[8]);
		kernel[2] = -44./105 - 32/(735*x[4]) - 64/(8085*x[6]) - 96/(35035*x[8]);
		kernel[4] = -46./105 + 24/(245*x[2]) - 8/(245*x[4]) - 8/(1617*x[6]) - 8/(5005*x[8]);
	}
	else {
		histo_t logx = 0.;
		if (my_abs(x[1]-1)>EPS) logx = my_log(my_abs((x[1] + 1)/(x[1] - 1)));
		kernel[0] = -1./84./x[1]*(2*x[1]*(19-24*x[2]+9*x[4])-9*power(x[2]-1,3)*logx);
		kernel[1] = 1./112./x[3]*(2*x[1]*(x[2]+1)*(3-14*x[2]+3*x[4])-3*power(x[2]-1,4)*logx);
		kernel[2] = 1./336./x[3]*(2*x[1]*(9-185*x[2]+159*x[4]-63*x[6])+9*power(x[2]-1,3)*(7*x[2]+1)*logx);
		kernel[4] = 1./336./x[3]*(2*x[1]*(9-109*x[2]+63*x[4]-27*x[6])+9*power(x[2]-1,3)*(3*x[2]+1)*logx);
	}
	kernel[3] = kernel[1];
	
	size_t ii;
	//Taruya 2010 (arXiv 1006.0699v1) eq A3
	for (ii=0;ii<NCOMP;ii++) kernel[ii] *= pk_k * pk_q;
}


void set_precision_A_1loop_q(size_t n_,histo_t min_,histo_t max_,char* interpol_)
{
	set_precision(&precision_q,n_,min_,max_,interpol_,&precision_q_default);
}

void set_precision_A_1loop_mu(size_t n_,char* interpol_)
{
	set_precision(&precision_mu,n_,1.,-1.,interpol_,&precision_mu_default);
}

void init_A_1loop()
{
	init_gauss_legendre_q(&gauss_legendre_q,&precision_q);
	init_gauss_legendre_mu(&gauss_legendre_mu,&precision_mu);
}


void free_A_1loop()
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
	update_gauss_legendre_mu(&gauss_legendre_mu,mumin,mumax);
	return 1;
}


void calc_pk_A_1loop(histo_t k, histo_t* pk_A)
{
	update_gauss_legendre_q(&gauss_legendre_q,k);
	histo_t pk_k;
	find_pk_lin(&k,&pk_k,1,precision_q.interpol);
	
	size_t ii,iq,nq=gauss_legendre_q.nq;
	for (ii=0;ii<NCOMP;ii++) pk_A[ii] = 0.;
		
	for (iq=0;iq<nq;iq++) {

		histo_t x = gauss_legendre_q.x[iq];
		if (!set_mu_range(x)) continue;
		
		histo_t integ_A_tA[NCOMP]={0.};
		histo_t kernel_A_tA[NCOMP];
		size_t imu,nmu=gauss_legendre_mu.nmu;	
	
		for (imu=0;imu<nmu;imu++) {
			kernel_A_tA_1loop(iq, gauss_legendre_mu.mu[imu], pk_k, kernel_A_tA);
			histo_t w = gauss_legendre_mu.w[imu];
			for (ii=0;ii<NCOMP;ii++) integ_A_tA[ii] += kernel_A_tA[ii]*w;
		}
		histo_t integ_a[NCOMP];
		kernel_a_1loop(iq, pk_k, integ_a);
		histo_t xw = x*gauss_legendre_q.w[iq]; //xw because integration in dlog(x)
		for (ii=0;ii<NCOMP;ii++) pk_A[ii] += (integ_A_tA[ii]+integ_a[ii])*xw; 
		//for (ii=0;ii<NCOMP;ii++) pk_A[ii] += integ_A_tA[ii]*xw; 
	}
	
	histo_t factor = k*k*k / (4.*M_PI*M_PI);
	for (ii=0;ii<NCOMP;ii++) pk_A[ii] *= factor;

}


/*
void calc_pk_A_1loop(histo_t k, histo_t* pk_A)
{
	update_gauss_legendre_q(&gauss_legendre_q,k);
	histo_t pk_k;
	find_pk_lin(&k,&pk_k,1,precision_q.interpol);
	
	size_t ii,iq,nq=gauss_legendre_q.nq;
	for (ii=0;ii<NCOMP;ii++) pk_A[ii] = 0.;
		
	for (iq=0;iq<nq;iq++) {

		histo_t x = gauss_legendre_q.x[iq];
		if (!set_mu_range(x)) continue;
		
		histo_t integ_A_tA[NCOMP]={0.};
		histo_t kernel_A_tA[NCOMP];
		histo_t kernel_a[NCOMP];
		size_t imu,nmu=gauss_legendre_mu.nmu;	
	
		for (imu=0;imu<nmu;imu++) {
			kernel_A_tA_1loop(iq, gauss_legendre_mu.mu[imu], pk_k, kernel_A_tA);
			kernel_a_1loop(iq, pk_k, kernel_a);
			histo_t w = gauss_legendre_mu.w[imu];
			for (ii=0;ii<NCOMP;ii++) integ_A_tA[ii] += (kernel_A_tA[ii]+kernel_a[ii]/2.)*w;
		}
		histo_t xw = x*gauss_legendre_q.w[iq];
		for (ii=0;ii<NCOMP;ii++) pk_A[ii] += integ_A_tA[ii]*xw; 
	}
	
	histo_t factor = k*k*k / (4.*M_PI*M_PI);
	for (ii=0;ii<NCOMP;ii++) pk_A[ii] *= factor;
}
*/


