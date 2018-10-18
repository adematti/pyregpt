#ifndef _REGPT_PKCORRAB_
#define _REGPT_PKCORRAB_

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
static const size_t INDEX_A[5][2] = {{0,0},{0,1},{1,1},{1,2},{2,2}};
static const size_t INDEX_B[12][3] = {{0,0,0},{0,0,1},{0,1,0},{0,1,1},{1,0,0},{1,0,1},{1,1,0},{1,1,1},{2,0,1},{2,1,0},{2,1,1},{3,1,1}};
static const size_t SHAPE_A[2] = {3,3};
static const size_t SHAPE_B[2] = {4,3};

histo_t kernel_pkcorr_A_B(size_t iq, histo_t mu, histo_t pk_k, size_t n, size_t a, size_t b, kernel_A_B kernel)
{
	histo_t x = gauss_legendre_q.x[iq];
	histo_t k = gauss_legendre_q.k;
	histo_t q = gauss_legendre_q.q[iq];
	histo_t kq = gauss_legendre_q.k*my_sqrt(1.+x*x-2.*mu*x);
	histo_t pk_q = gauss_legendre_q.pk[iq];
	histo_t pk_kq;
	find_pk_lin(&kq,&pk_kq,1,interpol_mu_tree);
	
	return (*kernel)(n,a,b,k,x,kq,mu,pk_k,pk_q,pk_kq);
}


histo_t calc_integ_pkcorr_A_B(size_t iq, histo_t pk_k, size_t n, size_t a, size_t b, kernel_A_B kernel)
{
	histo_t integ_A_B = 0.;
	histo_t x = gauss_legendre_q.x[iq];
	histo_t xmin = gauss_legendre_q.x[0];
	histo_t xmax = gauss_legendre_q.x[gauss_legendre_q.nq-1];
	histo_t mumin = MAX(-1.,(1.+x*x-xmax*xmax)/2./x);
	histo_t mumax = MIN(1.,(1.+x*x-xmin*xmin)/2./x);
	if (x>=0.5) mumax = 0.5/x;
	
	update_gauss_legendre_mu(&gauss_legendre_mu,mumin,mumax);
	//nodes_weights_gauss_legendre(mumin,mumax,gauss_legendre_mu.mu,gauss_legendre_mu.w,gauss_legendre_mu.nmu);
	size_t imu,nmu=gauss_legendre_mu.nmu;
	
	for (imu=0;imu<nmu;imu++) integ_A_B += kernel_pkcorr_A_B(iq, gauss_legendre_mu.mu[imu], pk_k, n, a, b, kernel) * gauss_legendre_mu.w[imu];
	
	return integ_A_B;
}

void set_precision_A_B_q(size_t nq_tree_,char* interpol_q_tree_)
{
	nq_tree = nq_tree_;
	interpol_q_tree = get_interpol(interpol_q_tree_);
}

void set_precision_A_B_mu(size_t nmu_tree_,char* interpol_mu_tree_)
{
	nmu_tree = nmu_tree_;
	interpol_mu_tree = get_interpol(interpol_mu_tree_);
}

void init_A_B()
{
	init_gauss_legendre_q(&gauss_legendre_q,nq_tree,interpol_q_tree,-1,-1);
	init_gauss_legendre_mu(&gauss_legendre_mu,nmu_tree);
}

void free_A_B()
{
	free_gauss_legendre_q(&gauss_legendre_q);
	free_gauss_legendre_mu(&gauss_legendre_mu);
}

void calc_pkcorr_from_A(histo_t k, histo_t pk_k, histo_t* A)
{

	update_gauss_legendre_q(&gauss_legendre_q,k);
	
	size_t ii,iq,nq=gauss_legendre_q.nq;
	
	for (ii=0;ii<SHAPE_A[0]*SHAPE_A[1];ii++) A[ii] = 0.;
	
	for (iq=0;iq<nq;iq++) {
		histo_t xw = gauss_legendre_q.x[iq]*gauss_legendre_q.w[iq];
		for (ii=0;ii<5;ii++) {
			size_t m = INDEX_A[ii][0];
			size_t n = INDEX_A[ii][1];
			A[m*SHAPE_A[1]+n] += calc_integ_pkcorr_A_B(iq,pk_k,m+1,n+1,0,kernel_A) * xw;
		}
	}
	
	for (ii=0;ii<SHAPE_A[0]*SHAPE_A[1];ii++) A[ii] *= k*k*k / (2.*M_PI*M_PI);

}

void calc_pkcorr_from_B(histo_t k, histo_t* B)
{

	update_gauss_legendre_q(&gauss_legendre_q,k);
	
	size_t ii,iq,nq=gauss_legendre_q.nq;
	
	for (ii=0;ii<SHAPE_B[0]*SHAPE_B[1];ii++) B[ii] = 0.;
	
	for (iq=0;iq<nq;iq++) {
		histo_t xw = gauss_legendre_q.x[iq]*gauss_legendre_q.w[iq];
		for (ii=0;ii<12;ii++) {
			size_t n = INDEX_B[ii][0];
			size_t a = INDEX_B[ii][1];
			size_t b = INDEX_B[ii][2];
			B[n*SHAPE_B[1]+a+b] += calc_integ_pkcorr_A_B(iq,0.,n+1,a+1,b+1,kernel_B) * xw;
		}
	}
	
	for (ii=0;ii<SHAPE_B[0]*SHAPE_B[1];ii++) B[ii] *= k*k*k / (2.*M_PI*M_PI);

}


#endif //_PKCORRAB_
