#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "define.h"
#include "common.h"
#include "kernels.h"

static GaussLegendreQ gauss_legendre_q;
static GaussLegendreMu gauss_legendre_mu;
#pragma omp threadprivate(gauss_legendre_q,gauss_legendre_mu)
static Precision precision_q = {.n=600,.min=-1.,.max=-1.,.interpol=POLY};
static Precision precision_mu = {.n=300,.min=-1.,.max=1.,.interpol=POLY};
static const size_t INDEX_A[5][2] = {{0,0},{0,1},{1,1},{1,2},{2,2}};
static const size_t INDEX_B[12][3] = {{0,0,0},{0,0,1},{0,1,0},{0,1,1},{1,0,0},{1,0,1},{1,1,0},{1,1,1},{2,0,1},{2,1,0},{2,1,1},{3,1,1}};
static const size_t SHAPE_A[2] = {3,3};
static const size_t SHAPE_B[2] = {4,3};



histo_t A_mat(size_t m,size_t n,histo_t* r,histo_t* x)
{
	if ((m==1)&&(n==1)) return -r[3]/7.*(x[1]+6*x[3]+r[2]*x[1]*(-3+10*x[2])+r[1]*(-3+x[2]-12*x[4]));
	if (((m==1)&&(n==2))||((m==2)&&(n==3))) return r[4]/14.*(x[2]-1)*(-1+7*r[1]*x[1]-6*x[2]);
	if ((m==2)&&(n==2)) return r[3]/14.*(r[2]*x[1]*(13-41*x[2])-4*(x[1]+6*x[3])+r[1]*(5+9*x[2]+42*x[4]));
	if ((m==3)&&(n==3)) return r[3]/14.*(1-7*r[1]*x[1]+6*x[2])*(-2*x[1]+r[1]*(-1+3*x[2]));
	return 0.;
}

histo_t Atilde_mat(size_t m,size_t n,histo_t* r,histo_t* x)
{
	if ((m==1)&&(n==1)) return 1./7.*(x[1]+r[1]-2*r[1]*x[2])*(3*r[1]+7*x[1]-10*r[1]*x[2]);
	if ((m==1)&&(n==2)) return r[1]/14.*(x[2]-1)*(3*r[1]+7*x[1]-10*r[1]*x[2]);
	if ((m==2)&&(n==2)) return 1./14.*(28*x[2]+r[1]*x[1]*(25-81*x[2])+r[2]*(1-27*x[2]+54*x[4]));
	if ((m==2)&&(n==3)) return r[1]/14.*(1-x[2])*(r[1]-7*x[1]+6*r[1]*x[2]);
	if ((m==3)&&(n==3)) return 1./14.*(r[1]-7*x[1]+6*r[1]*x[2])*(-2*x[1]-r[1]+3*r[1]*x[2]);
	return 0.;
}
/*
histo_t a_mat(size_t m,size_t n,histo_t* r)
{
	histo_t logr = my_log(my_abs((r[1] + 1)/(r[1] - 1)));
	
	if ((m==1)&&(n==1)) return -1./84./r[1]*(2*r[1]*(19-24*r[2]+9*r[4])-9*power(r[2]-1,3)*logr);
	if (((m==1)&&(n==2))||((m==2)&&(n==3))) return 1./112./r[3]*(2*r[1]*(r[2]+1)*(3-14*r[2]+3*r[4])-3*power(r[2]-1,4)*logr);
	if ((m==2)&&(n==2)) return 1./336./r[3]*(2*r[1]*(9-185*r[2]+159*r[4]-63*r[6])+9*power(r[2]-1,3)*(7*r[2]+1)*logr);
	if ((m==3)&&(n==3)) return 1./336./r[3]*(2*r[1]*(9-109*r[2]+63*r[4]-27*r[6])+9*power(r[2]-1,3)*(3*r[2]+1)*logr);

	return 0.;
}
*/

histo_t a_mat(size_t m,size_t n,histo_t* r)
{
	if (r[1]<1e-4) {
		if ((m==1)&&(n==1)) return 8*r[8]/735 + 24*r[6]/245 - 24*r[4]/35 + 8*r[2]/7 - 2./3;
		if (((m==1)&&(n==2))||((m==2)&&(n==3))) return -16*r[8]/8085 - 16*r[6]/735 + 48*r[4]/245 - 16*r[2]/35;
		if ((m==2)&&(n==2)) return 32*r[8]/1617 + 128*r[6]/735 - 288*r[4]/245 + 64*r[2]/35 - 4./3;
		if ((m==3)&&(n==3)) return 24*r[8]/2695 + 8*r[6]/105 - 24*r[4]/49 + 24*r[2]/35 - 2./3;
	}
	else if (r[1]>1e2) {
		if ((m==1)&&(n==1)) return 2./105 - 24/(245*r[2]) - 8/(735*r[4]) - 8/(2695*r[6]) - 8/(7007*r[8]);
		if (((m==1)&&(n==2))||((m==2)&&(n==3))) return -16./35 + 48/(245*r[2]) - 16/(735*r[4]) - 16/(8085*r[6]) - 16/(35035*r[8]);
		if ((m==2)&&(n==2)) return -44./105 - 32/(735*r[4]) - 64/(8085*r[6]) - 96/(35035*r[8]);
		if ((m==3)&&(n==3)) return -46./105 + 24/(245*r[2]) - 8/(245*r[4]) - 8/(1617*r[6]) - 8/(5005*r[8]);
	}
	else {
		histo_t logr = 0.;
		if (my_abs(r[1]-1)>EPS) logr = my_log(my_abs((r[1] + 1)/(r[1] - 1)));
	
		if ((m==1)&&(n==1)) return -1./84./r[1]*(2*r[1]*(19-24*r[2]+9*r[4])-9*power(r[2]-1,3)*logr);
		if (((m==1)&&(n==2))||((m==2)&&(n==3))) return 1./112./r[3]*(2*r[1]*(r[2]+1)*(3-14*r[2]+3*r[4])-3*power(r[2]-1,4)*logr);
		if ((m==2)&&(n==2)) return 1./336./r[3]*(2*r[1]*(9-185*r[2]+159*r[4]-63*r[6])+9*power(r[2]-1,3)*(7*r[2]+1)*logr);
		if ((m==3)&&(n==3)) return 1./336./r[3]*(2*r[1]*(9-109*r[2]+63*r[4]-27*r[6])+9*power(r[2]-1,3)*(3*r[2]+1)*logr);
	}
	return 0.;
}

histo_t B_mat(size_t n,size_t a,size_t b,histo_t* r,histo_t* x)
{
	//n=1	
	if ((n==1)&&(a==1)&&(b==1)) return r[2]/2.*(x[2]-1);
	if ((n==1)&&(a==1)&&(b==2)) return 3*r[2]/8.*power(x[2]-1,2);
	if ((n==1)&&(a==2)&&(b==1)) return 3*r[4]/8.*power(x[2]-1,2);
	if ((n==1)&&(a==2)&&(b==2)) return 5*r[4]/16.*power(x[2]-1,3);
	//n=2
	if ((n==2)&&(a==1)&&(b==1)) return r[1]/2.*(r[1]+2*x[1]-3*r[1]*x[2]);
	if ((n==2)&&(a==1)&&(b==2)) return -3*r[1]/4.*(x[2]-1)*(-r[1]-2*x[1]+5*r[1]*x[2]);
	if ((n==2)&&(a==2)&&(b==1)) return 3*r[2]/4.*(x[2]-1)*(-2+r[2]+6*r[1]*x[1]-5*r[2]*x[2]);
	if ((n==2)&&(a==2)&&(b==2)) return -3*r[2]/16.*power(x[2]-1,2)*(6-30*r[1]*x[1]-5*r[2]+35*r[2]*x[2]);
	//n=3
	if ((n==3)&&(a==1)&&(b==2)) return r[1]/8.*(4*x[1]*(3-5*x[2])+r[1]*(3-30*x[2]+35*x[4]));
	if ((n==3)&&(a==2)&&(b==1)) return r[1]/8.*(-8*x[1]+r[1]*(-12+36*x[2]+12*r[1]*x[1]*(3-5*x[2])+r[2]*(3-30*x[2]+35*x[4])));
	if ((n==3)&&(a==2)&&(b==2)) return 3*r[1]/16.*(x[2]-1)*(-8*x[1]+r[1]*(-12+60*x[2]+20*r[1]*x[1]*(3-7*x[2])+5*r[2]*(1-14*x[2]+21*x[4])));
	//n=4
	if ((n==4)&&(a==2)&&(b==2)) return r[1]/16.*(8*x[1]*(-3+5*x[2])-6*r[1]*(3-30*x[2]+35*x[4])+6*r[2]*x[1]*(15-70*x[2]+63*x[4])+r[3]*(5-21*x[2]*(5-15*x[2]+11*x[4])));
	return 0.;
}

histo_t kernel_A(size_t m, size_t n, size_t a, histo_t k, histo_t x_, histo_t kq, histo_t mu_, histo_t dmu_, histo_t pk_k, histo_t pk_q, histo_t pk_kq)
{
	//Taruya 2010 (arXiv 1006.0699v1) eq A3
	histo_t x[9],mu[5];
	powers(x_,x,9); powers(mu_,mu,5);
	return (A_mat(m,n,x,mu) * pk_k + Atilde_mat(m,n,x,mu) * pk_q) * pk_kq / power(kq/k,4) + 1./dmu_ * a_mat(m,n,x) * pk_k * pk_q;
}

histo_t kernel_B(size_t n, size_t a, size_t b, histo_t k, histo_t x_, histo_t kq, histo_t mu_, histo_t dmu_, histo_t pk_k, histo_t pk_q, histo_t pk_kq)
{
	//Taruya 2010 (arXiv 1006.0699v1) eq A4
	histo_t x[5],mu[5];
	powers(x_,x,5); powers(mu_,mu,5); 
	return B_mat(n,a,b,x,mu) * pk_kq * pk_q / power(kq/k,2*a);
}


histo_t kernel_pkcorr_A_B(size_t iq, size_t imu, histo_t pk_k, size_t n, size_t a, size_t b, kernel_A_B kernel)
{
	histo_t x = gauss_legendre_q.x[iq];
	histo_t k = gauss_legendre_q.k;
	histo_t q = gauss_legendre_q.q[iq];
	histo_t mu = gauss_legendre_mu.mu[imu];
	histo_t kq = gauss_legendre_q.k*my_sqrt(1.+x*x-2.*mu*x);
	histo_t pk_q = gauss_legendre_q.pk[iq];
	histo_t pk_kq;
	find_pk_lin(&kq,&pk_kq,1,precision_mu.interpol);
	
	return (*kernel)(n,a,b,k,x,kq,mu,2.,pk_k,pk_q,pk_kq);
}


void set_precision_A_B_q(size_t n_,histo_t min_,histo_t max_,char* interpol_)
{
	set_precision(&precision_q,n_,min_,max_,interpol_);
}

void set_precision_A_B_mu(size_t n_,char* interpol_)
{
	set_precision(&precision_mu,n_,-1.,1.,interpol_);
}

void init_A_B()
{
	init_gauss_legendre_q(&gauss_legendre_q,&precision_q);
	init_gauss_legendre_mu(&gauss_legendre_mu,&precision_mu);
}


void free_A_B()
{
	free_gauss_legendre_q(&gauss_legendre_q);
	free_gauss_legendre_mu(&gauss_legendre_mu);
}

_Bool set_mu_range(histo_t x)
{
	histo_t xmin = gauss_legendre_q.x[0];
	histo_t xmax = gauss_legendre_q.x[gauss_legendre_q.nq-1];
	histo_t mumin = MAX(-1.,(1.+x*x-xmax*xmax)/2./x);
	histo_t mumax = MIN(1.,(1.+x*x-xmin*xmin)/2./x);
	if ((mumin>=1.)||(mumax<=-1.)||(mumax<=mumin)) return 0;
	update_gauss_legendre_mu(&gauss_legendre_mu,mumin,mumax);
	return 1;
}

void calc_pkcorr_from_A(histo_t k, histo_t pk_k, histo_t* A)
{

	update_gauss_legendre_q(&gauss_legendre_q,k);
	
	size_t iq,nq=gauss_legendre_q.nq;
	for (ii=0;ii<SHAPE_A[0]*SHAPE_A[1];ii++) A[ii] = 0.;
	
	for (iq=0;iq<nq;iq++) {
		histo_t x = gauss_legendre_q.x[iq];
		if (!set_mu_range(x)) continue;
		
		size_t imu,nmu=gauss_legendre_mu.nmu;
		histo_t xw = gauss_legendre_q.x[iq]*gauss_legendre_q.w[iq];
		for (ii=0;ii<5;ii++) {
			size_t m = INDEX_A[ii][0];
			size_t n = INDEX_A[ii][1];
			histo_t integ_A;
			for (imu=0;imu<nmu;imu++) integ_A += kernel_pkcorr_A_B(iq, imu, pk_k, m+1, n+1, 0, kernel_A) * gauss_legendre_mu.w[imu];
			A[m*SHAPE_A[1]+n] += integ_A * xw;
		}
	}

	for (ii=0;ii<SHAPE_A[0]*SHAPE_A[1];ii++) A[ii] *= k*k*k / (4.*M_PI*M_PI);

}

void calc_pkcorr_from_B(histo_t k, histo_t* B)
{

	update_gauss_legendre_q(&gauss_legendre_q,k);
	
	size_t iq,nq=gauss_legendre_q.nq;
	for (ii=0;ii<SHAPE_B[0]*SHAPE_B[1];ii++) B[ii] = 0.;
	
	for (iq=0;iq<nq;iq++) {
		histo_t x = gauss_legendre_q.x[iq];
		if (!set_mu_range(x)) continue;
		
		size_t imu,nmu=gauss_legendre_mu.nmu;
		histo_t xw = gauss_legendre_q.x[iq]*gauss_legendre_q.w[iq];
		for (ii=0;ii<12;ii++) {
			size_t n = INDEX_B[ii][0];
			size_t a = INDEX_B[ii][1];
			size_t b = INDEX_B[ii][2];
			histo_t integ_B;
			for (imu=0;imu<nmu;imu++) integ_B += kernel_pkcorr_B_B(iq, imu, 0., n+1, a+1, b+1, kernel_B) * gauss_legendre_mu.w[imu];
			B[n*SHAPE_B[1]+a+b] += integ_B * xw;
		}
	}

	for (ii=0;ii<SHAPE_B[0]*SHAPE_B[1];ii++) B[ii] *= k*k*k / (4.*M_PI*M_PI);

}

