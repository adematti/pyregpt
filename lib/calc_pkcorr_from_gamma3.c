#ifndef _REGPT_PKCORRGAMMA3_
#define _REGPT_PKCORRGAMMA3_

#define NDIM 5
#define NCOMP 1
#define EPSREL 0.005
#define EPSABS 1e-12
#define VERBOSE 0
#define MINEVAL 0
#define MAXEVAL 110000000

#define NSTART 4000
#define NINCREASE 700

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cuba.h"
#include "define.h"
#include "common.h"
#include "kernels.h"

static histo_t k,logqmin,logqmax;
static Precision precision_q = {.n=0,.min=5e-4,.max=10.,.interpol=POLY};
static size_t a,b;

#pragma omp threadprivate(k)

static inline histo_t sigmaab(size_t n,size_t a,size_t b)
{
	histo_t sab = 0.;
	histo_t nf = (histo_t) n;
	if ((a==1)&&(b==1)) sab = 2.*nf + 1.;
	else if ((a==1)&&(b==2)) sab = 2.;
	else if ((a==2)&&(b==1)) sab = 3.;
	else if ((a==2)&&(b==2)) sab = 2.*nf;
	return sab / (2.*nf + 3.) / (nf - 1.);
}
/*
static histo_t gam_matrix(size_t a, size_t b, size_t c, histo_t* p, histo_t* q)
{
	histo_t pp = p[0]*p[0] + p[1]*p[1] + p[2]*p[2];
	histo_t qq = q[0]*q[0] + q[1]*q[1] + q[2]*q[2];
	histo_t pq = p[0]*q[0] + p[1]*q[1] + p[2]*q[2];
	if ((a==1)&&(b==1)&&(c==2)) return (1. + pq / qq)/2.;
	if ((a==1)&&(b==2)&&(c==1)) return (1. + pq / pp)/2.;
	if ((a==2)&&(b==2)&&(c==2)) return pq*(pp+qq+2.*pq)/(pp*qq)/2.;
	return 0.;
}

static histo_t F2_sym(size_t a, histo_t* p, histo_t* q)
{
	histo_t pp = my_sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2]);
	histo_t qq = my_sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2]);
	histo_t mu = (p[0]*q[0] + p[1]*q[1] + p[2]*q[2])/(pp*qq);
	if (a==1) return 5./7. + mu/2.*(pp/qq + qq/pp) + 2./7.*mu*mu;
	return 3./7. + mu/2.*(pp/qq + qq/pp) + 4./7.*mu*mu;
}


static histo_t F3(size_t a, histo_t* p, histo_t* q, histo_t* r)
{
	histo_t qr[3];
	qr[0] = q[0] + r[0];
	qr[1] = q[1] + r[1];
	qr[2] = q[2] + r[2];
	if (((my_abs(qr[0])<1e-30)&&(my_abs(qr[1])<1e-30)&&(my_abs(qr[2])<1e-30))
		||((my_abs(p[0]+qr[0])<1e-30)&&(my_abs(p[1]+qr[1])<1e-30)&&(my_abs(p[2]+qr[2])<1e-30))) return 0.;
	return 2*((sigmaab(3,a,1) * gam_matrix(1,1,2,p,qr) + sigmaab(3,a,2) * gam_matrix(2,2,2,p,qr)) * F2_sym(2,q,r) + sigmaab(3,a,1) * gam_matrix(1,2,1,p,qr) * F2_sym(1,q,r));
}
*/

static histo_t F3(size_t a, histo_t* p, histo_t* q, histo_t* r)
{
	histo_t qr[3];
	
	qr[0] = q[0] + r[0];
	qr[1] = q[1] + r[1];
	qr[2] = q[2] + r[2];
	if (((my_abs(qr[0])<1e-30)&&(my_abs(qr[1])<1e-30)&&(my_abs(qr[2])<1e-30))
		||((my_abs(p[0]+qr[0])<1e-30)&&(my_abs(p[1]+qr[1])<1e-30)&&(my_abs(p[2]+qr[2])<1e-30))) return 0.;

	histo_t pp = p[0]*p[0] + p[1]*p[1] + p[2]*p[2];
	histo_t qrqr = qr[0]*qr[0] + qr[1]*qr[1] + qr[2]*qr[2];
	histo_t pqr = p[0]*qr[0] + p[1]*qr[1] + p[2]*qr[2];
	histo_t gam_112 = (1. + pqr / qrqr)/2.;
	histo_t gam_222 = pqr*(pp+qrqr+2.*pqr)/(pp*qrqr)/2.;
	histo_t gam_121 = (1. + pqr / pp)/2.;
	histo_t qq = q[0]*q[0] + q[1]*q[1] + q[2]*q[2];
	histo_t rr = r[0]*r[0] + r[1]*r[1] + r[2]*r[2];
	histo_t mu = q[0]*r[0] + q[1]*r[1] + q[2]*r[2];
	return 2*((sigmaab(3,a,1) * gam_112 + sigmaab(3,a,2) * gam_222) * F2_sym(THETA,qq,rr,mu) + sigmaab(3,a,1) * gam_121 * F2_sym(DELTA,qq,rr,mu));
}


static inline histo_t F3_sym(size_t a, histo_t* p, histo_t* q, histo_t* r)
{
	return (F3(a, p, q, r) + F3(a, r, p, q) + F3(a, q, r, p))/3.;
}


static void integ_pkcorr_G3(const int *ndim, const double xx[],const int *ncomp, double ff[])
{

	histo_t f;
	histo_t pp[3],qq[3],kpq[3],kp[3];
	histo_t p = my_exp(logqmin + (logqmax - logqmin) * ((histo_t) xx[0]));
	histo_t theta1 = ((histo_t) xx[1]) * M_PI;
	histo_t phi1 = ((histo_t) xx[2]) * 2. * M_PI;
	histo_t q = my_exp(logqmin + (logqmax - logqmin) * ((histo_t) xx[3]));
	histo_t theta2 = ((histo_t) xx[4]) * M_PI;

	histo_t sintheta1 = my_sin(theta1);
	histo_t sintheta2 = my_sin(theta2);
	pp[0] = p * sintheta1 * my_cos(phi1);
	pp[1] = p * sintheta1 * my_sin(phi1);
	pp[2] = p * my_cos(theta1);
	qq[0] = q * sintheta2;
	qq[1] = 0.;
	qq[2] = q * my_cos(theta2); 
	kpq[0] = -pp[0]-qq[0];
	kpq[1] = -pp[1]-qq[1];
	kpq[2] = k-pp[2]-qq[2];

	histo_t k_p_q = my_sqrt(kpq[0]*kpq[0] + kpq[1]*kpq[1] + kpq[2]*kpq[2]);
	
	kp[0] = -pp[0];
	kp[1] = -pp[1];
	kp[2] = k-pp[2];
	histo_t k_p = my_sqrt(kp[0]*kp[0] + kp[1]*kp[1] + kp[2]*kp[2]);
	histo_t prod_q_kp = qq[0]*kp[0] + qq[1]*kp[1] + qq[2]*kp[2];
	
	ff[0] = 0.;
	if ((k_p_q>=precision_q.min)&&(k_p_q<=precision_q.max)&&(prod_q_kp<=0.5*k_p*k_p)) {
		histo_t pk_p,pk_q,pk_kpq;
		find_pk_lin(&p,&pk_p,1,precision_q.interpol);
		find_pk_lin(&q,&pk_q,1,precision_q.interpol);
		find_pk_lin(&k_p_q,&pk_kpq,1,precision_q.interpol);
		f = F3_sym(a,pp,qq,kpq) * F3_sym(b,pp,qq,kpq) * pk_p * pk_q * pk_kpq;
		histo_t jacobian = (logqmax-logqmin) * M_PI * 2. * M_PI;
		jacobian *= jacobian * sintheta1 * sintheta2;
		f *= p*p*p*q*q*q * jacobian;
		ff[0] = (double) (2.*f);
	}
}

void set_precision_gamma3(histo_t min_,histo_t max_,char* interpol_)
{
	set_precision(&precision_q,0,min_,max_,interpol_);
}

void init_gamma3()
{
	if ((precision_q.min<=0.)||(precision_q.max<=0.)) {
		precision_q.min = pk_lin.k[0];
		precision_q.max = pk_lin.k[pk_lin.nk-1];
	}
	logqmin = my_log(precision_q.min);
	logqmax = my_log(precision_q.max);
}

void free_gamma3() {}

void calc_pkcorr_from_gamma3(FLAG a_, FLAG b_, histo_t k_, histo_t *pkcorr_G3_tree)
{
	a = (a_==DELTA) ? 1:2;
	b = (b_==DELTA) ? 1:2;
	k = k_;
	
	int neval, fail;
	double integral[NCOMP]={0.}, error[NCOMP]={0.}, prob[NCOMP]={0.};

	Vegas(NDIM, NCOMP, integ_pkcorr_G3, EPSREL, EPSABS, VERBOSE, MINEVAL, MAXEVAL, NSTART, NINCREASE, &neval, &fail, integral, error, prob);
	*pkcorr_G3_tree = 6.*integral[0]/my_pow(2.*M_PI,6);

}

/*
/*
static void integ_pkcorr_G3(const int *ndim, const double xx[],const int *ncomp, double ff[])
{

	histo_t f;
	histo_t pp[3],qq[3],kpq[3],kp[3];
	histo_t p = my_exp(my_log(kmin) + (my_log(kmax) - my_log(kmin)) * ((histo_t) xx[0]));
	histo_t theta1 = ((histo_t) xx[1]) * M_PI;
	histo_t phi1 = ((histo_t) xx[2]) * 2. * M_PI;
	histo_t q = my_exp(my_log(kmin) + (my_log(kmax) - my_log(kmin)) * ((histo_t) xx[3]));
	histo_t theta2 = ((histo_t) xx[4]) * M_PI;
	histo_t jacobian = (my_log(kmax)-my_log(kmin)) * M_PI * 2. * M_PI;
	jacobian *= jacobian;

	pp[0] = p * my_sin(theta1) * my_cos(phi1);
	pp[1] = p * my_sin(theta1) * my_sin(phi1);
	pp[2] = p * my_cos(theta1);
	qq[0] = q * my_sin(theta2);
	qq[1] = 0.;
	qq[2] = q * my_cos(theta2); 
	kpq[0] = -pp[0]-qq[0];
	kpq[1] = -pp[1]-qq[1];
	kpq[2] = k-pp[2]-qq[2];

	histo_t k_p_q = my_sqrt(kpq[0]*kpq[0] + kpq[1]*kpq[1] + kpq[2]*kpq[2]);
	
	kp[0] = -pp[0];
	kp[1] = -pp[1];
	kp[2] = k-pp[2];
	histo_t k_p = my_sqrt(kp[0]*kp[0] + kp[1]*kp[1] + kp[2]*kp[2]);
	histo_t prod_q_kp = qq[0]*kp[0] + qq[1]*kp[1] + qq[2]*kp[2];
	
	f = 0.;
	if ((k_p_q>=kmin)&&(k_p_q<=kmax)&&(prod_q_kp<=0.5*k_p*k_p)) {
		histo_t pk_p,pk_q,pk_kpq;
		find_pk_lin(&p,&pk_p,1,POLY);
		find_pk_lin(&q,&pk_q,1,POLY);
		find_pk_lin(&k_p_q,&pk_kpq,1,POLY);
		f = F3_sym(a, pp, qq, kpq) * F3_sym(b, pp, qq, kpq) * pk_p * pk_q * pk_kpq;
		f *= p*p*p*q*q*q * my_sin(theta1) * my_sin(theta2);
	}
	
	ff[0] = (double) (2.*f*jacobian);
}
*/

/*
#define NDIM 5
#define NCOMP 1
#define USERDATA NULL
#define NVEC 1
#define EPSREL 0.005
#define EPSABS 1e-12
#define VERBOSE 0
#define SEED 0
#define MINEVAL 0
#define MAXEVAL 110000000

#define NSTART 4000
#define NINCREASE 700
#define NBATCH 1000
#define GRIDNO 0
#define STATEFILE NULL
#define SM_PIN NULL


static int integ_pkcorr_G3(const int *ndim, const cubareal xx[],const int *ncomp, cubareal ff[], void *userdata)
{
#define f ff[0]
	f = 0.;
}

void calc_pkcorr_from_Gamma3(FLAG a, FLAG b, histo_t k, histo_t *pkcorr_G3_tree)
{
	int neval, fail;
	double integral[NCOMP], error[NCOMP], prob[NCOMP];

	Vegas(NDIM, NCOMP, integ_pkcorr_G3, USERDATA, NVEC, EPSREL, EPSABS, VERBOSE, SEED, MINEVAL, MAXEVAL, NSTART, NINCREASE, NBATCH, GRIDNO, STATEFILE, SM_PIN, &neval, &fail, integral, error, prob);

	printf("VEGAS RESULT:\tneval %d\tfail %d\n",neval, fail);
	size_t comp;
	for(comp = 0; comp < NCOMP; ++comp)	
		printf("VEGAS RESULT:\t%.8f +- %.8f\tp = %.3f\n",integral[comp], error[comp], prob[comp]);
}

*/

#endif //_REGPT_PKCORRGAMMA3_
