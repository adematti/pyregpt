#ifndef _REGPT_PKCORRGAMMA1_
#define _REGPT_PKCORRGAMMA1_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "define.h"
#include "common.h"

static GaussLegendreQ gauss_legendre_q_1loop;
static GaussLegendreQ gauss_legendre_q_2loop;
#pragma omp threadprivate(gauss_legendre_q_1loop,gauss_legendre_q_2loop)
static Precision precision_q_1loop = {.n=600,.min=5e-4,.max=10.,.interpol=POLY};
static Precision precision_q_2loop = {.n=500,.min=5e-4,.max=10.,.interpol=POLY};

histo_t kernel_ff(histo_t x)
{
	histo_t kernel_ff,x2;
	histo_t diffx = x-1.;
	if (my_abs(diffx)<0.01) {
	 	kernel_ff = - 11./126. + diffx/126. - 29./252. * diffx*diffx;
	}
	else if (x>10.) {
		x2 = x*x;
		kernel_ff = - 61./630. + 2./105./x2 - 10./1323./x2/x2;
	}
	else {
		x2 = x*x;
		kernel_ff = 6./x2 - 79. + 50.*x2 - 21.*x2*x2 + 0.75*power(1./x-x,3) * (2. + 7.*x2) * 2 * my_log(my_abs(diffx/(1.+x)));
		kernel_ff /= 504.;
#ifdef _DEBUG
		//printf("%.10g %.10g %.10g\n",x,6./x2 - 79. + 50.*x2 - 21.*x2*x2,0.75*power(1./x-x,3) * (2. + 7.*x2) * 2 * my_log(my_abs(diffx/(1.+x))));
#endif //_DEBUG
	}
	return kernel_ff * x;
}


histo_t kernel_gg(histo_t x)
{
	histo_t kernel_gg,x2;
	histo_t diff = x-1.;
	if (my_abs(diff)<0.01) {
	 	kernel_gg = - 3./14. - 5./42.*diff + diff*diff/84.;
	}
	else if (x>10.) {
		x2 = x*x;
		kernel_gg = - 3./10. + 26./245./x2 - 38./2205./x2/x2;
	}
	else {
		x2 = x*x;
		kernel_gg = 6./x2 - 41. + 2.*x2 - 3.*x2*x2 + 0.75*power(1./x-x,3) * (2. + x2) * 2 * my_log(my_abs(diff/(1.+x)));
		kernel_gg /= 168.;
	}
	return kernel_gg * x;
}


histo_t one_loop_gamma1(FLAG a)
{
	size_t iq,nq = gauss_legendre_q_1loop.nq;
	histo_t gamma = 0.;
	
	if (a==DELTA) {
		for (iq=0;iq<nq;iq++) gamma += kernel_ff(gauss_legendre_q_1loop.x[iq])*gauss_legendre_q_1loop.pk[iq]*gauss_legendre_q_1loop.w[iq];
	}
	else { //THETA
		for (iq=0;iq<nq;iq++) gamma += kernel_gg(gauss_legendre_q_1loop.x[iq])*gauss_legendre_q_1loop.pk[iq]*gauss_legendre_q_1loop.w[iq];
	}
#ifdef _DEBUG
	//printf("%.18f %.18f\n",gauss_legendre_q.x[0],gauss_legendre_q.x[MAX_IQ-1]);
	//for (iq=0;iq<10;iq++) printf("%.10g %.10g %.10g %.10g %.10g\n",gauss_legendre_q.x[iq],gauss_legendre_q.q[iq],kernel_ff(gauss_legendre_q.x[iq]),gauss_legendre_q.pk[iq],gauss_legendre_q.w[iq]);
	//printf("G1-1loop %f %f\n",k,gamma);
#endif //_DEBUG
	histo_t k = gauss_legendre_q_1loop.k;
	return gamma * k*k*k/(2.*M_PI*M_PI);
}


histo_t beta(FLAG a, histo_t q1, histo_t q2) {

	histo_t y = my_abs((q1 - q2) / (q1 + q2));
	histo_t diffy = 1. - y;
	
	if (a==DELTA) {
		if (my_abs(diffy) < 0.05) {
			histo_t diffy2 = diffy*diffy;
			histo_t diffy3 = diffy2*diffy;
			histo_t diffy4 = diffy3*diffy;
			return 120424. / 3009825. + (2792.*diffy2)/429975. + (2792.*diffy3)/429975. + (392606.*diffy4)/9.9324225e7;
		}
		else if (y < 0.01) {
			return 22382./429975. - 57052.*y*y/429975.;
		}
		else {
			histo_t y2 = y*y;
			histo_t y4 = y2*y2;
			histo_t y8 = y4*y4;
			histo_t y10 = y2*y8;
			histo_t y12 = y2*y10;
			return (2.*(1. + y2)*(-11191. + 118054.*y2 - 18215.*y4 + 18215.*y8 - 118054.*y10 + 11191.*y12 + 60.*y4*(3467. - 790.*y2 + 3467.*y4)*2*my_log(y))) / (429975.*power((-1. + y2),7));
		}
	}
	else { //THETA
		if (my_abs(diffy) < 0.05) {
			histo_t diffy2 = diffy*diffy;
			histo_t diffy3 = diffy2*diffy;
			histo_t diffy4 = diffy3*diffy;
			return 594232. / 33108075. + (91912.*diffy2)/33108075. - (91912.*diffy3)/33108075. + (1818458.*diffy4)/1092566475.;
		}
		else if (y < 0.01) {
			return 9886./429975. - 254356.*y*y/4729725.;
		}
		else {
			histo_t y2 = y*y;
			histo_t y4 = y2*y2;
			histo_t y8 = y4*y4;
			histo_t y10 = y2*y8;
			histo_t y12 = y2*y10;
			return (2.*(1. + y2)*(-54373. + 562162.*y2 - 408245.*y4 + 408245.*y8 - 562162.*y10 + 54373.*y12 + 60.*y4*(14561. - 10690.*y2 + 14561.*y4)*2*my_log(y))) / (4.729725e6*power((-1. + y2),7));
		}
	}
}

histo_t cc(FLAG a, histo_t qs) {

	if(a==DELTA) {
		histo_t qs2 = qs*qs;
		histo_t qs3 = qs*qs2;
		histo_t qs4 = qs*qs3;
		histo_t qs6 = qs2*qs4;
		return 0.02088557981734545/(35.09866396385646*qs4 + 4.133811416743832*qs2 + 1.) - 0.076100391588544*qs3 / (77.79670692480381*qs6 + 1.);
	}
	else { //THETA
		histo_t qs2 = qs*qs;
		histo_t qs3 = qs*qs2;
		histo_t qs4 = qs*qs3;
		histo_t qs8 = qs4*qs4;
		return - 0.008217140060512867 / (42.14072830553836*qs4 + 1.367564560397748*qs2 + 1.) + 0.01099093588476197*qs3 / (28.490424851390667*qs8 + 1.);
	}
}

histo_t dd(FLAG a, histo_t qr) {

	if(a==DELTA) {
		histo_t qr2 = qr*qr;
		histo_t qr4 = qr2*qr2;
		histo_t qr10 = qr2*qr4*qr4;
		return - 0.022168478217299517 / (7.030631093970638*qr4 + 2.457866449142683*qr2 + 1.) + 0.009267495321465601*qr2 / (4.11633699497035*qr10 + 1.);
    }
	else { //THETA
		histo_t qr2 = qr*qr;
		histo_t qr5 = qr*qr2*qr2;
		return 0.008023147297149955 / (2.238261369090066*qr5 + 1.) - 0.006173880966928251*qr2 / (0.4711737436482179*qr5 + 1.);
     }
}

histo_t delta(FLAG a, histo_t qs) {

	qs *= my_sqrt(2.);
	histo_t qs2 = qs*qs;
	histo_t qs4 = qs2*qs2;

	if(a==DELTA) {
		return  0.3191221482038663*qs4 / (1.3549058352752525*qs4 + 1.) + 1.2805575495849764 / (18.192939946270577*qs4 + 3.98817716852858*qs2 + 1.) + 0.764469131436698;
	}
	else { //THETA
		return 1.528058751211026 * (2.4414566000839355*qs4 + 1.8616263354608626*qs2) / (2.4414566000839355*qs4 + 1.) + 2.5227965281961247 / (0.0028106312591877226*qs4 + 1.0332351481570086*qs2 + 1.) - 0.528058751211026;
	}
}

histo_t alpha(FLAG a, histo_t q1, histo_t q2) {
	
	histo_t qr = my_sqrt(q1*q1 + q2*q2);
	histo_t qs = q1*q2 / qr;
	return ( beta(a, q1, q2) - cc(a, qs) - dd(a, qr) ) * delta(a, qs);
}


histo_t integ_gamma1_2loop(FLAG a, histo_t k, histo_t q1, histo_t q2) {

      histo_t x1 = q1 / k;
      histo_t x2 = q2 / k;
      return - alpha(a, x1, x2) / (x1*x1 + x2*x2);
}

histo_t two_loop_gamma1(FLAG a)
{

	size_t iq1,iq2,nq = gauss_legendre_q_2loop.nq;
	histo_t gamma = 0.;
	
	for (iq1=0;iq1<nq;iq1++) {
		histo_t integ_pkcorr_gamma1 = 0.;
		histo_t q1 = gauss_legendre_q_2loop.q[iq1];
		histo_t q13 = q1*q1*q1;
		for (iq2=0;iq2<nq;iq2++) {
			histo_t q2 = gauss_legendre_q_2loop.q[iq2];
			integ_pkcorr_gamma1 += gauss_legendre_q_2loop.w[iq2] * integ_gamma1_2loop(a, gauss_legendre_q_2loop.k, q1, q2) * gauss_legendre_q_2loop.pk[iq2] * q13 * q2*q2*q2;
		}
		gamma += gauss_legendre_q_2loop.w[iq1] * integ_pkcorr_gamma1 * gauss_legendre_q_2loop.pk[iq1];
	}
	return gamma / (4.*power(M_PI,4));
}

void set_precision_gamma1_1loop(size_t n_,histo_t min_,histo_t max_,char* interpol_)
{
	set_precision(&precision_q_1loop,n_,min_,max_,interpol_);
}

void set_precision_gamma1_2loop(size_t n_,histo_t min_,histo_t max_,char* interpol_)
{
	set_precision(&precision_q_2loop,n_,min_,max_,interpol_);
}

void init_gamma1()
{
	init_gauss_legendre_q(&gauss_legendre_q_1loop,&precision_q_1loop);
	init_gauss_legendre_q(&gauss_legendre_q_2loop,&precision_q_2loop);
}

void free_gamma1()
{
	free_gauss_legendre_q(&gauss_legendre_q_1loop);
	free_gauss_legendre_q(&gauss_legendre_q_2loop);
}

void calc_pkcorr_from_gamma1(FLAG a,FLAG b,histo_t k,histo_t *G1a_1loop,histo_t *G1a_2loop,histo_t *G1b_1loop,histo_t *G1b_2loop)
{
	update_gauss_legendre_q(&gauss_legendre_q_1loop,k);
	update_gauss_legendre_q(&gauss_legendre_q_2loop,k);
	if (a==b) {
		*G1a_1loop = one_loop_gamma1(a);
		*G1a_2loop = two_loop_gamma1(a);
		*G1a_2loop = (*G1a_2loop) + (*G1a_1loop)*(*G1a_1loop)/2.;
		*G1b_1loop = *G1a_1loop;
		*G1b_2loop = *G1a_2loop;
#ifdef _DEBUG
		//printf("%.10g %.10g %.10g\n",gauss_legendre_q.k,*G1a_1loop,*G1a_2loop);
#endif //_DEBUG
	}
	else {
		*G1a_1loop = one_loop_gamma1(a);
		*G1a_2loop = two_loop_gamma1(a);
		*G1b_1loop = one_loop_gamma1(b);
		*G1b_2loop = two_loop_gamma1(b);
		*G1a_2loop = (*G1a_2loop) + (*G1a_1loop)*(*G1a_1loop)/2.;
        *G1b_2loop = (*G1b_2loop) + (*G1b_1loop)*(*G1b_1loop)/2.;
	}

}

#endif //_REGPT_PKCORRGAMMA1_
