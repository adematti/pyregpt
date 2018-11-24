#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "define.h"
#include "common.h"

static gammaaussLegendreQ gauss_legendre_q;
#pragma omp threadprivate(gauss_legendre_q)
static const Precision precision_q_default = {.n=500,.min=5e-4,.max=10.,.interpol=POLY};
static Precision precision_q = {.n=500,.min=5e-4,.max=10.,.interpol=POLY};

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

histo_t gamma1_2loop(FLAG a,histo_t k)
{
	update_gauss_legendre_q(&gauss_legendre_q,k);
	size_t iq1,iq2,nq = gauss_legendre_q.nq;
	histo_t integ_gamma = 0.;
	
	for (iq1=0;iq1<nq;iq1++) {
		histo_t q1 = gauss_legendre_q.q[iq1];
		histo_t q13 = q1*q1*q1;
		histo_t integ_gamma_2 = 0.;
		for (iq2=0;iq2<nq;iq2++) {
			histo_t q2 = gauss_legendre_q.q[iq2];
			integ_gamma_2 += gauss_legendre_q.w[iq2] * integ_gamma1_2loop(a, gauss_legendre_q.k, q1, q2) * gauss_legendre_q.pk[iq2] * q13 * q2*q2*q2;
		}
		integ_gamma += gauss_legendre_q.w[iq1] * integ_gamma_2 * gauss_legendre_q.pk[iq1];
	}
	return integ_gamma / (4.*power(M_PI,4));
}

void set_precision_gamma1_2loop_q(size_t n_,histo_t min_,histo_t max_,char* interpol_)
{
	set_precision(&precision_q,n_,min_,max_,interpol_,&precision_q_default);
}

void init_gamma1_2loop()
{
	init_gauss_legendre_q(&gauss_legendre_q,&precision_q);
}

void free_gamma1_2loop()
{
	free_gauss_legendre_q(&gauss_legendre_q);
}

