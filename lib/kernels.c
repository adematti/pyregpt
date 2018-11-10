#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "define.h"
#include "common.h"
#include "kernels.h"

histo_t F2_sym_fast(FLAG a, histo_t p2, histo_t q2, histo_t pq)
{
	if (a==DELTA) return 5./7. + 0.5*pq*(1/p2 + 1/q2) + 2./7.*pq*pq/p2/q2;
	return 3./7. + 0.5*pq*(1/p2 + 1/q2) + 4./7.*pq*pq/p2/q2;
}

histo_t F2_sym_full(FLAG a, histo_t *p, histo_t *q)
{
	histo_t pp = p[0]*p[0]+p[1]*p[1]+p[2]*p[2];
    histo_t qq = q[0]*q[0]+q[1]*q[1]+q[2]*q[2];
	histo_t pq = p[0]*q[0]+p[1]*q[1]+p[2]*q[2]; 
    return F2_sym_fast(a,pp,qq,pq);
}

histo_t F2_sym(FLAG a, histo_t p, histo_t q, histo_t mu)
{
	if (a==DELTA) return 5./7. + 0.5*mu*(q/p + p/q) + 2./7.*mu*mu;
	return 3./7. + 0.5*mu*(q/p + p/q) + 4./7.*mu*mu;
}

histo_t S2(histo_t mu)
{
	return mu*mu-1./3.;
}

histo_t D2(histo_t mu)
{
	return 2./7.*(S2(mu)-2./3.);
}

histo_t LFunc(histo_t k, histo_t q)
{
	//return my_log((k + q)*(k + q)/(k - q)/(k - q));
	return 2.*my_log(my_abs((k + q)/(k - q)));
}

histo_t WFunc(histo_t k1, histo_t k2, histo_t k3, histo_t q)
{
	histo_t k12 = k1*k1;
	histo_t k22 = k2*k2;
	histo_t k32 = k3*k3;
	histo_t q2 = q*q;
	histo_t aa = -4.*k32*q2 - 2.*(k12 - q2)*(k22 - q2);
	histo_t bb = 4.*k3*q*my_sqrt(k12*k22 + (-k12 - k22 + k32)*q2 + q2*q2);
	
	//if (isnan(my_log((aa-bb)/(aa+bb)))) printf("%.3lf %.3lf %.3lf %.3lf %.3lf %.3lf\n",k1,k2,k3,q,aa,k12*k22 + (-k12 - k22 + k32)*q2 + q2*q2);
	
	return my_log((aa-bb)/(aa+bb));
}

histo_t betafunc(size_t a,histo_t z)
{
	histo_t z2 = z*z;
	histo_t z3 = z2*z;
	histo_t z4 = z3*z;
	if (z<=0.1) {
		histo_t fa = (histo_t) a;
		histo_t z5 = z4*z;
		histo_t z6 = z5*z;
		return power(z,a)*(1./fa + z/(1. + fa) + z2/(2. + fa) + z3/(3. + fa) + z4/(4. + fa) + z5/(5. + fa) + z6/(6. + fa));
	}
	if (a==2) return z2*(-2./z - 2.*my_log(1. - z)/z2)/2.;
	if (a==4) return z4*(-2.*(6. + 3.*z + 2.*z2)/(3.*z3) - 4.*my_log(1. - z)/z4)/4.;
	if (a==6) {
		histo_t z5 = z4*z;
		histo_t z6 = z5*z;
		return z6*((-60. - 30.*z - 20.*z2 - 15.*z3 - 12.*z4)/(10.*z5) - 6.*my_log(1. - z)/z6)/6.;
	}
	return 0.;
}

histo_t small_beta(histo_t k, histo_t q)
{
	histo_t y = (k - q) / (k + q);
	if (my_abs(y) <= 1.e-6) return 0.;
	//if (my_abs(betafunc(4, 1. - y*y))>1e6) printf("small ");
    return y * betafunc(4, 1. - y*y); 
}

histo_t big_beta(histo_t k1,histo_t k2,histo_t k3,histo_t q)
{
	histo_t x = (k1*k1-q*q) * (k2*k2-q*q);
	histo_t a = k3 * q;
	histo_t y = my_sqrt(x+a*a)/a;
	histo_t y1 = my_abs(y-1);
	if (y1<=1.e-5) return 0.;
	if (y1<=1.e-2) return 2.*a*(-1. + y)*(-1. + my_log(4.) - 2.*my_log(y1)) + a*y1*y1*(3.-my_log(4.) + 2.*my_log(y1));
	//if (betafunc(2,1.-y1*y1/(1.+y)/(1.+y))/(a*y)>1e6) printf("big ");
	return x*betafunc(2,1.-y1*y1/(1.+y)/(1.+y))/(a*y);
}

histo_t sigmaab(size_t n,size_t a,size_t b)
{
	histo_t sab = 0.;
	histo_t nf = (histo_t) n;
	if ((a==1)&&(b==1)) sab = 2.*nf + 1.;
	else if ((a==1)&&(b==2)) sab = 2.;
	else if ((a==2)&&(b==1)) sab = 3.;
	else if ((a==2)&&(b==2)) sab = 2.*nf;
	return sab / (2.*nf + 3.) / (nf - 1.);
}

histo_t F3(FLAG a_, histo_t* p, histo_t* q, histo_t* r)
{
	size_t a = (a_==DELTA) ? 1:2;
	histo_t qr[3];
	qr[0] = q[0] + r[0];
	qr[1] = q[1] + r[1];
	qr[2] = q[2] + r[2];
	if (((my_abs(qr[0])<EPS)&&(my_abs(qr[1])<EPS)&&(my_abs(qr[2])<EPS))
		||((my_abs(p[0]+qr[0])<EPS)&&(my_abs(p[1]+qr[1])<EPS)&&(my_abs(p[2]+qr[2])<EPS))) return 0.;

	histo_t pp = p[0]*p[0] + p[1]*p[1] + p[2]*p[2];
	histo_t qrqr = qr[0]*qr[0] + qr[1]*qr[1] + qr[2]*qr[2];
	histo_t pqr = p[0]*qr[0] + p[1]*qr[1] + p[2]*qr[2];
	histo_t gam_112 = (1. + pqr / qrqr)/2.;
	histo_t gam_222 = pqr*(pp+qrqr+2.*pqr)/(pp*qrqr)/2.;
	histo_t gam_121 = (1. + pqr / pp)/2.;
	histo_t qq = q[0]*q[0] + q[1]*q[1] + q[2]*q[2];
	histo_t rr = r[0]*r[0] + r[1]*r[1] + r[2]*r[2];
	histo_t mu = q[0]*r[0] + q[1]*r[1] + q[2]*r[2];
	return 2*((sigmaab(3,a,1) * gam_112 + sigmaab(3,a,2) * gam_222) * F2_sym_fast(THETA,qq,rr,mu) + sigmaab(3,a,1) * gam_121 * F2_sym_fast(DELTA,qq,rr,mu));
}


histo_t F3_sym(FLAG a, histo_t* p, histo_t* q, histo_t* r)
{
	return (F3(a, p, q, r) + F3(a, r, p, q) + F3(a, q, r, p))/3.;
}
