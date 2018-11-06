#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "define.h"
#include "common.h"
#include "kernels.h"

histo_t F2_sym(FLAG a, histo_t p2, histo_t q2, histo_t pq)
{
	if (a==DELTA) return 5./7. + 0.5*pq*(1/p2 + 1/q2) + 2./7.*pq*pq/p2/q2;
	return 3./7. + 0.5*pq*(1/p2 + 1/q2) + 4./7.*pq*pq/p2/q2;
}

histo_t F2_sym_full(FLAG a, histo_t p, histo_t q, histo_t mu)
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


histo_t kernel_b2(FLAG a, histo_t q, histo_t kq, histo_t mu, histo_t mukkq, histo_t pk_q, histo_t pk_kq)
{
	//return pk_q * pk_kq * F2_sym_full(a,q,kq,mukkq);
	return pk_q * pk_kq * F2_sym(a,q*q,kq*kq,mukkq*q*kq);
}

histo_t kernel_bs2(FLAG a, histo_t q, histo_t kq, histo_t mu, histo_t mukkq, histo_t pk_q, histo_t pk_kq)
{
	return pk_q * pk_kq * F2_sym(a,q*q,kq*kq,mukkq*q*kq) * S2(mukkq);
}

histo_t kernel_b22(FLAG a, histo_t q, histo_t kq, histo_t mu, histo_t mukkq, histo_t pk_q, histo_t pk_kq)
{
	return 0.5 * pk_q * (pk_kq - pk_q);
}

histo_t kernel_b2s2(FLAG a, histo_t q, histo_t kq, histo_t mu, histo_t mukkq, histo_t pk_q, histo_t pk_kq)
{
	return -0.5 * pk_q * (2./3. * pk_q - pk_kq * S2(mukkq));
}

histo_t kernel_bs22(FLAG a, histo_t q, histo_t kq, histo_t mu, histo_t mukkq, histo_t pk_q, histo_t pk_kq)
{
	histo_t s2 = S2(mukkq);
	return -0.5 * pk_q * (4./9. * pk_q - pk_kq * s2*s2);
}

histo_t kernel_sigma3sq(FLAG a, histo_t q, histo_t kq, histo_t mu, histo_t mukkq, histo_t pk_q, histo_t pk_kq)
{
	return 105./16. * pk_q * (D2(mu)*S2(mukkq) + 8./63.);
}


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

histo_t kernel_A(size_t m, size_t n, size_t a, histo_t k, histo_t x_, histo_t kq, histo_t mu_, histo_t pk_k, histo_t pk_q, histo_t pk_kq)
{
	//Taruya 2010 (arXiv 1006.0699v1) eq A3
	histo_t x[9],mu[5];
	powers(x_,x,9); powers(mu_,mu,5);
	return (A_mat(m,n,x,mu) * pk_k + Atilde_mat(m,n,x,mu) * pk_q) * pk_kq / power(kq/k,4) + 1./2. * a_mat(m,n,x) * pk_k * pk_q;
}

histo_t kernel_B(size_t n, size_t a, size_t b, histo_t k, histo_t x_, histo_t kq, histo_t mu_, histo_t pk_k, histo_t pk_q, histo_t pk_kq)
{
	//Taruya 2010 (arXiv 1006.0699v1) eq A4
	histo_t x[5],mu[5];
	powers(x_,x,5); powers(mu_,mu,5); 
	return B_mat(n,a,b,x,mu) * pk_kq * pk_q / power(kq/k,2*a);
}
