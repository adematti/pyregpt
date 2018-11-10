#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include "define.h"
#include "common.h"
#include "kernels.h"

static INTERPOL interpol_pk_lin = POLY;
static histo_t uvcutoff = 0.5;

void gamma1_reg_1loop(histo_t k, histo_t p, histo_t kp, histo_t sigmav2k, histo_t sigmav2p, histo_t sigmav2kp, histo_t* G1d_k, histo_t* G1t_k, histo_t* G1d_p, histo_t* G1t_p, histo_t* G1d_kp, histo_t* G1t_kp)
{
	if(k>=1e-2) {
		*G1d_k = 1. + k*k*sigmav2k/2. + gamma1_1loop(DELTA,k);
        *G1t_k = 1. + k*k*sigmav2k/2. + gamma1_1loop(THETA,k);
	}
	else {
		*G1d_k = 1.;
		*G1t_k = 1.;
	}
	if ((p>=1e-2)&&(k>=1.e-2)) {
		*G1d_p = 1. + p*p*sigmav2p/2. + gamma1_1loop(DELTA,p);
        *G1t_p = 1. + p*p*sigmav2p/2. + gamma1_1loop(THETA,p);
	}
	else {
		*G1d_p = 1.;
		*G1t_p = 1.;
	}
	if ((kp>=3e-2)&&(k>=1.e-2)) {
		*G1d_kp = 1. + kp*kp*sigmav2kp/2. + gamma1_1loop(DELTA,kp);
        *G1t_kp = 1. + kp*kp*sigmav2kp/2. + gamma1_1loop(THETA,kp);
	}
	else {
		*G1d_kp = 1.;
		*G1t_kp = 1.;
	}	

}

void gamma2_reg_1loop(histo_t k,histo_t p,histo_t kp,histo_t sigmav2k,histo_t sigmav2p,histo_t sigmav2kp,histo_t* G2d_kp_k,histo_t* G2t_kp_k,histo_t* G2d_p_k,histo_t* G2t_p_k,histo_t* G2d_p_kp,histo_t* G2t_p_kp)
{
	if ((MAX3(kp, k, p)>=1e-2) && (k>=1.e-2)) {
		*G2d_kp_k = gamma2_tree(DELTA, kp, k, p) * (1. + p*p*sigmav2p/2.) + gamma2d_1loop(kp, k, p);
    	*G2t_kp_k = gamma2_tree(THETA, kp, k, p) * (1. + p*p*sigmav2p/2.) + gamma2t_1loop(kp, k, p);
   		*G2d_p_k = gamma2_tree(DELTA, p, k, kp) * (1. + p*p*sigmav2kp/2.) + gamma2d_1loop(p, k, kp);
    	*G2t_p_k = gamma2_tree(THETA, p, k, kp) * (1. + p*p*sigmav2kp/2.) + gamma2t_1loop(p, k, kp);
    	*G2d_p_kp = gamma2_tree(DELTA, p, kp, k) * (1. + p*p*sigmav2k/2.) + gamma2d_1loop(p, kp, k);
    	*G2t_p_kp = gamma2_tree(THETA, p, kp, k) * (1. + p*p*sigmav2k/2.) + gamma2t_1loop(p, kp, k);
	}    
	
	else {
         *G2d_kp_k = gamma2_tree(DELTA, kp, k, p);
         *G2t_kp_k = gamma2_tree(THETA, kp, k, p);
         *G2d_p_k = gamma2_tree(DELTA, p, k, kp);
         *G2t_p_k = gamma2_tree(THETA, p, k, kp);
         *G2d_p_kp = gamma2_tree(DELTA, p, kp, k);
         *G2t_p_kp = gamma2_tree(THETA, p, kp, k);
	}

}

void set_precision_bispectrum_1loop_pk_lin(char* interpol_)
{
	set_interpol(&interpol_pk_lin,interpol_);
}

void set_running_uvcutoff_bispectrum_1loop(histo_t uvcutoff_)
{
	uvcutoff = uvcutoff_;
}

void get_precision_bispectrum_1loop_pk_lin(INTERPOL *interpol_)
{
	*interpol_ = interpol_pk_lin;
}

void get_running_uvcutoff_bispectrum_1loop(histo_t *uvcutoff_)
{
	*uvcutoff_ = uvcutoff;
}


void bispectrum_1loop_I(histo_t k, histo_t p, histo_t kp, histo_t* b211A, histo_t* b221A, histo_t* b212A, histo_t* b222A, histo_t* b211tA, histo_t* b221tA, histo_t* b212tA, histo_t* b222tA)
{
	histo_t sigmav2k,sigmav2p,sigmav2kp;
	calc_running_sigma_v2(&k,&sigmav2k,1,uvcutoff);
	calc_running_sigma_v2(&p,&sigmav2p,1,uvcutoff);
	calc_running_sigma_v2(&kp,&sigmav2kp,1,uvcutoff);
	
	histo_t exp_factor = (k*k*sigmav2k + p*p*sigmav2p + kp*kp*sigmav2kp) / 2.;
	if(exp_factor>1e2) {
         *b211A = 0.;
         *b221A = 0.;
         *b212A = 0.;
         *b222A = 0.;
         *b211tA = 0.;
         *b221tA = 0.;
         *b212tA = 0.;
         *b222tA = 0.;
	}
	exp_factor = my_exp(-exp_factor);
	
	histo_t pk_k,pk_p,pk_kp;
	find_pk_lin(&k,&pk_k,1,interpol_pk_lin);
	find_pk_lin(&p,&pk_p,1,interpol_pk_lin);
	find_pk_lin(&kp,&pk_kp,1,interpol_pk_lin);
	
	histo_t G1d_k, G1t_k, G1d_p, G1t_p, G1d_kp, G1t_kp;
	gamma1_reg_1loop(k, p, kp, sigmav2k, sigmav2p, sigmav2kp, &G1d_k, &G1t_k, &G1d_p, &G1t_p, &G1d_kp, &G1t_kp);
	histo_t G2d_kp_k, G2t_kp_k, G2d_p_k, G2t_p_k, G2d_p_kp, G2t_p_kp;
	gamma2_reg_1loop(k, p, kp, sigmav2k, sigmav2p, sigmav2kp, &G2d_kp_k, &G2t_kp_k, &G2d_p_k, &G2t_p_k, &G2d_p_kp, &G2t_p_kp);

	*b211A = 2. * ( G2t_kp_k * G1d_kp * G1d_k * pk_kp * pk_k
			+ G2d_p_k * G1t_p * G1d_k * pk_p * pk_k
			+ G2d_p_kp * G1t_p * G1d_kp * pk_p * pk_kp )
			* exp_factor;
			
 	*b221A = 2. * ( G2t_kp_k * G1t_kp * G1d_k * pk_kp * pk_k
			+ G2t_p_k * G1t_p * G1d_k * pk_p * pk_k
			+ G2d_p_kp * G1t_p * G1t_kp * pk_p * pk_kp )
			* exp_factor;

	*b212A = 2. * ( G2t_kp_k * G1d_kp * G1t_k * pk_kp * pk_k
			+ G2d_p_k * G1t_p * G1t_k * pk_p * pk_k
			+ G2t_p_kp * G1t_p * G1d_kp * pk_p * pk_kp )
			* exp_factor;

	*b222A = 2. * ( G2t_kp_k * G1t_kp * G1t_k * pk_kp * pk_k
			+ G2t_p_k * G1t_p * G1t_k * pk_p * pk_k
			+ G2t_p_kp * G1t_p * G1t_kp * pk_p * pk_kp )
			* exp_factor;

	*b211tA = 2. * ( G2t_p_k * G1d_p * G1d_k * pk_p * pk_k
			+ G2d_kp_k * G1t_kp * G1d_k * pk_kp * pk_k
			+ G2d_p_kp * G1t_kp * G1d_p * pk_kp * pk_p )
			* exp_factor;

	*b221tA = 2. * ( G2t_p_k * G1t_p * G1d_k * pk_p * pk_k
			+ G2t_kp_k * G1t_kp * G1d_k * pk_kp * pk_k
			+ G2d_p_kp * G1t_kp * G1t_p * pk_kp * pk_p )
			* exp_factor;

	*b212tA = 2. * ( G2t_p_k * G1d_p * G1t_k * pk_p * pk_k
			+ G2d_kp_k * G1t_kp * G1t_k * pk_kp * pk_k
			+ G2t_p_kp * G1t_kp * G1d_p * pk_kp * pk_p )
			* exp_factor;

	*b222tA = 2. * ( G2t_p_k * G1t_p * G1t_k * pk_p * pk_k
			+ G2t_kp_k * G1t_kp * G1t_k * pk_kp * pk_k
			+ G2t_p_kp * G1t_kp * G1t_p * pk_kp * pk_p )
			* exp_factor;
}

void bispectrum_1loop_II_III(histo_t* kk1,histo_t* kk2,histo_t* kk3,histo_t* qq,FLAG a,FLAG b,FLAG c, histo_t *bk222_abc, histo_t *bk321_abc)
{
	*bk222_abc = 0.;
	*bk321_abc = 0.;
	size_t ii,idim;
	histo_t *kk[3] = {kk1,kk2,kk3};
	histo_t pp[3][3],rr[3][3],p[3],r[3],k[3],q;
	
	for (ii=0;ii<3;ii++) {
		for (idim=0;idim<3;idim++) {
			pp[ii][idim] = kk[ii][idim] - qq[idim];
			rr[ii][idim] = kk[ii][idim] + qq[idim];
		}
		p[ii] = my_sqrt(pp[ii][0]*pp[ii][0] + pp[ii][1]*pp[ii][1] + pp[ii][2]*pp[ii][2]);
		r[ii] = my_sqrt(rr[ii][0]*rr[ii][0] + rr[ii][1]*rr[ii][1] + rr[ii][2]*rr[ii][2]);
		k[ii] = my_sqrt(kk[ii][0]*kk[ii][0] + kk[ii][1]*kk[ii][1] + kk[ii][2]*kk[ii][2]);
	}
	histo_t oqq[3] = {-qq[0],-qq[1],-qq[2]};
	q = my_sqrt(qq[0]*qq[0] + qq[1]*qq[1] + qq[2]*qq[2]);
	histo_t pk_q,pk_k[3],pk_p[3],pk_r[3];
	
	find_pk_lin(&q,&pk_q,1,POLY);
	for (ii=0;ii<3;ii++) {
		find_pk_lin(&(k[ii]),&(pk_k[ii]),1,POLY);
		find_pk_lin(&(p[ii]),&(pk_p[ii]),1,POLY);
		find_pk_lin(&(r[ii]),&(pk_r[ii]),1,POLY);
	}
	
	if ((p[0]>q) && (r[1]>q)) {
		*bk222_abc += F2_sym_full(a, pp[0], qq) * F2_sym_full(b, rr[1], oqq) * F2_sym_full(c, rr[1], pp[0]) * pk_p[0] * pk_q * pk_r[1];
	}
	if ((r[0]>q) && (p[1]>q)) {
		*bk222_abc += F2_sym_full(a, rr[0], oqq) * F2_sym_full(b, pp[1], qq) * F2_sym_full(c, pp[1], rr[0]) * pk_r[0] * pk_q * pk_p[1]; 
	}
	if ((p[2]>q) && (r[1]>q)) {
		*bk222_abc += F2_sym_full(c, pp[2], qq) * F2_sym_full(b, rr[1], oqq) * F2_sym_full(a, rr[1], pp[2]) * pk_p[2] * pk_q * pk_r[1];
	}
	if ((r[2]>q) && (p[1]>q)) {
		*bk222_abc += F2_sym_full(c, rr[2], oqq) * F2_sym_full(b, pp[1], qq) * F2_sym_full(a, pp[1], rr[2]) * pk_r[2] * pk_q * pk_p[1]; 
	}
	if ((p[0]>q) && (r[2]>q)) {
		*bk222_abc += F2_sym_full(a, pp[0], qq) * F2_sym_full(c, rr[2], oqq) * F2_sym_full(b, rr[2], pp[0]) * pk_p[0] * pk_q * pk_r[2];
	}
	if ((r[0]>q) && (p[2]>q)) {
		*bk222_abc += F2_sym_full(a, rr[0], oqq) * F2_sym_full(c, pp[2], qq) * F2_sym_full(b, pp[2], rr[0]) * pk_r[0] * pk_q * pk_p[2]; 
	}
	*bk222_abc /= 2.;
	
	if (p[1]>q) {
		*bk321_abc += F3_sym(a, kk[2], pp[1], qq) * F2_sym_full(b, pp[1], qq) * pk_p[1] * pk_q * pk_k[2] 
		+ F3_sym(c, kk[0], pp[1], qq) * F2_sym_full(b, pp[1], qq) * pk_p[1] * pk_q * pk_k[0];
	}
	if (r[1]>q) {
		*bk321_abc += F3_sym(a, kk[2], rr[1], oqq) * F2_sym_full(b, rr[1], oqq) * pk_r[1] * pk_q * pk_k[2] 
		+ F3_sym(c, kk[0], rr[1], oqq) * F2_sym_full(b, rr[1], oqq) * pk_r[1] * pk_q * pk_k[0];
	}
	if (p[2]>q) {
		*bk321_abc += F3_sym(a, kk[1], pp[2], qq) * F2_sym_full(c, pp[2], qq) * pk_p[2] * pk_q * pk_k[1] 
		+ F3_sym(b, kk[0], pp[2], qq) * F2_sym_full(c, pp[2], qq) * pk_p[2] * pk_q * pk_k[0];
	}
	if (r[2]>q) {
		*bk321_abc += F3_sym(a, kk[1], rr[2], oqq) * F2_sym_full(c, rr[2], oqq) * pk_r[2] * pk_q * pk_k[1] 
		+ F3_sym(b, kk[0], rr[2], oqq) * F2_sym_full(c, rr[2], oqq) * pk_r[2] * pk_q * pk_k[0];
	}
	if (p[0]>q) {
		*bk321_abc += F3_sym(b, kk[2], pp[0], qq) * F2_sym_full(a, pp[0], qq) * pk_p[0] * pk_q * pk_k[2] 
		+ F3_sym(c, kk[1], pp[0], qq) * F2_sym_full(a, pp[0], qq) * pk_p[0] * pk_q * pk_k[1];
	}
	if (r[0]>q) {
		*bk321_abc += F3_sym(b, kk[2], rr[0], oqq) * F2_sym_full(a, rr[0], oqq) * pk_r[0] * pk_q * pk_k[2] 
		+ F3_sym(c, kk[1], rr[0], oqq) * F2_sym_full(a, rr[0], oqq) * pk_r[0] * pk_q * pk_k[1];
	}
	
	*bk222_abc /= power(M_PI,3);
	*bk321_abc /= power(M_PI,3);
	
}
