#ifndef _REGPT_REGPT_
#define _REGPT_REGPT_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include "define.h"
#include "common.h"
#include "kernels.h"

#define STEP_VERBOSE 10

static FLAG a,b;
static INTERPOL interpol_pk_lin = POLY;
static Terms2Loop terms_2loop;
static TermsBias terms_bias;
static TermsAB terms_A_B;

void print_num_threads()
{
	//Calculate number of threads
	size_t num_threads=0;
#pragma omp parallel
	{
#pragma omp atomic
		num_threads++;
	}
	printf(" - Using %zu threads\n",num_threads);
}


void set_num_threads(size_t num_threads)
{
	omp_set_num_threads(num_threads);
#ifdef _VERBOSE
	print_num_threads();
#endif //_VERBOSE
}

void print_pk_lin()
{
	printf("*** Linear power spectrum\n");
	printf(" - #modes: %zu\n",pk_lin.nk);
	printf(" - Range: %.3g < k < %.3g\n",pk_lin.k[0],pk_lin.k[pk_lin.nk-1]);
}

void set_pk_lin(histo_t* k,histo_t* pk,size_t nk)
{
	pk_lin.k = k;
	pk_lin.pk = pk;
	pk_lin.nk = nk;
#ifdef _VERBOSE
	print_pk_lin();
	printf("\n");
#endif //_VERBOSE
}

void print_k(histo_t *k,size_t nk)
{
	printf("*** k-output\n");
	printf(" - #modes: %zu\n",nk);
	printf(" - Range: %.3g < k < %.3g\n",k[0],k[nk-1]);
}


void set_k_2loop(size_t nk,histo_t* k,histo_t* pk_lin,histo_t* sigma_v2,histo_t* G1a_1loop,histo_t* G1a_2loop,histo_t* G1b_1loop,histo_t* G1b_2loop,histo_t* pkcorr_G2_tree_tree,histo_t *pkcorr_G2_tree_1loop,histo_t *pkcorr_G2_1loop_1loop,histo_t* pkcorr_G3_tree)
{
	terms_2loop.nk = nk;
	terms_2loop.k = k;
	terms_2loop.pk_lin = pk_lin;
	terms_2loop.sigma_v2 = sigma_v2;
	terms_2loop.G1a_1loop = G1a_1loop;
	terms_2loop.G1a_2loop = G1a_2loop;
	terms_2loop.G1b_1loop = G1b_1loop;
	terms_2loop.G1b_2loop = G1b_2loop;
	terms_2loop.pkcorr_G2_tree_tree = pkcorr_G2_tree_tree;
	terms_2loop.pkcorr_G2_tree_1loop = pkcorr_G2_tree_1loop;
	terms_2loop.pkcorr_G2_1loop_1loop = pkcorr_G2_1loop_1loop;
	terms_2loop.pkcorr_G3_tree = pkcorr_G3_tree;

#ifdef _VERBOSE
	print_k(terms_2loop.k,nk);
	printf("\n");
#endif //_VERBOSE
}

void print_flags(FLAG a,FLAG b)
{
	if ((a==DELTA)&&(b==DELTA)) printf(" - Calculating P_delta_delta\n");
	if ((a==DELTA)&&(b==THETA)) printf(" - Calculating P_delta_theta\n");
	if ((a==THETA)&&(b==DELTA)) printf(" - Calculating P_theta_delta\n");
	if ((a==THETA)&&(b==THETA)) printf(" - Calculating P_theta_theta\n");
}

FLAG set_flag(char* f)
{
	if (!strcmp(f,"delta")) return DELTA;
	if (!strcmp(f,"theta")) return THETA;
	return DELTA;
}

void set_precision_pk_lin(char* interpol_)
{
	set_interpol(&interpol_pk_lin,interpol_);
}

void calc_running_sigma_v2(histo_t *k,histo_t *sigmav2,size_t nk)
{
	//sigmav^2 = int_0^{k/2} dq P0(q)/(6*pi^2)
	size_t ik,iklin=0;
	histo_t sigmav2_ = 0.;
	for (ik=0;ik<nk;ik++) {
		histo_t kmax = k[ik]/2.;
		if (kmax < pk_lin.k[0]) {
			sigmav2[ik] = 0.;
			continue;
		}
		for (iklin=iklin;iklin<pk_lin.nk-1;iklin++) {
			if (kmax < pk_lin.k[iklin+1]) {
				histo_t pk_kmax = interpol_lin(kmax,pk_lin.k[iklin],pk_lin.k[iklin+1],pk_lin.pk[iklin],pk_lin.pk[iklin+1]);
				sigmav2[ik] = (sigmav2_ + (pk_kmax + pk_lin.pk[iklin]) * (kmax - pk_lin.k[iklin]))/(12.*M_PI*M_PI);
				break;
			}
			else sigmav2_ += (pk_lin.pk[iklin+1] + pk_lin.pk[iklin]) * (pk_lin.k[iklin+1] - pk_lin.k[iklin]);
		}
		if (iklin==pk_lin.nk-1) sigmav2[ik] = sigmav2_/(12.*M_PI*M_PI);
	}
}

void run_2loop(char* a_,char* b_,size_t num_threads)
{

	a = set_flag(a_);
	b = set_flag(b_);
#ifdef _VERBOSE
	printf("*** Calculation at two-loop order\n");
	print_flags(a,b);
#endif //_VERBOSE
	set_num_threads(num_threads);
	size_t ik=0;
	size_t step_verbose = MAX(terms_2loop.nk*STEP_VERBOSE/100,1);

	timer(0);
	find_pk_lin(terms_2loop.k,terms_2loop.pk_lin,terms_2loop.nk,interpol_pk_lin);
	calc_running_sigma_v2(terms_2loop.k,terms_2loop.sigma_v2,terms_2loop.nk);
#pragma omp parallel default(none) shared(terms_2loop,a,b,step_verbose) private(ik)
	{
		init_gamma1();
		init_gamma2();
		init_gamma2d();
		init_gamma2v();
#pragma omp for nowait schedule(dynamic)
		for (ik=0;ik<terms_2loop.nk;ik++) {

#ifdef _VERBOSE
			if (ik % step_verbose == 0) printf(" - Computation done at %zu percent.\n",ik*STEP_VERBOSE/step_verbose);
#endif //_VERBOSE
			histo_t k = terms_2loop.k[ik];
			calc_pkcorr_from_gamma1(a,b,k,&(terms_2loop.G1a_1loop[ik]),&(terms_2loop.G1a_2loop[ik]),&(terms_2loop.G1b_1loop[ik]),&(terms_2loop.G1b_2loop[ik]));
			calc_pkcorr_from_gamma2(a,b,k,&(terms_2loop.pkcorr_G2_tree_tree[ik]),&(terms_2loop.pkcorr_G2_tree_1loop[ik]),&(terms_2loop.pkcorr_G2_1loop_1loop[ik]));
		}
#pragma omp critical
		{
			free_gamma1();
			free_gamma2();
			free_gamma2d();
			free_gamma2v();
		}
	}
	init_gamma3();
	for (ik=0;ik<terms_2loop.nk;ik++) calc_pkcorr_from_gamma3(a,b,terms_2loop.k[ik],&(terms_2loop.pkcorr_G3_tree[ik]));
	free_gamma3();
	
#ifdef _VERBOSE
	timer(1);
#endif //_VERBOSE
#ifdef _DEBUG
	write_2loop(terms_2loop,"debug_2loop.dat");
#endif //_DEBUG
}


void set_k_bias(size_t nk,histo_t* k,histo_t* pk_lin,histo_t* sigma_v2,histo_t* pkbias_b2d,histo_t* pkbias_bs2d,histo_t* pkbias_b2t,histo_t* pkbias_bs2t,histo_t* pkbias_b22,histo_t* pkbias_b2s2,histo_t* pkbias_bs22,histo_t* sigma3sq)
{
	terms_bias.nk = nk;
	terms_bias.k = k;
	terms_bias.pk_lin = pk_lin;
	terms_bias.sigma_v2 = sigma_v2;
	terms_bias.pkbias_b2d = pkbias_b2d;
	terms_bias.pkbias_bs2d = pkbias_bs2d;
	terms_bias.pkbias_b2t = pkbias_b2t;
	terms_bias.pkbias_bs2t = pkbias_bs2t;
	terms_bias.pkbias_b22 = pkbias_b22;
	terms_bias.pkbias_b2s2 = pkbias_b2s2;
	terms_bias.pkbias_bs22 = pkbias_bs22;
	terms_bias.sigma3sq = sigma3sq;
	
#ifdef _VERBOSE
	print_k(terms_bias.k,nk);
	printf("\n");
#endif //_VERBOSE
}

void run_bias(size_t num_threads)
{

#ifdef _VERBOSE
	printf("*** Calculation of bias terms\n");
#endif //_VERBOSE
	set_num_threads(num_threads);
	size_t ik=0;
	size_t step_verbose = MAX(terms_bias.nk*STEP_VERBOSE/100,1);

	timer(0);
	find_pk_lin(terms_bias.k,terms_bias.pk_lin,terms_bias.nk,interpol_pk_lin);
	calc_running_sigma_v2(terms_bias.k,terms_bias.sigma_v2,terms_bias.nk);
#pragma omp parallel default(none) shared(terms_bias,step_verbose) private(ik)
	{
		init_bias();
#pragma omp for nowait schedule(dynamic)
		for (ik=0;ik<terms_bias.nk;ik++) {
			histo_t k = terms_bias.k[ik];
#ifdef _VERBOSE
			if (ik % step_verbose == 0) printf(" - Computation done at %zu percent.\n",ik*STEP_VERBOSE/step_verbose);
#endif //_VERBOSE
			terms_bias.pkbias_b2d[ik] = calc_pkcorr_from_bias(DELTA,k,kernel_b2,1);
			terms_bias.pkbias_bs2d[ik] = calc_pkcorr_from_bias(DELTA,k,kernel_bs2,1);
			terms_bias.pkbias_b2t[ik] = calc_pkcorr_from_bias(THETA,k,kernel_b2,1);
			terms_bias.pkbias_bs2t[ik] = calc_pkcorr_from_bias(THETA,k,kernel_bs2,1);
			terms_bias.pkbias_b22[ik] = calc_pkcorr_from_bias(DELTA,k,kernel_b22,0);
			terms_bias.pkbias_b2s2[ik] = calc_pkcorr_from_bias(DELTA,k,kernel_b2s2,0);
			terms_bias.pkbias_bs22[ik] = calc_pkcorr_from_bias(DELTA,k,kernel_bs22,0);
			terms_bias.sigma3sq[ik] = calc_pkcorr_from_bias(DELTA,k,kernel_sigma3sq,0);
		}
#pragma omp critical
		{
			free_bias();
		}
	}
	
#ifdef _VERBOSE
	timer(1);
#endif //_VERBOSE
}

void set_k_A_B(size_t nk,histo_t* k,histo_t* pk_lin,histo_t* sigma_v2,histo_t* A,histo_t* B)
{
	terms_A_B.nk = nk;
	terms_A_B.k = k;
	terms_A_B.pk_lin = pk_lin;
	terms_A_B.sigma_v2 = sigma_v2;
	terms_A_B.A = A;
	terms_A_B.B = B;
	
#ifdef _VERBOSE
	print_k(terms_A_B.k,nk);
	printf("\n");
#endif //_VERBOSE
}

void run_A_B(size_t num_threads)
{

#ifdef _VERBOSE
	printf("*** Calculation of A and B terms\n");
#endif //_VERBOSE
	set_num_threads(num_threads);
	size_t ik=0;
	size_t step_verbose = MAX(terms_A_B.nk*STEP_VERBOSE/100,1);

	timer(0);
	find_pk_lin(terms_A_B.k,terms_A_B.pk_lin,terms_A_B.nk,interpol_pk_lin);
	calc_running_sigma_v2(terms_A_B.k,terms_A_B.sigma_v2,terms_A_B.nk);
#pragma omp parallel default(none) shared(terms_A_B,step_verbose) private(ik)
	{
		init_A_B();
#pragma omp for nowait schedule(dynamic)
		for (ik=0;ik<terms_A_B.nk;ik++) {
			histo_t k = terms_A_B.k[ik];
#ifdef _VERBOSE
			if (ik % step_verbose == 0) printf(" - Computation done at %zu percent.\n",ik*STEP_VERBOSE/step_verbose);
#endif //_VERBOSE
			calc_pkcorr_from_A(k,terms_A_B.pk_lin[ik],&(terms_A_B.A[ik*9]));
			calc_pkcorr_from_B(k,&(terms_A_B.B[ik*12]));
		}
#pragma omp critical
		{
			free_A_B();
		}
	}
	
#ifdef _VERBOSE
	timer(1);
#endif //_VERBOSE
}


/*
void run_2loop(char* a_,char* b_)
{

	a = set_flag(a_);
	b = set_flag(b_);
#ifdef _VERBOSE
	printf("*** Calculation at two-loop order\n");
	print_flags(a,b);
#endif //_VERBOSE
	size_t ik=0;
	size_t step_verbose = MAX(terms_2loop.nk*STEP_VERBOSE/100,1);
	init_gamma1();
	init_gamma2();
	init_gamma2d();
	init_gamma2v();
	init_gamma3();

	timer(0);
	for (ik=0;ik<terms_2loop.nk;ik++) {

#ifdef _VERBOSE
		if (ik % step_verbose == 0) printf(" - Direct computation done at %zu percent.\n",ik*STEP_VERBOSE/step_verbose);
#endif //_VERBOSE
		histo_t k = terms_2loop.k[ik];
		calc_pkcorr_from_gamma1(a,b,k,&(terms_2loop.G1a_1loop[ik]),&(terms_2loop.G1a_2loop[ik]),&(terms_2loop.G1b_1loop[ik]),&(terms_2loop.G1b_2loop[ik]));
		calc_pkcorr_from_gamma2(a,b,k,&(terms_2loop.pkcorr_G2_tree_tree[ik]),&(terms_2loop.pkcorr_G2_tree_1loop[ik]),&(terms_2loop.pkcorr_G2_1loop_1loop[ik]));
		calc_pkcorr_from_gamma3(a,b,k,&(terms_2loop.pkcorr_G3_tree[ik]));
	}
#ifdef _VERBOSE
	timer(1);
#endif //_VERBOSE
#ifdef _DEBUG
	write_2loop("debug_2loop.dat");
#endif //_DEBUG
	free_gamma1();
	free_gamma2();
	free_gamma2d();
	free_gamma2v();
	free_gamma3();
}
*/

/*
void calc_running_sigma_v2(histo_t k,histo_t *sigmav2_)
{
	//sigmav^2 = int_0^{k/2} dq P0(q)/(6*pi^2)
	
	histo_t kmax = k/2.;
	if (kmax < pk_lin.k[0]) {
		*sigmav2_ = 0.;
		return;
	}
	size_t ik;
	histo_t sigmav2 = 0.;
	for (ik=0;ik<pk_lin.nk-1;ik++) {
		if (kmax < pk_lin.k[ik+1]) {
			histo_t pk_kmax = interpol_lin(kmax,pk_lin.k[ik],pk_lin.k[ik+1],pk_lin.pk[ik],pk_lin.pk[ik+1]);
			sigmav2 += (pk_kmax + pk_lin.pk[ik]) * (kmax - pk_lin.k[ik]);
			break;
		}
		else sigmav2 += (pk_lin.pk[ik+1] + pk_lin.pk[ik]) * (pk_lin.k[ik+1] - pk_lin.k[ik]);		
	}
	*sigmav2_ = sigmav2/(12.*M_PI*M_PI);
}
*/

#endif //_REGPT_REGPT_
