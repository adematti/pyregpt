#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include "define.h"
#include "common.h"

#define STEP_VERBOSE 10

static INTERPOL interpol_pk_lin = POLY;
static histo_t uvcutoff = 0.5;
static Terms2Loop terms_2loop;

void set_precision_spectrum_2loop_pk_lin(char* interpol_)
{
	set_interpol(&interpol_pk_lin,interpol_);
}

void set_running_uvcutoff_spectrum_2loop(histo_t uvcutoff_)
{
	uvcutoff = uvcutoff_;
}

void write_terms_spectrum_2loop(char *fn)
{
	// Writes 2-loop terms into file fn, only used for debugging
	FILE *fr;
	size_t ik;
	fr=fopen(fn,"w");
	if(fr==NULL) error_open_file(fn);
	for(ik=0;ik<terms_2loop.nk;ik++) {
		fprintf(fr,"%f %f %f %f %f %f %f %f %f\n",terms_2loop.k[ik],terms_2loop.gamma1a_1loop[ik],terms_2loop.gamma1a_2loop[ik],terms_2loop.gamma1b_1loop[ik],terms_2loop.gamma1b_2loop[ik],terms_2loop.pk_gamma2_tree_tree[ik],terms_2loop.pk_gamma2_tree_1loop[ik],terms_2loop.pk_gamma2_1loop_1loop[ik],terms_2loop.pk_gamma3_tree[ik]);
	}
	fclose(fr);
}

void set_terms_spectrum_2loop(size_t nk,histo_t* k,histo_t* pk_lin,histo_t* sigmad2,histo_t* gamma1a_1loop,histo_t* gamma1a_2loop,histo_t* gamma1b_1loop,histo_t* gamma1b_2loop,histo_t* pk_gamma2_tree_tree,histo_t *pk_gamma2_tree_1loop,histo_t *pk_gamma2_1loop_1loop,histo_t* pk_gamma3_tree)
{
	terms_2loop.nk = nk;
	terms_2loop.k = k;
	terms_2loop.pk_lin = pk_lin;
	terms_2loop.sigmad2 = sigmad2;
	terms_2loop.gamma1a_1loop = gamma1a_1loop;
	terms_2loop.gamma1a_2loop = gamma1a_2loop;
	terms_2loop.gamma1b_1loop = gamma1b_1loop;
	terms_2loop.gamma1b_2loop = gamma1b_2loop;
	terms_2loop.pk_gamma2_tree_tree = pk_gamma2_tree_tree;
	terms_2loop.pk_gamma2_tree_1loop = pk_gamma2_tree_1loop;
	terms_2loop.pk_gamma2_1loop_1loop = pk_gamma2_1loop_1loop;
	terms_2loop.pk_gamma3_tree = pk_gamma3_tree;

	if (verbose == INFO) {
		print_k(terms_2loop.k,nk);
		printf("\n");
	}
}


void run_terms_spectrum_2loop(char* a_,char* b_,size_t num_threads)
{
	FLAG a = set_flag(a_);
	FLAG b = set_flag(b_);
	if (verbose == INFO) {
		printf("*** Calculation of power spectrum at 2 loop\n");
		print_flags(a,b);
	}
	set_num_threads(num_threads);
	size_t ik=0;
	size_t step_verbose = MAX(terms_2loop.nk*STEP_VERBOSE/100,1);

	timer(0);
	find_pk_lin(terms_2loop.k,terms_2loop.pk_lin,terms_2loop.nk,interpol_pk_lin);
	calc_running_sigmad2(terms_2loop.k,terms_2loop.sigmad2,terms_2loop.nk,uvcutoff);
#pragma omp parallel default(none) shared(terms_2loop,a,b,step_verbose,verbose) private(ik)
	{
		init_gamma1_1loop();
		init_gamma1_2loop();
		init_gamma2_tree();
		init_gamma2d_1loop();
		init_gamma2t_1loop();
#pragma omp for schedule(static)
		for (ik=0;ik<terms_2loop.nk;ik++) {
			if ((verbose == INFO) && (ik % step_verbose == 0)) printf(" - computation I done at %zu percent\n",ik*STEP_VERBOSE/step_verbose);
			histo_t k = terms_2loop.k[ik];
			calc_pk_gamma1_2loop(a,b,k,&(terms_2loop.gamma1a_1loop[ik]),&(terms_2loop.gamma1a_2loop[ik]),&(terms_2loop.gamma1b_1loop[ik]),&(terms_2loop.gamma1b_2loop[ik]));
			calc_pk_gamma2_1loop(a,b,k,&(terms_2loop.pk_gamma2_tree_tree[ik]),&(terms_2loop.pk_gamma2_tree_1loop[ik]),&(terms_2loop.pk_gamma2_1loop_1loop[ik]));
		}
#pragma omp critical
		{
			free_gamma1_1loop();
			free_gamma1_2loop();
			free_gamma2_tree();
			free_gamma2d_1loop();
			free_gamma2t_1loop();
		}
	}
	init_gamma3_tree();
	for (ik=0;ik<terms_2loop.nk;ik++) {
		if ((verbose == INFO) && (ik % step_verbose == 0)) printf(" - computation II done at %zu percent\n",ik*STEP_VERBOSE/step_verbose);
		calc_pk_gamma3_tree(a,b,terms_2loop.k[ik],&(terms_2loop.pk_gamma3_tree[ik]));
	}
	free_gamma3_tree();
	if (verbose == INFO) timer(1);
	if (verbose == DEBUG) write_terms_spectrum_2loop("debug_terms_spectrum_2loop.dat");
}


void spectrum_2loop(histo_t *pk_2loop)
{
	size_t ik;
	for (ik=0;ik<terms_2loop.nk;ik++) {
		histo_t factor = 0.5 * terms_2loop.k[ik]*terms_2loop.k[ik] * terms_2loop.sigmad2[ik];
		histo_t exp_factor = my_exp(-factor);
		histo_t gamma1a_reg = exp_factor * (1. + factor + 0.5*factor*factor + terms_2loop.gamma1a_1loop[ik]*(1. + factor) + terms_2loop.gamma1a_2loop[ik]);
		histo_t gamma1b_reg = exp_factor * (1. + factor + 0.5*factor*factor + terms_2loop.gamma1b_1loop[ik]*(1. + factor) + terms_2loop.gamma1b_2loop[ik]);
		histo_t pk_gamma1 = gamma1a_reg * gamma1b_reg * terms_2loop.pk_lin[ik];
		//Taruya 2012 (arXiv 1208.1191v1) second term of eq 23
		histo_t pk_gamma2 = exp_factor*exp_factor * (terms_2loop.pk_gamma2_tree_tree[ik] * (1. + factor)*(1. + factor) + terms_2loop.pk_gamma2_tree_1loop[ik] * (1. + factor) + terms_2loop.pk_gamma2_1loop_1loop[ik]);
		//Taruya 2012 (arXiv 1208.1191v1) third term of eq 23
		histo_t pk_gamma3 = exp_factor*exp_factor * terms_2loop.pk_gamma3_tree[ik];
		pk_2loop[ik] = pk_gamma1 + pk_gamma2 + pk_gamma3;
	}
}

/*
void run_terms_spectrum_2loop(char* a_,char* b_,size_t num_threads)
{
	FLAG a = set_flag(a_);
	FLAG b = set_flag(b_);
	if (verbose == INFO) {
		printf("*** Calculation of power spectrum at 2 loop\n");
		print_flags(a,b);
	}
	set_num_threads(num_threads);
	size_t ik=0;
	size_t step_verbose = MAX(terms_2loop.nk*STEP_VERBOSE/100,1);

	timer(0);
	find_pk_lin(terms_2loop.k,terms_2loop.pk_lin,terms_2loop.nk,interpol_pk_lin);
	calc_running_sigmad2(terms_2loop.k,terms_2loop.sigmad2,terms_2loop.nk,uvcutoff);
#pragma omp parallel default(none) shared(terms_2loop,a,b,step_verbose,verbose) private(ik)
	{
		init_gamma1_1loop();
		init_gamma1_2loop();
		init_gamma2_tree();
		init_gamma2d_1loop();
		init_gamma2t_1loop();
		init_gamma3_tree();
#pragma omp for schedule(static)
		for (ik=0;ik<terms_2loop.nk;ik++) {
			if ((verbose == INFO) && (ik % step_verbose == 0)) printf(" - computation done at %zu percent\n",ik*STEP_VERBOSE/step_verbose);
			histo_t k = terms_2loop.k[ik];
			calc_pk_gamma1_2loop(a,b,k,&(terms_2loop.gamma1a_1loop[ik]),&(terms_2loop.gamma1a_2loop[ik]),&(terms_2loop.gamma1b_1loop[ik]),&(terms_2loop.gamma1b_2loop[ik]));
			calc_pk_gamma2_1loop(a,b,k,&(terms_2loop.pk_gamma2_tree_tree[ik]),&(terms_2loop.pk_gamma2_tree_1loop[ik]),&(terms_2loop.pk_gamma2_1loop_1loop[ik]));
			calc_pk_gamma3_tree(a,b,terms_2loop.k[ik],&(terms_2loop.pk_gamma3_tree[ik]));
		}
#pragma omp critical
		{
			free_gamma1_1loop();
			free_gamma1_2loop();
			free_gamma2_tree();
			free_gamma2d_1loop();
			free_gamma2t_1loop();
			free_gamma3_tree();
		}
	}
	if (verbose == INFO) timer(1);
	if (verbose == DEBUG) write_terms_spectrum_2loop("debug_terms_spectrum_2loop.dat");
}
*/
