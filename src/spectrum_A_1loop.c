#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include "define.h"
#include "common.h"
#include "kernels.h"

#define NCOMP 5

#define STEP_VERBOSE 10

static INTERPOL interpol_pk_lin = POLY;
static histo_t uvcutoff = 0.5;
static TermsA terms_A;

void set_precision_A_1loop_pk_lin(char* interpol_)
{
	set_interpol(&interpol_pk_lin,interpol_);
}

void set_precision_A_1loop_sigmad2(histo_t uvcutoff_)
{
	uvcutoff = uvcutoff_;
}

void set_terms_A_1loop(size_t nk,histo_t* k,histo_t* pk_lin,histo_t* sigmad2,histo_t* A,histo_t* B)
{
	terms_A.nk = nk;
	terms_A.k = k;
	terms_A.pk_lin = pk_lin;
	terms_A.sigmad2 = sigmad2;
	terms_A.A = A;
	
	if (verbose == INFO) {
		print_k(terms_A.k,nk);
		printf("\n");
	}
}

void run_terms_A_1loop(size_t num_threads)
{
	if (verbose == INFO) printf("*** Calculation of A term at 1 loop\n");
	set_num_threads(num_threads);
	size_t ik=0;
	size_t step_verbose = MAX(terms_A.nk*STEP_VERBOSE/100,1);

	timer(0);
	find_pk_lin(terms_A.k,terms_A.pk_lin,terms_A.nk,interpol_pk_lin);
	calc_running_sigmad2(terms_A.k,terms_A.sigmad2,terms_A.nk,uvcutoff);
#pragma omp parallel default(none) shared(terms_A,step_verbose,verbose) private(ik)
	{
		init_A_1loop();
#pragma omp for schedule(static)
		for (ik=0;ik<terms_A.nk;ik++) {
			histo_t k = terms_A.k[ik];
			if ((verbose == INFO) && (ik % step_verbose == 0)) printf(" - computation done at %zu percent\n",ik*STEP_VERBOSE/step_verbose);
			calc_pk_A_1loop(k,&(terms_A.A[ik*NCOMP]));
		}
#pragma omp critical
		{
			free_A_1loop();
		}
	}
	if (verbose == INFO) timer(1);
}

