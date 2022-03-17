#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include "cuba.h"
#include "define.h"
#include "common.h"

#define STEP_VERBOSE 10

void print_num_threads()
{
	//Calculate number of threads
	size_t num_threads=0;
#pragma omp parallel
	{
#pragma omp atomic
		num_threads++;
	}
	printf(" - using %zu threads\n",num_threads);
}

void set_verbosity(char* mode)
{
	if (!strcmp(mode,"quiet")) verbose = QUIET;
	if (!strcmp(mode,"info")) verbose = INFO;
	if (!strcmp(mode,"debug")) verbose = DEBUG;
}

void print_pk_lin()
{
	printf("*** Linear power spectrum\n");
	printf(" - #modes: %zu\n",pk_lin.nk);
	printf(" - range: %.3g < k < %.3g\n",pk_lin.k[0],pk_lin.k[pk_lin.nk-1]);
}

void print_k(histo_t *k,size_t nk)
{
	printf("*** k-output\n");
	printf(" - #modes: %zu\n",nk);
	printf(" - range: %.3g < k < %.3g\n",k[0],k[nk-1]);
}

void print_flags(FLAG a,FLAG b)
{
	if ((a==DELTA)&&(b==DELTA)) printf(" - calculating P_delta_delta\n");
	if ((a==DELTA)&&(b==THETA)) printf(" - calculating P_delta_theta\n");
	if ((a==THETA)&&(b==DELTA)) printf(" - calculating P_theta_delta\n");
	if ((a==THETA)&&(b==THETA)) printf(" - calculating P_theta_theta\n");
}

void set_num_threads(size_t num_threads)
{
	cubaaccel(0,1000); //default values
	if (num_threads>0) {
		omp_set_num_threads(num_threads);
		cubacores(num_threads,10000);
	}
	if (verbose == INFO) print_num_threads();
}

FLAG set_flag(char* flag)
{
	if (!strcmp(flag,"delta")) return DELTA;
	if (!strcmp(flag,"theta")) return THETA;
	return DELTA;
}

void set_pk_lin(histo_t* k,histo_t* pk,size_t nk)
{
	pk_lin.k = k;
	pk_lin.pk = pk;
	pk_lin.nk = nk;
	if (verbose == INFO) {
		print_pk_lin();
		printf("\n");
	}
}
