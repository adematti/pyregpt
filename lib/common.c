#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "define.h"
#include "common.h"

//  Timing variables
#ifdef _HAVE_OMP
#include <omp.h>
static double relbeg,relend,absbeg,absend;
#else //_HAVE_OMP
#include <time.h>
static time_t relbeg,relend,absbeg,absend;
#endif //_HAVE_OMP

histo_t my_abs(histo_t x)
{
#ifdef _FLOAT32
	return fabsf(x);
#else
	return fabs(x);
#endif //_FLOAT32
}

histo_t my_log(histo_t x)
{
#ifdef _FLOAT32
	return logf(x);
#else
	return log(x);
#endif //_FLOAT32
}

histo_t my_exp(histo_t x)
{
#ifdef _FLOAT32
	return expf(x);
#else
	return exp(x);
#endif //_FLOAT32
}

histo_t my_pow(histo_t x,histo_t exp)
{
#ifdef _FLOAT32
	return powf(x,exp);
#else
	return pow(x,exp);
#endif //_FLOAT32
}

histo_t my_sqrt(histo_t x)
{
#ifdef _FLOAT32
	return sqrtf(x);
#else
	return sqrt(x);
#endif //_FLOAT32
}

histo_t my_sin(histo_t x)
{
#ifdef _FLOAT32
	return sinf(x);
#else
	return sin(x);
#endif //_FLOAT32
}

histo_t my_cos(histo_t x)
{
#ifdef _FLOAT32
	return cosf(x);
#else
	return cos(x);
#endif //_FLOAT32
}

void powers(histo_t x,histo_t *tab, const size_t npowers)
{
	tab[0] = 1.;
	tab[1] = x;
	size_t i;
	for (i=2;i<npowers;i++) tab[i] = tab[i-1]*x;
}

size_t get_dichotomy_index(histo_t x,histo_t *tab,size_t min,size_t max)
{
	if (min==max-1) return min;
	else {
		size_t ind=(min+max)/2;
		if (x<tab[ind]) return get_dichotomy_index(x,tab,min,ind);
		else if (x>tab[ind]) return get_dichotomy_index(x,tab,ind,max);
		else return ind;
	}
}

histo_t interpol_lin(histo_t x,histo_t xmin,histo_t xmax,histo_t ymin,histo_t ymax) 
{
	return (x-xmin)/(xmax-xmin)*(ymax-ymin) + ymin;
}

histo_t interpol_poly(histo_t x,histo_t *tabx,histo_t *taby,size_t nx) 
{
	size_t ix,jx,ixclose = 0;
	histo_t* c = (histo_t*) malloc(nx*sizeof(histo_t));
	histo_t* d = (histo_t*) malloc(nx*sizeof(histo_t));
	histo_t diff = my_abs(x-tabx[ixclose]);
	for (ix=0;ix<nx;ix++) {
		histo_t tmpdiff = my_abs(x-tabx[ix]);
        if (tmpdiff<diff) {
        	ixclose = ix;
        	diff = tmpdiff;
        }
        c[ix] = taby[ix];
        d[ix] = taby[ix];
	}
	//printf("%d %.3f\n",(int) ixclose,taby[ixclose]);
	histo_t y = taby[ixclose];
	ixclose -= 1;
	for (jx=1;jx<nx;jx++) {
		for (ix=0;ix<nx-jx;ix++) {
			histo_t den = (c[ix+1]-d[ix])/(tabx[ix]-tabx[ix+jx]);
			c[ix] = (tabx[ix]-x)*den;
			d[ix] = (tabx[ix+jx]-x)*den;
			//printf("%.3f %.3f %.3f %.3f\n",c[ix+1]-d[ix],tabx[ix]-tabx[ix+jx],c[ix],d[ix]);
		}
		histo_t dy;
		if (2*(ixclose+1)<nx-jx) dy = c[ixclose+1];
		else {
			dy = d[ixclose];
			ixclose -= 1;			
		}
		y += dy;
		//printf("%d %.3f\n",(int) ixclose,y);
	}
	free(c);
	free(d);
	return y;
}

void interpol_pk_lin(histo_t* k,histo_t* pk,size_t nk)
{
	size_t ik;
	size_t start=0;
	size_t end=pk_lin.nk;
	histo_t kstart = pk_lin.k[0];
	histo_t kend = pk_lin.k[end-1];
	for (ik=0;ik<nk;ik++) {
		if ((k[ik]>=kstart)&&(k[ik]<=kend)) {
			size_t ind = get_dichotomy_index(k[ik],pk_lin.k,start,end);
			start = ind;
			pk[ik] = interpol_lin(k[ik],pk_lin.k[ind],pk_lin.k[ind+1],pk_lin.pk[ind],pk_lin.pk[ind+1]);
		}
		else pk[ik] = 0.;
	}
}

histo_t extrapol_pk_lin(histo_t k)
{
	size_t ik;
	size_t start=MAX(0,((long) pk_lin.nk)-15);
	size_t end=pk_lin.nk;	
	histo_t dlnp_dlnk = 0.;
	for (ik=start;ik<end;ik++) dlnp_dlnk += my_log(pk_lin.pk[ik+1]/pk_lin.pk[ik-1])/my_log(pk_lin.k[ik+1]/pk_lin.k[ik-1]);
	histo_t n_eff = MAX(dlnp_dlnk/((histo_t) (end-start)),-3.);
	if (k>1000.) n_eff = 3.;
	return pk_lin.pk[end-1]*my_pow((k / pk_lin.k[end-1]),n_eff);
}

void find_pk_lin(histo_t* k,histo_t* pk,size_t nk,INTERPOL interpol)
{
	size_t ik;
	size_t start=0;
	size_t end=pk_lin.nk;
	histo_t kstart = pk_lin.k[0];
	histo_t kend = pk_lin.k[end-1];
	for (ik=0;ik<nk;ik++) {
		if (k[ik]>=kend) pk[ik] = extrapol_pk_lin(k[ik]);
		else if (k[ik]<kstart) pk[ik] = 0.;
		else {
			size_t ind = get_dichotomy_index(k[ik],pk_lin.k,start,end);
			start = ind;
			if (interpol==POLY) {
				size_t ikmin = MAX(0,((long) ind)-2);
				size_t ikmax = MIN(end-1,ind+2);
				pk[ik] = interpol_poly(k[ik],&(pk_lin.k[ikmin]),&(pk_lin.pk[ikmin]),ikmax-ikmin+1);
			}
			else pk[ik] = interpol_lin(k[ik],pk_lin.k[ind],pk_lin.k[ind+1],pk_lin.pk[ind],pk_lin.pk[ind+1]);
		}
	}
}

INTERPOL get_interpol(char *interpol)
{
	if (!strcmp(interpol,"poly")) return POLY;
	return LIN;
	
}

void nodes_weights_gauss_legendre(histo_t xmin,histo_t xmax,histo_t *x,histo_t *w,size_t n) {

	/*
	xmin: minimum x
	xmax: maximum x
	x: array of nodes
	w: array of weights
	n: size of x,w
	*/

	size_t ii,jj,mid = (n+1)/2;
	histo_t xmid = (xmax+xmin)/2.;
	histo_t xl = (xmax-xmin)/2.;
	histo_t p1=0.,p2=0.,p3=0.,pp=0.,z=0.,z1=0.;
	
	for (ii=0;ii<mid;ii++) {
	
		z = cos(M_PI*(ii+.75)/(n+.5));
		z1 = z + EPS + 1.;
		while (my_abs(z-z1)>EPS) {
			p1 = 1.;
			p2 = 0.;
			for (jj=0;jj<n;jj++) {
				p3 = p2;
				p2 = p1;
				p1 = ((2.*jj+1.)*z*p2-jj*p3)/(jj+1.);
			}
			pp=n*(z*p1-p2)/(z*z-1.);
			z1=z;
			z=z1-p1/pp;
		}
		x[ii] = xmid-xl*z;
		x[n-ii-1] = xmid+xl*z;
		w[ii] = 2.*xl/((1.-z*z)*pp*pp);
 		w[n-ii-1] = w[ii];

	}
}

void init_gauss_legendre_q(GaussLegendreQ *gauss_legendre,size_t nq,INTERPOL interpol,histo_t kmin_,histo_t kmax_)
{
	//histo_t kmin=pk_lin.k[0]*(1.+1e-6),kmax=pk_lin.k[pk_lin.nk-1]*(1.-1e-6);
	histo_t kmin=pk_lin.k[0],kmax=pk_lin.k[pk_lin.nk-1];
	if ((kmin_>0.)&&(kmax_>0.)) {
		kmin = kmin_;
		kmax = kmax_;
	}
	gauss_legendre->k = 1.;
	size_t size = sizeof(histo_t);
	gauss_legendre->nq = nq;
	gauss_legendre->x = (histo_t *) calloc(nq,size);
	gauss_legendre->q = (histo_t *) calloc(nq,size);
	gauss_legendre->w = (histo_t *) calloc(nq,size);
	gauss_legendre->pk = (histo_t *) calloc(nq,size);
	
	nodes_weights_gauss_legendre(my_log(kmin),my_log(kmax),gauss_legendre->x,gauss_legendre->w,gauss_legendre->nq); 
	
	size_t iq;
	for (iq=0;iq<nq;iq++) {
		gauss_legendre->q[iq] = my_exp(gauss_legendre->x[iq]); //independent of k
		gauss_legendre->x[iq] = gauss_legendre->q[iq];
	}
	find_pk_lin(gauss_legendre->q,gauss_legendre->pk,nq,interpol);
}

void update_gauss_legendre_q(GaussLegendreQ *gauss_legendre,histo_t k)
{
	gauss_legendre->k = k;
	size_t iq;
	for (iq=0;iq<gauss_legendre->nq;iq++) gauss_legendre->x[iq] = gauss_legendre->q[iq]/k;
}


void free_gauss_legendre_q(GaussLegendreQ *gauss_legendre)
{
	free(gauss_legendre->x);
	free(gauss_legendre->q);
	free(gauss_legendre->w);
	free(gauss_legendre->pk);
}


void init_gauss_legendre_mu(GaussLegendreMu *gauss_legendre,size_t nmu)
{
	histo_t mumin=-1.,mumax=1.;
	size_t size = sizeof(histo_t);
	//gauss_legendre->mumin = mumin;
	//gauss_legendre->mumax = mumax;
	gauss_legendre->nmu = nmu;
	gauss_legendre->muref = (histo_t *) calloc(nmu,size);
	gauss_legendre->wref = (histo_t *) calloc(nmu,size);
	gauss_legendre->mu = (histo_t *) calloc(nmu,size);
	gauss_legendre->w = (histo_t *) calloc(nmu,size);
	
	nodes_weights_gauss_legendre(mumin,mumax,gauss_legendre->muref,gauss_legendre->wref,gauss_legendre->nmu); 
	
	size_t imu;
	for (imu=0;imu<nmu;imu++) {
		gauss_legendre->mu[imu] = gauss_legendre->muref[imu];
		gauss_legendre->w[imu] = gauss_legendre->wref[imu];
	}
}

void update_gauss_legendre_mu(GaussLegendreMu *gauss_legendre,histo_t mumin,histo_t mumax)
{
	//gauss_legendre->mumin = mumin;
	//gauss_legendre->mumin = mumin;
	size_t imu;
	histo_t jacobian = (mumax-mumin)/2.;
	for (imu=0;imu<gauss_legendre->nmu;imu++) {
		gauss_legendre->mu[imu] = (gauss_legendre->muref[imu]+1.)*jacobian+mumin;
		gauss_legendre->w[imu] = gauss_legendre->wref[imu]*jacobian;
	}
}

void free_gauss_legendre_mu(GaussLegendreMu *gauss_legendre)
{
	free(gauss_legendre->muref);
	free(gauss_legendre->wref);
	free(gauss_legendre->mu);
	free(gauss_legendre->w);
}

void timer(size_t i)
{
	/////
	// Timing routine
	// timer(0) -> initialize relative clock
	// timer(1) -> read relative clock
	// timer(2) -> read relative clock and initialize it afterwards
	// timer(4) -> initialize absolute clock
	// timer(5) -> read absolute clock
#ifdef _HAVE_OMP
	if(i==0)
		relbeg=omp_get_wtime();
	else if(i==1) {
		relend=omp_get_wtime();
		printf("    Relative time ellapsed %.1f ms\n",1000*(relend-relbeg));
	}    
	else if(i==2) {
		relend=omp_get_wtime();
		printf("    Relative time ellapsed %.1f ms\n",1000*(relend-relbeg));
		relbeg=omp_get_wtime();
	}
	else if(i==4)
		absbeg=omp_get_wtime();
	else if(i==5) {
		absend=omp_get_wtime();
		printf("    Total time ellapsed %.1f ms \n",1000*(absend-absbeg));
	}
#else //_HAVE_OMP
	int diff;
	
	if(i==0)
		relbeg=time(NULL);
	else if(i==1) {
		relend=time(NULL);
		diff=(int)(difftime(relend,relbeg));
		printf("    Relative time ellapsed %02d:%02d:%02d \n",
		 diff/3600,(diff/60)%60,diff%60);
	}    
	else if(i==2) {
		relend=time(NULL);
		diff=(size_t)(difftime(relend,relbeg));
		printf("    Relative time ellapsed %02d:%02d:%02d \n",
		 diff/3600,(diff/60)%60,diff%60);
		relbeg=time(NULL);
	}
	else if(i==4)
		absbeg=time(NULL);
	else if(i==5) {
		absend=time(NULL);
		diff=(size_t)(difftime(absend,absbeg));
		printf("    Total time ellapsed %02d:%02d:%02d \n",
		 diff/3600,(diff/60)%60,diff%60);
	}
#endif //_HAVE_OMP
}

void error_open_file(char *fname)
{
	//////
	// Open error handler
	fprintf(stderr,"REGPT: Could not open file %s \n",fname);
	exit(1);
}

void write_gauss_legendre_q(GaussLegendreQ gauss_legendre,char *fn)
{
	//////
	// Writes gauss legendre terms into file fn, only used for debugging
	FILE *fr;
	size_t iq,nq=gauss_legendre.nq;
	fr=fopen(fn,"w");
	if (fr==NULL) error_open_file(fn);
	for (iq=0;iq<nq;iq++) fprintf(fr,"%f %f %f\n",gauss_legendre.q[iq],gauss_legendre.w[iq],gauss_legendre.pk[iq]);
	fclose(fr);
}

void write_gauss_legendre_mu(GaussLegendreMu gauss_legendre,char *fn)
{
	//////
	// Writes gauss legendre terms into file fn, only used for debugging
	FILE *fr;
	size_t imu,nmu=gauss_legendre.nmu;
	fr=fopen(fn,"w");
	if (fr==NULL) error_open_file(fn);
	for (imu=0;imu<nmu;imu++) fprintf(fr,"%f %f %f\n",gauss_legendre.mu[imu],gauss_legendre.w[imu]);
	fclose(fr);
}

void write_2loop(Terms2Loop terms_2loop,char *fn)
{
	//////
	// Writes 2-loop terms into file fn, only used for debugging
	FILE *fr;
	size_t ik;
	fr=fopen(fn,"w");
	if(fr==NULL) error_open_file(fn);
	for(ik=0;ik<terms_2loop.nk;ik++) {
		fprintf(fr,"%f %f %f %f %f %f %f %f %f\n",terms_2loop.k[ik],terms_2loop.G1a_1loop[ik],terms_2loop.G1a_2loop[ik],terms_2loop.G1b_1loop[ik],terms_2loop.G1b_2loop[ik],terms_2loop.pkcorr_G2_tree_tree[ik],terms_2loop.pkcorr_G2_tree_1loop[ik],terms_2loop.pkcorr_G2_1loop_1loop[ik],terms_2loop.pkcorr_G3_tree[ik]);
	}
	fclose(fr);
}
