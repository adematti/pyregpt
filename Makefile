####################################################
###          User definable stuff

EXEC = regpt

DEFINEOPTIONS =
#DEFINEOPTIONS += -D_CUBA15
#DEFINEOPTIONS += -D_FLOAT32
#GSL options
GSL_INC = /opt/rhel-6.x86_64/gnu4.6.2/gsl/1.15/include
GSL_LIB = /opt/rhel-6.x86_64/gnu4.6.2/gsl/1.15/lib
CUBA_INC = $(CUBA)
CUBA_LIB = $(CUBA)
CUBA_OBJ = $(CUBA)/*.o
REGPT_LIB = ./lib

### End of user-definable stuff
####################################################

LGSL = -L$(GSL_LIB) -lgsl -lgslcblas
#LCUBA = -L$(CUBA_LIB) -Wl,-whole-archive -lcuba
#LCUBA = -L$(CUBA_LIB)/libcuba.a
LCUBA = -L$(CUBA_LIB)

# DEFINES for the OpenMP version
DEFINEFLAGSCPU = $(DEFINEOPTIONS)

# COMPILER AND OPTIONS
COMPCPU = gcc
OPTCPU = -Wall -O3 -fopenmp $(DEFINEFLAGSCPU)
#OPTCPU = -Wextra -O3 -fopenmp $(DEFINEFLAGSCPU)

#INCLUDES AND LIBRARIES
INCLUDECOM = -I./lib -I$(GSL_INC) -I$(CUBA_INC)
LIBCPU = $(LGSL) $(LCUBA) -lm

#.c FILES

CREGPT = $(REGPT_LIB)/regpt.c
CCOMMON = $(REGPT_LIB)/common.c
CKERNELS = $(REGPT_LIB)/kernels.c
CGAMMA1_1LOOP = $(REGPT_LIB)/gamma1_1loop.c
CGAMMA1_2LOOP = $(REGPT_LIB)/gamma1_2loop.c
CGAMMA2d_1LOOP = $(REGPT_LIB)/gamma2d_1loop.c
CGAMMA2t_1LOOP = $(REGPT_LIB)/gamma2t_1loop.c
CPKCORR_GAMMA1 = $(REGPT_LIB)/calc_pkcorr_from_gamma1.c
CPKCORR_GAMMA2 = $(REGPT_LIB)/calc_pkcorr_from_gamma2.c
CPKCORR_GAMMA3 = $(REGPT_LIB)/calc_pkcorr_from_gamma3.c
CSPECTRUM_1LOOP = $(REGPT_LIB)/spectrum_1loop.c
CSPECTRUM_2LOOP = $(REGPT_LIB)/spectrum_2loop.c

CPKCORR_A_1LOOP = $(REGPT_LIB)/calc_pkcorr_from_A_1loop.c
CSPECTRUM_A_1LOOP = $(REGPT_LIB)/spectrum_A_1loop.c

CBISPECTRUM_1LOOP = $(REGPT_LIB)/bispectrum_1loop.c
CPKCORR_A_2LOOP_I = $(REGPT_LIB)/calc_pkcorr_from_A_2loop_I.c
CPKCORR_A_2LOOP_II_III = $(REGPT_LIB)/calc_pkcorr_from_A_2loop_II_III.c
CSPECTRUM_A_2LOOP = $(REGPT_LIB)/spectrum_A_2loop.c

CPKCORR_B = $(REGPT_LIB)/calc_pkcorr_from_B.c
CSPECTRUM_B = $(REGPT_LIB)/spectrum_B.c

CPKCORR_BIAS_1LOOP = $(REGPT_LIB)/calc_pkcorr_from_bias_1loop.c
CSPECTRUM_BIAS_1LOOP = $(REGPT_LIB)/spectrum_bias_1loop.c

SRC = $(CREGPT) $(CCOMMON) $(CKERNELS) $(CGAMMA1_1LOOP) $(CGAMMA1_2LOOP) $(CGAMMA2d_1LOOP) $(CGAMMA2t_1LOOP) $(CPKCORR_GAMMA1) $(CPKCORR_GAMMA2) $(CPKCORR_GAMMA3) $(CSPECTRUM_1LOOP) $(CSPECTRUM_2LOOP) $(CPKCORR_A_1LOOP) $(CSPECTRUM_A_1LOOP) $(CBISPECTRUM_1LOOP) $(CPKCORR_A_2LOOP_I) $(CPKCORR_A_2LOOP_II_III) $(CSPECTRUM_A_2LOOP) $(CPKCORR_B) $(CSPECTRUM_B) $(CPKCORR_BIAS_1LOOP) $(CSPECTRUM_BIAS_1LOOP)

#.o FILES

OBJ := $(SRC:.c=.o)

#RULES

default: $(EXEC)

all: $(EXEC)

$(EXEC): $(OBJ) $(CUBA_OBJ)
	$(COMPCPU) $(OPTCPU) -fPIC -shared $^ -o $@.so

%.o: %.c
	$(COMPCPU) $(OPTCPU) -fPIC -o $@ -c $< $(INCLUDECOM) $(LIBCPU)

#CLEANING RULES
clean :
	rm -f $(REGPT_LIB)/*.o
	rm -f ./*.so
