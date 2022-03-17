####################################################
###          User definable stuff
CC ?= gcc
OMPFLAG ?= -fopenmp
LIBDIR ?= ../lib

DEFINEOPTIONS =
#DEFINEOPTIONS += -D_CUBA15
#DEFINEOPTIONS += -D_FLOAT32
# COMPILER AND OPTIONS
OPT ?= $(OMPFLAG) $(DEFINEOPTIONS) -pedantic -Wall -Wextra -O3

#.c FILES
SRC = regpt.c common.c kernels.c gamma1_1loop.c gamma1_2loop.c gamma2d_1loop.c gamma2t_1loop.c calc_pkcorr_from_gamma1.c calc_pkcorr_from_gamma2.c calc_pkcorr_from_gamma3.c spectrum_1loop.c spectrum_2loop.c
SRC += calc_pkcorr_from_A_1loop.c spectrum_A_1loop.c bispectrum_1loop.c calc_pkcorr_from_A_2loop_I.c calc_pkcorr_from_A_2loop_II_III.c spectrum_A_2loop.c calc_pkcorr_from_B.c spectrum_B.c
SRC += calc_pkcorr_from_bias_1loop.c spectrum_bias_1loop.c
#.o FILES
OBJ := $(SRC:.c=.o)

vpath %.so $(LIBDIR)

regpt.so: $(OBJ)
	@mkdir -p $(LIBDIR)
	$(CC) $(OPT) -lm -shared $^ -o $(LIBDIR)/$@ $(CUBA)/libcuba.a

%.o: %.c
	$(CC) $(OPT) -I$(CUBA) -fPIC -o $@ -c $<

all: regpt.so

clean:
	rm -rf $(LIBDIR)/regpt.so
	rm $(OBJ)

.PHONY: all clean
