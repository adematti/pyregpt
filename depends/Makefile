CUBA_VERSION ?= 4.1
CC ?= gcc
DEST ?= src

TARBALL=Cuba-v$(CUBA_VERSION).tar.gz
UNPACK=Cuba-$(CUBA_VERSION)

$(TARBALL):
	curl -fsSL http://www.feynarts.de/cuba/Cuba-$(CUBA_VERSION).tar.gz > $(TARBALL)
	#wget -O $(TARBALL) http://www.feynarts.de/cuba/Cuba-$(CUBA_VERSION).tar.gz

$(UNPACK): $(TARBALL)
	tar -zxvf $(TARBALL)
	touch $@

libcuba.a: $(UNPACK)
	cd $(UNPACK); ./configure
	cd $(UNPACK); sed 's/CFLAGS = -O3 -fomit-frame-pointer/CFLAGS = -O3 -fPIC -fomit-frame-pointer/g' --in-place makefile
	cd $(UNPACK); make -B libcuba.a
	mkdir -p $(DEST)
	cp $(UNPACK)/*.h $(UNPACK)/*.a $(DEST)

install: libcuba.a
clean:
	rm -rf $(TARBALL)
	rm -rf $(UNPACK)
