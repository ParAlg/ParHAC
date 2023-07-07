all: parhac

#parhac: ParHac-CPAM-CPAM-Diff ParHac-CPAM-CPAM ParHac-PAM-CPAM ParHac-PAM-CPAM-Diff ParHac-PAM-PAM
parhac: ParHac-CPAM-CPAM-Diff  ParHac-CPAM-CPAM
#export LD_LIBRARY_PATH=/usr0/home/ldhulipa/tmp/jemalloc-5.2.1/lib;
ParHac-CPAM-CPAM-Diff:		ParHac.cc
	export LD_DEBUG=files; LD_LIBRARY_PATH=/usr0/home/ldhulipa/tmp/jemalloc-5.2.1/lib g++ -O3 -DNDEBUG -DUSE_DIFF_ENCODING -mcx16 -march=native -DHOMEGROWN -pthread -std=c++17 -Wall -I../../../parlaylib/include -I../../../pam/include -I../../../include -I../ -o ParHac-CPAM-CPAM-Diff ParHac.cc -L/usr0/home/ldhulipa/tmp/jemalloc-5.2.1/lib -ljemalloc

#-Wl,--rpath=/usr0/home/ldhulipa/tmp/jemalloc-5.2.1/lib \
#-Wl,--dynamic-linker=/usr0/home/ldhulipa/tmp/jemalloc-5.2.1/lib/libjemalloc.so
#-Wl,-rpath,/usr0/home/ldhulipa/tmp/jemalloc-5.2.1/lib -ljemalloc

ParHac-CPAM-CPAM:		ParHac.cc
	export LD_DEBUG=files; LD_LIBRARY_PATH=/usr0/home/ldhulipa/tmp/jemalloc-5.2.1/lib g++ -O3 -DNDEBUG -mcx16 -march=native -DHOMEGROWN -pthread -std=c++17 -Wall -I../../../parlaylib/include -I../../../pam/include -I../../../include -I../ -o ParHac-CPAM-CPAM ParHac.cc -L/usr0/home/ldhulipa/tmp/jemalloc-5.2.1/lib -ljemalloc

ParHac-PAM-CPAM-Diff:		ParHac.cc
	g++ -O3 -g -DUSE_DIFF_ENCODING -DUSE_PAM_UPPER   -DNDEBUG -mcx16 -march=native -DHOMEGROWN -pthread -std=c++17 -Wall -I../../../parlaylib/include -I../../../pam/include -I../../../include -I../ -o ParHac-PAM-CPAM-Diff ParHac.cc -L/usr/local/lib -ljemalloc

ParHac-PAM-CPAM:		ParHac.cc
	g++ -O3 -g -DUSE_PAM_UPPER   -DNDEBUG -mcx16 -march=native -DHOMEGROWN -pthread -std=c++17 -Wall -I../../../parlaylib/include -I../../../pam/include -I../../../include -I../ -o ParHac-PAM-CPAM ParHac.cc -L/usr/local/lib -ljemalloc

ParHac-PAM-PAM:		ParHac.cc
	g++ -O3 -g -DUSE_PAM -DUSE_PAM_UPPER   -DNDEBUG -mcx16 -march=native -DHOMEGROWN -pthread -std=c++17 -Wall -I../../../parlaylib/include -I../../../pam/include -I../../../include -I../ -o ParHac-PAM-PAM ParHac.cc -L/usr/local/lib -ljemalloc

clean:
	rm -f ParHac-CPAM-CPAM-Diff ParHac-CPAM-CPAM ParHac-PAM-CPAM ParHac-PAM-CPAM-Diff ParHac-PAM-PAM  MIS-CPAM-CPAM-Diff MIS-CPAM-CPAM MIS-PAM-CPAM MIS-PAM-CPAM-Diff MIS-PAM-PAM LDD-CPAM-CPAM-Diff LDD-CPAM-CPAM LDD-PAM-CPAM LDD-PAM-CPAM-Diff LDD-PAM-PAM ParHac-CPAM-CPAM-Diff-Seq ParHac-CPAM-CPAM-Seq ParHac-PAM-CPAM-Seq ParHac-PAM-CPAM-Diff-Seq ParHac-PAM-PAM-Seq MIS-CPAM-CPAM-Diff-Seq MIS-CPAM-CPAM-Seq MIS-PAM-CPAM-Seq MIS-PAM-CPAM-Diff-Seq MIS-PAM-PAM-Seq LDD-CPAM-CPAM-Diff-Seq LDD-CPAM-CPAM-Seq LDD-PAM-CPAM-Seq LDD-PAM-CPAM-Diff-Seq LDD-PAM-PAM-Seq
