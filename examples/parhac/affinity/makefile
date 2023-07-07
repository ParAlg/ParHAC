all: parhac

#parhac: Affinity-CPAM-CPAM-Diff Affinity-CPAM-CPAM Affinity-PAM-CPAM Affinity-PAM-CPAM-Diff Affinity-PAM-PAM 
parhac: Affinity-CPAM-CPAM-Diff

Affinity-CPAM-CPAM-Diff:		Affinity.cc
	g++ -O3 -DNDEBUG -DUSE_DIFF_ENCODING -mcx16 -march=native -DHOMEGROWN -pthread -std=c++17 -Wall -I../../../parlaylib/include -I../../../pam/include -I../../../include -I../ -o Affinity-CPAM-CPAM-Diff Affinity.cc  -L/usr/local/lib -ljemalloc

Affinity-CPAM-CPAM:		Affinity.cc
	g++ -O3 -g   -DNDEBUG -mcx16 -march=native -DHOMEGROWN -pthread -std=c++17 -Wall -I../../../parlaylib/include -I../../../pam/include -I../../../include -I../ -o Affinity-CPAM-CPAM Affinity.cc -L/usr/local/lib -ljemalloc

Affinity-PAM-CPAM-Diff:		Affinity.cc
	g++ -O3 -g -DUSE_DIFF_ENCODING -DUSE_PAM_UPPER   -DNDEBUG -mcx16 -march=native -DHOMEGROWN -pthread -std=c++17 -Wall -I../../../parlaylib/include -I../../../pam/include -I../../../include -I../ -o Affinity-PAM-CPAM-Diff Affinity.cc -L/usr/local/lib -ljemalloc

Affinity-PAM-CPAM:		Affinity.cc
	g++ -O3 -g -DUSE_PAM_UPPER   -DNDEBUG -mcx16 -march=native -DHOMEGROWN -pthread -std=c++17 -Wall -I../../../parlaylib/include -I../../../pam/include -I../../../include -I../ -o Affinity-PAM-CPAM Affinity.cc -L/usr/local/lib -ljemalloc

Affinity-PAM-PAM:		Affinity.cc
	g++ -O3 -g -DUSE_PAM -DUSE_PAM_UPPER   -DNDEBUG -mcx16 -march=native -DHOMEGROWN -pthread -std=c++17 -Wall -I../../../parlaylib/include -I../../../pam/include -I../../../include -I../ -o Affinity-PAM-PAM Affinity.cc -L/usr/local/lib -ljemalloc

clean:
	rm -f Affinity-CPAM-CPAM-Diff Affinity-CPAM-CPAM Affinity-PAM-CPAM Affinity-PAM-CPAM-Diff Affinity-PAM-PAM  MIS-CPAM-CPAM-Diff MIS-CPAM-CPAM MIS-PAM-CPAM MIS-PAM-CPAM-Diff MIS-PAM-PAM LDD-CPAM-CPAM-Diff LDD-CPAM-CPAM LDD-PAM-CPAM LDD-PAM-CPAM-Diff LDD-PAM-PAM Affinity-CPAM-CPAM-Diff-Seq Affinity-CPAM-CPAM-Seq Affinity-PAM-CPAM-Seq Affinity-PAM-CPAM-Diff-Seq Affinity-PAM-PAM-Seq MIS-CPAM-CPAM-Diff-Seq MIS-CPAM-CPAM-Seq MIS-PAM-CPAM-Seq MIS-PAM-CPAM-Diff-Seq MIS-PAM-PAM-Seq LDD-CPAM-CPAM-Diff-Seq LDD-CPAM-CPAM-Seq LDD-PAM-CPAM-Seq LDD-PAM-CPAM-Diff-Seq LDD-PAM-PAM-Seq
