// Warning: Does not compute ground truths ingestible by faiss.


#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <omp.h>
#include <sys/time.h>

#include "faiss/Index.h"

#include <fstream>
#include <cstdio>
#include <iostream>
#include <random>
#include <unordered_set>

/*****************************************************
 * I/O functions for fvecs and ivecs
 *****************************************************/

#define DC(classname) classname *ix = dynamic_cast<classname *>(index)

float *fvecs_read(const char *fname, int *d_out, size_t *n_out) {
    FILE *f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int d;
    fread(&d, sizeof(int), 1, f);
    assert((d > 0 && d < 1000000) || !"unreasonable dimension");
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    assert(sz % ((d + 1) * 4) == 0 || !"weird file size");
    size_t n = sz / ((d + 1) * 4);

    *d_out = d;
    *n_out = n;
    float *x = new float[n * (d + 1)];
    size_t nr = fread(x, sizeof(float), n * (d + 1), f);
    assert(nr == n * (d + 1) || !"could not read whole file");

    // shift array to remove row headers
    for (size_t i = 0; i < n; i++)
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

    fclose(f);
    return x;
}

// not very clean, but works as long as sizeof(int) == sizeof(float)
int *ivecs_read(const char *fname, int *d_out, size_t *n_out) {
    return (int *)fvecs_read(fname, d_out, n_out);
}

double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

void write_queries(const std::string &filename, const float *vectors,
                        size_t n, int d_in) {
    FILE *f = fopen(filename.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "could not open %s for writing\n", filename.c_str());
        perror("");
        abort();
    }
    for(size_t i=0;i < n; i++){
        fwrite(&d_in, sizeof(int), 1, f);
        fwrite(vectors + (i * d_in), sizeof(float), d_in, f);
    }
    // fwrite(&n, sizeof(size_t), 1, f); // number of queries
    // fwrite(&k, sizeof(int), 1, f);    // number of neighbors (top k)
    // fwrite(d_in, sizeof(int), 1, f);
    // fwrite(distances, sizeof(float), n * k, f);
    fclose(f);
}

/// Command like this: ./sample_queries ./sift1M/db.fvecs 10000 queries.fvecs
int main(int argc, char **argv) {
    std::cout << argc << " arguments" << std::endl;
    if (argc - 1 != 3) {
        printf("You should at least input 2 params: the dataset path, sample size, output filename (with .fvecs ending)\n");
        return 0;
    }
    std::string param1 = argv[1];
    std::string param2 = argv[2];
    std::string param3 = argv[3];

    std::string db = param1;
    std::string base = db.substr(0, db.find_last_of("/\\"));
    std::string output_filepath = base + "/" + param3;
    int sample_size = std::stoi(param2);

    // std::string db, query, gtI, gtD;
    // if (param1 == "sift10k") {
    //     db = "../data/sift10k/siftsmall_base.fvecs";
    //     query = "../data/sift10k/siftsmall_query.fvecs";
    // } else if (param1 == "sift1M") {
    //     db = "../data/sift1M/sift_base.fvecs";
    //     query = "../data/sift1M/sift_query.fvecs";
    // } else if (param1 == "bert") {
    //     db = "../data/bert/db.fvecs";
    //     query = "../data/bert/queries.fvecs";
    // } else {
    //     printf("Your dataset name is illegal\n");
    //     return 0;
    // }

    omp_set_num_threads(60);
    double t0 = elapsed();

    printf("[%.3f s] Loading database\n", elapsed() - t0);

    size_t nb;
    int d;
    float *xb = fvecs_read(db.c_str(), &d, &nb);

    printf("[%.3f s] Query not set, sampling queries from the database\n", elapsed() - t0);

    // Sample nq random queries from the database
    // int nq = nb * sample_fraction;
    int nq = sample_size;
    if (nq > nb) {
        printf("Error: Cannot sample more queries than available vectors\n");
        return 1;
    }

    float* xq = new float[nq * d];
    // sampling with replacement to produce valid i.i.d. data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> dis(0, nb - 1);

    for (size_t i = 0; i < nq; ++i) {
        size_t random_index = dis(gen);
        std::memcpy(xq + i * d, xb + random_index * d, d * sizeof(float));
    }

    // Print the first query (xq[0]):
    std::cout << "first query (xq[0]): ";
    for (int i = 0; i < d; i++) {
        std::cout << xq[i] << " ";
    }
    std::cout << std::endl;

    write_queries(output_filepath, xq, nq, d);
    printf("[%.3f s] Sampled queries written to %s\n", elapsed() - t0, output_filepath.c_str());

    delete[] xq;
    delete[] xb;
}