
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

#include "faiss/AutoTune.h"
#include "faiss/IndexFlat.h"
#include "faiss/IndexIVFFlat.h"
#include "faiss/index_io.h"

#include <fstream>
#include <cstdio>
#include <iostream>
#include <random>

/*****************************************************
 * I/O functions for fvecs and ivecs
 *****************************************************/

#define DC(classname) classname *ix = dynamic_cast<classname *>(index)

float *fvecs_read(const char *fname, size_t *d_out, size_t *n_out) {
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
int *ivecs_read(const char *fname, size_t *d_out, size_t *n_out) {
    return (int *)fvecs_read(fname, d_out, n_out);
}

double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

void write_gt_indices(const std::string &filename, const int *int_indices,
                      size_t n, int input_k, int out_k) {
    FILE *f = fopen(filename.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "could not open %s for writing\n", filename.c_str());
        perror("");
        abort();
    }

    for(size_t i=0;i < n; i++){
        fwrite(&out_k, sizeof(int), 1, f);
        fwrite(int_indices + (i * input_k), sizeof(int), out_k, f);
    }
    // fwrite(&n, sizeof(size_t), 1, f); // number of queries
    // fwrite(&k, sizeof(int), 1, f);    // number of neighbors (top k)
    // fwrite(d_in, sizeof(int), 1, f);
    // fwrite(indices, sizeof(int), n * k, f);
    fclose(f);
}

void write_gt_distances(const std::string &filename, const float *distances,
                        size_t n, int input_k, int out_k) {
    FILE *f = fopen(filename.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "could not open %s for writing\n", filename.c_str());
        perror("");
        abort();
    }
    for(size_t i=0;i < n; i++){
        fwrite(&out_k, sizeof(int), 1, f);
        fwrite(distances + (i * input_k), sizeof(float), out_k, f);
    }
    // fwrite(&n, sizeof(size_t), 1, f); // number of queries
    // fwrite(&k, sizeof(int), 1, f);    // number of neighbors (top k)
    // fwrite(d_in, sizeof(int), 1, f);
    // fwrite(distances, sizeof(float), n * k, f);
    fclose(f);
}

/// Command like this: ./compute_gt gist 100
int main(int argc, char **argv) {
    std::cout << argc << " arguments" << std::endl;
    if (argc - 1 != 2) {
        printf("You should at least input 2 params: the dataset name, k \n");
        return 0;
    }
    std::string param1 = argv[1];
    std::string param2 = argv[2];

    int input_k = std::stoi(param2);

    std::string db, query, gtI, gtD;
    if (param1 == "sift10k") {
        db = "../../data/sift10k/siftsmall_base.fvecs";
        query = "../../data/sift10k/siftsmall_query.fvecs";
    } else if (param1 == "sift1M") {
        db = "../../data/sift1M/sift_base.fvecs";
        query = "../../data/sift1M/sift_query.fvecs";
    } else if (param1 == "bert") {
        db = "../../data/bert/db.fvecs";
        query = "../../data/bert/queries.fvecs";
    } else if (param1 == "gist") {
        db = "../../data/gist/gist_base.fvecs";
        query = "../../data/gist/queries.fvecs";
    } else if (param1 == "glove") {
        db = "../../data/glove/db.fvecs";
        query = "../../data/glove/queries.fvecs";
    } else if (param1 == "deep10M") {
        db = "../../data/deep10M/deep10M.fvecs";
        query = "../../data/deep10M/queries.fvecs";
    } else {
        printf("Your dataset name is illegal\n");
        return 0;
    }

    omp_set_num_threads(32);
    double t0 = elapsed();

    // this is typically the fastest one.
    const char *index_key = "IndexFlatL2";

    printf("[%.3f s] Loading database\n", elapsed() - t0);

    size_t nb, d;
    float *xb = fvecs_read(db.c_str(), &d, &nb);

    printf("[%.3f s] Indexing database, size %ld*%ld\n", elapsed() - t0, nb, d);

    faiss::IndexFlatL2 exact_index(d);
    exact_index.add(nb, xb);
    size_t nq;
    float *xq;

    // if (query.empty()) {
    //     printf("[%.3f s] Query not set, sampling 1k queries from the database\n", elapsed() - t0);

    //     // Sample 1000 random queries from the database
    //     nq = 1000;
    //     xq = new float[nq * d];

    //     std::random_device rd;
    //     std::mt19937 gen(rd());
    //     std::uniform_int_distribution<size_t> dis(0, nb - 1);

    //     for (size_t i = 0; i < nq; ++i) {
    //         size_t random_index = dis(gen);
    //         std::memcpy(xq + i * d, xb + random_index * d, d * sizeof(float));
    //     }

    //     std::string output_filepath = "../data/bert/queries.fvecs";
    //     write_fvecs(output_filepath, xq, nq, d);
    //     printf("[%.3f s] Sampled queries written to %s\n", elapsed() - t0, output_filepath.c_str());
    // } else {
        printf("[%.3f s] Loading queries\n", elapsed() - t0);

        size_t d2;
        xq = fvecs_read(query.c_str(), &d2, &nq);
        assert(d == d2 || !"query does not have same dimension as train set");
    // }

    delete[] xb;

    faiss::idx_t *gt_indices = new faiss::idx_t[nq * input_k];
    float *gt_distances = new float[nq * input_k];

    printf("[%.3f s] Computing gts...\n", elapsed() - t0);
    exact_index.search(nq, xq, input_k, gt_distances, gt_indices);

    // conversion to integer
    int* int_indices = new int[input_k * nq];
    for (int i = 0; i < nq * input_k; i++) {
        int_indices[i] = gt_indices[i];
    }

    // Print gt_indices and gt_distances for the first query (xq[0]):
    // std::cout << "gt_indices for the first query (xq[0]): ";
    // for (int i = 0; i < input_k; i++) {
    //     std::cout << gt_indices[i] << " "; // Indices are integers
    // }
    // std::cout << std::endl;

    // std::cout << "gt_distances for the first query (xq[0]): ";
    // for (int i = 0; i < input_k; i++) {
    //     std::cout << gt_distances[i] << " "; // Distances should be floats
    // }
    // std::cout << std::endl;
    // std::cout << "number of queries: " << nq << std::endl;

    printf("[%.3f s] Writing gts...\n", elapsed() - t0);
    std::string base = db.substr(0, db.find_last_of("/\\"));

    std::vector<int> output_ks = {10, 100, 1000};
    for (int output_k : output_ks) {
        std::string filename_idx = base + "/indices-" + std::to_string(output_k) + ".fvecs";
        std::string filename_dis = base + "/distances-" + std::to_string(output_k) + ".fvecs";
        write_gt_indices(filename_idx, int_indices, nq, input_k, output_k);
        write_gt_distances(filename_dis, gt_distances, nq, input_k, output_k);
    }

    delete[] xq;
    delete[] int_indices;
}