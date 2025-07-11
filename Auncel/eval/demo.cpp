/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <sys/time.h>

#include "faiss/AutoTune.h"
#include "faiss/IndexFlat.h"
#include "faiss/IndexIVFFlat.h"

#include <omp.h>
#define DC(classname) classname* ix = dynamic_cast<classname*>(index)

/**
 * To run this demo, please download the ANN_SIFT1M dataset from
 *
 *   http://corpus-texmex.irisa.fr/
 *
 * and unzip it to the sudirectory sift1M.
 **/

/*****************************************************
 * I/O functions for fvecs and ivecs
 *****************************************************/

/**
 * To run this demo, please download the ANN_SIFT1M dataset from
 *
 *   http://corpus-texmex.irisa.fr/
 *
 * and unzip it to the sudirectory sift1M.
 **/

/*****************************************************
 * I/O functions for fvecs and ivecs
 *****************************************************/

float* fvecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int d;
    fread(&d, 1, sizeof(int), f);
    assert((d > 0 && d < 1000000) || !"unreasonable dimension");
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    assert(sz % ((d + 1) * 4) == 0 || !"weird file size");
    size_t n = sz / ((d + 1) * 4);

    *d_out = d;
    *n_out = n;
    float* x = new float[n * (d + 1)];
    size_t nr = fread(x, sizeof(float), n * (d + 1), f);
    assert(nr == n * (d + 1) || !"could not read whole file");

    // shift array to remove row headers
    // int MAX_TO_MOVE = 1000;
    // printf("WARNING[ConANN]:: limited to only %d vectors to test functionality.\n", MAX_TO_MOVE);
    // *n_out = MAX_TO_MOVE;
    for (size_t i = 0; i < n; i++) {
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));
        // MAX_TO_MOVE --;
        // if (MAX_TO_MOVE <= 0) {
        //     break;
        // }
    }

    fclose(f);
    return x;
}

// not very clean, but works as long as sizeof(int) == sizeof(float)
int* ivecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    return (int*)fvecs_read(fname, d_out, n_out);
}

double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main() {
    omp_set_num_threads(32);
    double t0 = elapsed();

    // this is typically the fastest one.
    const char* index_key = "IVF1024,Flat";

    // these ones have better memory usage
    // const char *index_key = "Flat";
    // const char *index_key = "PQ32";
    // const char *index_key = "PCA80,Flat";
    // const char *index_key = "IVF4096,PQ8+16";
    // const char *index_key = "IVF4096,PQ32";
    // const char *index_key = "IMI2x8,PQ32";
    // const char *index_key = "IMI2x8,PQ8+16";
    // const char *index_key = "OPQ16_64,IMI2x8,PQ8+16";

    faiss::Index* index;

    size_t d;

    {
        printf("[%.3f s] Loading train set\n", elapsed() - t0);

        size_t nt;
        printf("WARNING[ConANN]: train queries should be different than the database!\n");
        // float* xt = fvecs_read("../data/sift10k/siftsmall_base.fvecs", &d, &nt);

         float* xt = fvecs_read("../data/bert/db.fvecs", &d, &nt);

        printf("[%.3f s] Preparing index \"%s\" d=%ld\n",
               elapsed() - t0,
               index_key,
               d);
        int nlist = 100;   // 1024 as per index_key
        printf("WARNING[ConANN]: hardcoded nlist to %d for testing purposes.\n", nlist);
        faiss::IndexFlatL2* flat_index = new faiss::IndexFlatL2(d);
        index = new faiss::IndexIVFFlat(flat_index, d, nlist, faiss::METRIC_L2);

        printf("[%.3f s] Training on %ld vectors\n", elapsed() - t0, nt);

        index->train(nt, xt);
        delete[] xt;
    }

    {
        printf("[%.3f s] Loading database\n", elapsed() - t0);

        size_t nb, d2;
        // float* xb = fvecs_read("../data/sift10k/siftsmall_base.fvecs", &d2, &nb);
        float* xb = fvecs_read("../data/bert/db.fvecs", &d2, &nb);

        assert(d == d2 || !"dataset does not have same dimension as train set");

        printf("[%.3f s] Indexing database, size %ld*%ld\n",
               elapsed() - t0,
               nb,
               d);

        index->add(nb, xb);

        delete[] xb;
    }

    size_t nq;
    float* xq;

    {
        printf("[%.3f s] Loading queries\n", elapsed() - t0);

        size_t d2;
        // xq = fvecs_read("../data/sift10k/siftsmall_query.fvecs", &d2, &nq);
        xq = fvecs_read("../data/bert/queries.fvecs", &d2, &nq);
        assert(d == d2 || !"query does not have same dimension as train set");
    }

    size_t k;                // nb of results per query in the GT
    faiss::idx_t* gt; // nq * k matrix of ground-truth nearest-neighbors

    {
        printf("[%.3f s] Loading ground truth for %ld queries\n",
               elapsed() - t0,
               nq);

        // load ground-truth and convert int to long
        size_t nq2;
        // int* gt_int = ivecs_read("../data/sift10k/siftsmall_groundtruth.ivecs", &k, &nq2);
        int* gt_int = ivecs_read("../data/bert/indices.fvecs", &k, &nq2);

        std::cout << "GT[0]=" << gt_int[0] << "\n";
        assert(nq2 == nq || !"incorrect nb of ground truth entries");

        gt = new faiss::idx_t[k * nq];
        for (int i = 0; i < k * nq; i++) {
            gt[i] = gt_int[i];
        }
        delete[] gt_int;
    }

    // Result of the auto-tuning
    std::string selected_params;
    nq /= 2;

    { // run auto-tuning

        printf("[%.3f s] Preparing auto-tune criterion 1-recall at 1 "
               "criterion, with k=%ld nq=%ld\n",
               elapsed() - t0,
               k,
               nq);

        faiss::IntersectionCriterion crit(nq, 100);
        crit.set_groundtruth(k, nullptr, gt);
        crit.nnn = k; // by default, the criterion will request only 1 NN

        printf("[%.3f s] Preparing auto-tune parameters\n", elapsed() - t0);

        faiss::ParameterSpace params;
        params.initialize(index);

        printf("[%.3f s] Auto-tuning over %ld parameters (%ld combinations)\n",
               elapsed() - t0,
               params.parameter_ranges.size(),
               params.n_combinations());

        faiss::OperatingPoints ops;
        params.explore(index, nq, xq, crit, &ops);

        printf("[%.3f s] Found the following operating points: \n",
               elapsed() - t0);

        ops.display();

        // keep the first parameter that obtains > 0.5 1-recall@1
        int ind = 0;
        for (; ind < ops.optimal_pts.size(); ind++) {
            // std::cout << ops.optimal_pts[ind].key << "\n";
            if (ops.optimal_pts[ind].perf >= 0.9) {
                selected_params = ops.optimal_pts[ind].key;
                break;
            }
        }
        assert(selected_params.size() >= 0 ||
               !"could not find good enough op point");
    }

    { // Use the found configuration to perform a search

        faiss::ParameterSpace params;

        printf("[%.3f s] Setting parameter configuration \"%s\" on index\n",
               elapsed() - t0,
               selected_params.c_str());

        params.set_index_parameters(index, selected_params.c_str());

        printf("[%.3f s] Perform a search on %ld queries\n",
               elapsed() - t0,
               nq);

        // output buffers
        faiss::idx_t* I = new faiss::idx_t[nq * k];
        float* D = new float[nq * k];

        index->search(nq, xq, k, D, I);

        printf("[%.3f s] Compute recalls\n", elapsed() - t0);

        // evaluate result by hand.
        int n_1 = 0, n_10 = 0, n_100 = 0;
        for (int i = 0; i < nq; i++) {
            int gt_nn = gt[i * k];
            for (int j = 0; j < k; j++) {
                if (I[i * k + j] == gt_nn) {
                    if (j < 1)
                        n_1++;
                    if (j < 10)
                        n_10++;
                    if (j < 100)
                        n_100++;
                }
            }
        }
        printf("R@1 = %.4f\n", n_1 / float(nq));
        printf("R@10 = %.4f\n", n_10 / float(nq));
        printf("R@100 = %.4f\n", n_100 / float(nq));

        delete[] I;
        delete[] D;
    }

    { // ConANN block
        // Run auto-tuning with a 10% error bound
        printf("[%.3f s] Preparing auto-tune criterion 1-recall at 1 with k=%ld nq=%ld\n",
            elapsed() - t0, k, nq);

        // Use IntersectionCriterion to calculate recall@1 with a tolerance
        faiss::IntersectionCriterion crit(nq, 100);
        crit.set_groundtruth(k, nullptr, gt);
        crit.nnn = k; // The number of nearest neighbors

        printf("[%.3f s] Preparing auto-tune parameters\n", elapsed() - t0);

        faiss::ParameterSpace params;
        params.initialize(index);

        printf("[%.3f s] Auto-tuning over %ld parameters (%ld combinations)\n",
            elapsed() - t0,
            params.parameter_ranges.size(),
            params.n_combinations());

        faiss::OperatingPoints ops;
        params.explore(index, nq, xq, crit, &ops);

        printf("[%.3f s] Found the following operating points: \n", elapsed() - t0);
        ops.display();

        // Select a parameter configuration that meets the error bound
        int ind = 0;
        for (; ind < ops.optimal_pts.size(); ind++) {
            float recall = ops.optimal_pts[ind].perf;
            if (1.0 - recall <= 0.1) { // 10% error bound
                selected_params = ops.optimal_pts[ind].key;
                break;
            }
        }
        assert(!selected_params.empty() || !"Could not find a parameter set within the error bound");
        printf("[%.3f s] Setting parameter configuration \"%s\" on index\n",
               elapsed() - t0,
               selected_params.c_str());
    }


    delete[] xq;
    delete[] gt;
    delete index;
    return 0;
}