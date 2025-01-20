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
#include<fstream>
#include <random>
#include <algorithm>

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <sys/time.h>

#include "faiss/AutoTune.h"
#include "faiss/IndexFlat.h"
#include "faiss/IndexIVFFlat.h"
#include "faiss/impl/FaissAssert.h"

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
    for (size_t i = 0; i < n; i++) {
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));
    }

    fclose(f);
    return x;
}

// not very clean, but works as long as sizeof(int) == sizeof(float)
int* ivecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    return (int*)fvecs_read(fname, d_out, n_out);
}

float* fbin_read(const char* fname, size_t* d_out, size_t* n_out, int num = 10000000, int bytes = 4) {
    FILE* f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    if (bytes == 1){
        int d,n;
        fread(&n, sizeof(int), 1, f);
        fread(&d, sizeof(int), 1, f);
        printf("d : %d, n: %d\n", d, n);
        assert((d > 0 && d < 1000000) || !"unreasonable dimension");
        *d_out = d;
        *n_out = n;
        int64_t total_size = int64_t(d) * num;
        int8_t* x = new int8_t[total_size];
        int64_t nr = 0;
        nr += fread(x, bytes, total_size, f);
        assert(nr == int64_t(d) * num || !"could not read whole file");
        fclose(f);
        float* fx = new float[total_size];
        for (int64_t ij = 0; ij < total_size; ij++){
            fx[ij] = float(x[ij]);
        }
        delete[] x;
        return fx;
    }
    else{
        int d,n;
        fread(&n, sizeof(int), 1, f);
        fread(&d, sizeof(int), 1, f);
        printf("d : %d, n: %d\n", d, n);
        assert((d > 0 && d < 1000000) || !"unreasonable dimension");
        *d_out = d;
        *n_out = n;
        int64_t total_size = int64_t(d) * num;
        float* x = new float[total_size];
        int64_t nr = 0;
        nr += fread(x, sizeof(float), total_size, f);
        assert(nr == int64_t(d) * num || !"could not read whole file");
        fclose(f);
        return x;
    }
}

// not very clean, but works as long as sizeof(int) == sizeof(float)
int* ibin_read(const char* fname, size_t* d_out, size_t* n_out, int num = 10000000, int bytes = 4) {
    return (int*)fbin_read(fname, d_out, n_out, num, bytes);
}

double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

/* type = 0 : L2, 1 : IP*/
size_t inter_sec(size_t max_topk, const float *gt, size_t topk, const float *I, int type = 0){
    size_t res = 0;
    float t_val = gt[topk-1];
    for(int i = 0; i < topk;i++){
        float c_val = I[i];
        if (c_val <= t_val + 1e-6 && type == 0)
            res++;
        if (c_val >= t_val - 1e-6 && type == 1)
            res++;
    }
    return res;
}

float calculate_fnr(const faiss::idx_t* query_indices, const faiss::idx_t* ground_truth, size_t nq_sampled, size_t k) {
    int false_negatives = 0;
    for (size_t i = 0; i < nq_sampled; i++) {

        // Create sets for the current query and ground truth
        std::unordered_set<faiss::idx_t> query_set(query_indices + i * k, query_indices + (i + 1) * k);
        std::unordered_set<faiss::idx_t> gt_set(ground_truth + i * k, ground_truth + (i + 1) * k);

        int local_fn = 0;
        // Measure the intersection between query set and ground truth set
        for (auto& gt_idx : gt_set) {
            if (query_set.find(gt_idx) == query_set.end()) {
                false_negatives++;
            }
        }
    }
    return (float) false_negatives / (nq_sampled * k);
}


int main(int argc,char **argv) {
    if(argc - 1 != 5){
        printf("You should at least input 5 params: the dataset name, train size, query size, topk and error bound\n");
        return 0;
    }
    std::string p1 = argv[1];
    std::string p2 = argv[2];
    std::string p3 = argv[3];
    std::string p4 = argv[4];
    std::string p5 = argv[5];

    int input_k = std::stoi(p4);
    float error_bound = std::stof(p5);
    int trains = std::stoi(p2);
    int tests = std::stoi(p3);

    std::string db, query, gtI, gtD;

    if(p1 == "sift1M"){
        db = "/workspace/data/sift/sift1M.fvecs";
        query = "/workspace/data/sift/1M_query.fvecs";
        gtI = "/workspace/data/sift/idx_1M.ivecs";
        gtD = "/workspace/data/sift/dis_1M.fvecs";
    } else if (p1 == "bert") {
        db = "../data/bert/db.fvecs";
        query = "../data/bert/queries.fvecs";
        gtI = "../data/bert/indices.fvecs";
        gtD = "../data/bert/distances.fvecs";
    }  
    else if(p1 == "sift10M"){
        db = "/workspace/data/sift/sift10M/sift10M.fvecs";
        query = "/workspace/data/sift/sift10M/query.fvecs";
        gtI = "/workspace/data/sift/sift10M/idx.ivecs";
        gtD = "/workspace/data/sift/sift10M/dis.fvecs";
    }
    else if(p1 == "deep10M"){
        db = "/workspace/data/deep/deep10M.fvecs";
        query = "/workspace/data/deep/query.fvecs";
        gtI = "/workspace/data/deep/idx.ivecs";
        gtD = "/workspace/data/deep/dis.fvecs";
    }
    else if(p1 == "gist"){
        db = "../data/gist/gist_base.fvecs";
        query = "../data/gist/gist_query.fvecs";
        gtI = "../data/gist/gist_groundtruth.ivecs";
        gtD = "../data/gist/dis.fvecs";
    }
    else if(p1 == "spacev"){
        db = "/workspace/data/spacev/spacev10M.fvecs";
        query = "/workspace/data/spacev/query.fvecs";
        gtI = "/workspace/data/spacev/idx.ivecs";
        gtD = "/workspace/data/spacev/dis.fvecs";
    }
    else if(p1 == "glove"){
        db = "/workspace/data/glove/glove.fvecs";
        query = "/workspace/data/glove/query.fvecs";
        gtI = "/workspace/data/glove/idx.ivecs";
        gtD = "/workspace/data/glove/dis.fvecs";
    }
    else if(p1 == "text"){
        db = "/workspace/data/text/text10M.fvecs";
        query = "/workspace/data/text/query.fvecs";
        gtI = "/workspace/data/text/idx.ivecs";
        gtD = "/workspace/data/text/dis.fvecs";
    }
    else{
        printf("Your dataset name is illegal\n");
        return 0;
    }

    omp_set_num_threads(16);
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

    faiss::IndexIVFFlat* index;

    size_t d;
    int nlist = 100;   // as per index_key

    {
        printf("[%.3f s] Loading train set\n", elapsed() - t0);

        size_t nt;
        float* xt = fvecs_read(db.c_str(), &d, &nt);

        printf("[%.3f s] Preparing index \"%s\" d=%ld\n",
               elapsed() - t0,
               index_key,
               d);
        // TODO(sonia): text
        // if (p1 == "text")
        //     index = faiss::index_factory(d, index_key, faiss::METRIC_INNER_PRODUCT);
        // else
        //    index = faiss::index_factory(d, index_key);

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
        float* xb = fvecs_read(db.c_str(), &d2, &nb);
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
        xq = fvecs_read(query.c_str(), &d2, &nq);
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
        int* gt_int = ivecs_read(gtI.c_str(), &k, &nq2);
        assert(nq2 == nq || !"incorrect nb of ground truth entries");

        gt = new faiss::idx_t[k * nq];
        for (int i = 0; i < k * nq; i++) {
            gt[i] = gt_int[i];
        }
        delete[] gt_int;
    }

    float* gt_D;

    {
        printf("[%.3f s] Loading ground truth distance for %ld queries\n",
               elapsed() - t0,
               nq);

        // load ground-truth and convert int to long
        size_t nq2;
        gt_D = fvecs_read(gtD.c_str(), &k, &nq2);
        assert(nq2 == nq || !"incorrect nb of ground truth entries");
    }

    // Result of the auto-tuning
    std::string selected_params;
    FAISS_ASSERT(nq == trains + tests);

    // { // run auto-tuning

    //     printf("[%.3f s] Preparing auto-tune with k=%ld nq=%ld\n",
    //            elapsed() - t0,
    //            k,
    //            trains);

    //     faiss::IntersectionCriterion crit(trains, input_k);
    //     crit.set_groundtruth(k, nullptr, gt);
    //     crit.nnn = k; // by default, the criterion will request only 1 NN

    //     printf("[%.3f s] Preparing auto-tune parameters\n", elapsed() - t0);

    //     faiss::ParameterSpace params;
    //     params.initialize(index);

    //     printf("[%.3f s] Auto-tuning over %ld parameters (%ld combinations)\n",
    //            elapsed() - t0,
    //            params.parameter_ranges.size(),
    //            params.n_combinations());

    //     faiss::OperatingPoints ops;
    //     params.explore(index, trains, xq, crit, &ops);

    //     printf("[%.3f s] Found the following operating points: \n",
    //            elapsed() - t0);

    //     // ops.display(); //

    //     // keep the first parameter that obtains > 0.5 1-recall@1
    //     int ind = 0;
    //     for (; ind < ops.optimal_pts.size(); ind++) {
    //         if (ops.optimal_pts[ind].perf >= (1 - error_bound)) {
    //             selected_params = ops.optimal_pts[ind].key;
    //             printf("[%.3f s] Optimal found: ",elapsed() - t0);
    //             std::cout << ops.optimal_pts[ind].key << "\n";
    //             break;
    //         }
    //     }
    //     assert(selected_params.size() >= 0 ||
    //            !"could not find good enough op point");
    // }

    // { // Use the found configuration to perform a search

    //     faiss::ParameterSpace params;

    //     params.set_index_parameters(index, selected_params.c_str());

    //     printf("[%.3f s] Perform parameter search on %ld queries according to Auncel\n",
    //            elapsed() - t0,
    //            tests);

    //     // output buffers
    //     omp_set_num_threads(1);

    //     faiss::idx_t* I = new faiss::idx_t[tests * input_k];
    //     float* D = new float[tests * input_k];

    //     // if(DC(faiss::IndexIVF)){
    //     //     // ix->nprobe = ix->nlist / 2;
    //     //     ix -> nprobe = 300;
    //     // }

    //     std::vector<double> perf;
    //     for (int i = 0; i < tests; i++) {
    //         auto tt0 = elapsed();
    //         index->search(1, xq + d * trains + d * i, input_k, D + i * input_k, I + i * input_k);
    //         auto tt1 = elapsed();
    //         perf.push_back(tt1 - tt0);
    //     }

    //     printf("[%.3f s] Compute Bound Error\n", elapsed() - t0);

    //     int type = 0;
    //     if (p1 == "text")
    //         type = 1;

    //     float minf = 1.;
    //     std::vector<float> error_per_query;

    //     for (int i = trains; i < tests + trains; i++) {
    //         float query_error = inter_sec(k, &gt_D[i * 100], 
    //                               input_k, D + (i - trains) * input_k , type) / float(input_k);
    //         minf = std::min(minf, query_error);
    //         error_per_query.push_back(query_error);
    //     }

    //     float sum = 0.;
    //     int count = 0;
    //     for (int i = trains; i < tests + trains; i++) {
    //         sum += inter_sec(k, &gt_D[i * 100], 
    //                         input_k, D + (i - trains) * input_k , type) / float(input_k);
    //         count++;
    //     }
    //     float avg = sum / count;
    //     printf("[%.3f s] Got average error %.6f\n", elapsed() - t0, avg);


    //     // Output the latency to file
    //     std::stringstream fn;
    //     fn<<"Faiss_Latency_Error" << "_" << p1 << "_" << input_k << "_" << int(error_bound*100) <<".log";
    //     std::string filename = fn.str();

    //     std::ofstream outfile;
    //     outfile.open(filename);
    //     for(int i = 0;i < tests; i++){
    //         outfile << perf[i] << "\t" << error_per_query[i] << std::endl;
    //     }
    //     outfile.close();

    //     delete[] I;
    //     delete[] D;
    // }

    {
        printf("[%.3f s] Perform parameter search on %ld queries according to ConANN\n",
               elapsed() - t0,
               tests);

        int optimal_nprobe = 0;

        // Sample 50% of queries
        size_t nq_sampled = tests;
        std::vector<size_t> sampled_indices(nq_sampled);
        std::iota(sampled_indices.begin(), sampled_indices.end(), 0);
        std::shuffle(sampled_indices.begin(), sampled_indices.end(), std::mt19937{std::random_device{}()});

        float* sampled_queries = new float[nq_sampled * d];
        for (size_t i = 0; i < nq_sampled; i++) {
            std::memcpy(sampled_queries + i * d, xq + sampled_indices[i] * d, d * sizeof(float));
        }

        faiss::idx_t* sampled_gt = new faiss::idx_t[nq_sampled * k];
        float* sampled_gt_D = new float[nq_sampled * k];
        for (size_t i = 0; i < nq_sampled; i++) {
            size_t query_idx = sampled_indices[i];
            std::memcpy(sampled_gt + i * k, gt + query_idx * k, k * sizeof(faiss::idx_t));
            std::memcpy(sampled_gt_D + i * k, gt_D + query_idx * k, k * sizeof(float));
        }

        // Iterate over nprobe values
        for (size_t nprobe = 1; nprobe <= nlist; nprobe++) {
            index->nprobe = nprobe;
            printf("[%.3f s] Testing nprobe = %ld\n", elapsed() - t0, nprobe);

            // Perform knn search
            std::vector<faiss::idx_t> I(nq_sampled * k);
            std::vector<float> D(nq_sampled * k);
            index->search(nq_sampled, sampled_queries, k, D.data(), I.data());

            // Calculate average FNR
            float avg_fnr = calculate_fnr(I.data(), sampled_gt, nq_sampled, k);
            printf("Average FNR = %.5f\n", avg_fnr);

            if (avg_fnr <= error_bound) {
                printf("Stopping search at nprobe = %ld with FNR = %.5f\n", nprobe, avg_fnr);
                optimal_nprobe = nprobe;
                break;
            }
        }

        delete[] sampled_queries;

        printf("[%.3f s] Evalauting on %ld queries according to ConANN\n",
               elapsed() - t0,
               tests);
        size_t nq_remaining = tests - nq_sampled;
        std::vector<size_t> remaining_indices(nq_remaining);
        std::iota(remaining_indices.begin(), remaining_indices.end(), nq_sampled);
        std::shuffle(remaining_indices.begin(), remaining_indices.end(), std::mt19937{std::random_device{}()});

        float* remaining_queries = new float[nq_remaining * d];
        for (size_t i = 0; i < nq_remaining; i++) {
            std::memcpy(remaining_queries + i * d, xq + remaining_indices[i] * d, d * sizeof(float));
        }

        faiss::idx_t* remaining_gt = new faiss::idx_t[nq_remaining * k];
        float* remaining_gt_D = new float[nq_remaining * k];
        for (size_t i = 0; i < nq_remaining; i++) {
            size_t query_idx = remaining_indices[i];
            std::memcpy(remaining_gt + i * k, gt + query_idx * k, k * sizeof(faiss::idx_t));
            std::memcpy(remaining_gt_D + i * k, gt_D + query_idx * k, k * sizeof(float));
        }

        // Perform knn search with optimal nprobe on the remaining queries
        index->nprobe = optimal_nprobe;
        std::vector<faiss::idx_t> I_remaining(nq_remaining * k);
        std::vector<float> D_remaining(nq_remaining * k);
        index->search(nq_remaining, remaining_queries, k, D_remaining.data(), I_remaining.data());

        // Calculate and print FNR for remaining queries
        float avg_fnr_remaining = calculate_fnr(I_remaining.data(), remaining_gt, nq_remaining, k);
        printf("Average FNR for remaining queries = %.5f\n", avg_fnr_remaining);

        delete[] remaining_queries;
        delete[] remaining_gt;
        delete[] remaining_gt_D;
    }

    delete[] xq;
    delete[] gt;
    delete[] gt_D;
    delete index;
    return 0;
}