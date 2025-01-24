#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <numeric>

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <ctime>

#include <omp.h>
#include <sys/time.h>

#include "faiss/AutoTune.h"
#include "faiss/IndexFlat.h"
#include "faiss/IndexIVFFlat.h"
#include "faiss/index_io.h"

#include <fstream>
#include <iostream>

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

template <typename T>
void write_to_file(const std::vector<T> &data, const std::string &filename) {
    std::ofstream file(filename);
    for (const auto &value : data) {
        file << value << '\n';
    }
    file.close();
}

std::pair<float, std::vector<float>>
calculate_fnr(const faiss::idx_t *query_indices,
              const faiss::idx_t *ground_truth, size_t nq_sampled, size_t k) {
    int total_false_negatives = 0;
    std::vector<float> fnrs_per_query(nq_sampled);

    for (size_t i = 0; i < nq_sampled; i++) {
        // Create sets for the current query and ground truth
        std::unordered_set<faiss::idx_t> query_set(query_indices + i * k,
                                                   query_indices + (i + 1) * k);
        std::unordered_set<faiss::idx_t> gt_set(ground_truth + i * k,
                                                ground_truth + (i + 1) * k);

        int local_fn = 0;
        // Measure the intersection between query set and ground truth set
        for (const auto &gt_idx : gt_set) {
            if (query_set.find(gt_idx) == query_set.end()) {
                local_fn++;
            }
        }

        fnrs_per_query[i] = static_cast<float>(local_fn) / k;
        total_false_negatives += local_fn;
    }

    float overall_fnr =
        static_cast<float>(total_false_negatives) / (nq_sampled * k);
    return {overall_fnr, fnrs_per_query};
}

/// Command like this: ./error sift1M 0.5 0.1
int main(int argc, char **argv) {
    std::cout << argc << " arguments" << std::endl;
    if (argc - 1 != 3) {
        printf("You should at least input 4 params: the dataset name, calib "
               "size percentage, alpha\n");
        return 0;
    }
    std::string param1 = argv[1];
    std::string param2 = argv[2];
    std::string param3 = argv[3];
    float calib_sz = std::stof(param2);
    float alpha = std::stof(param3);

    std::string db, query, gtI, gtD;
    if (param1 == "bert_10") {
        db = "../data/bert/db.fvecs";
        query = "../data/bert/queries.fvecs";
        gtI = "../data/bert/indices-10.fvecs";
        gtD = "../data/bert/distances-10.fvecs";
    } else if (param1 == "bert_100") {
        db = "../data/bert/db.fvecs";
        query = "../data/bert/queries.fvecs";
        gtI = "../data/bert/indices-100.fvecs";
        gtD = "../data/bert/distances-100.fvecs";
    } else if (param1 == "bert_1000") {
        db = "../data/bert/db.fvecs";
        query = "../data/bert/queries.fvecs";
        gtI = "../data/bert/indices-1000.fvecs";
        gtD = "../data/bert/distances-1000.fvecs";
    } else if (param1 == "sift1M") {
        db = "../data/sift1M/sift1M.fvecs";
        query = "../data/sift1M/1M_query.fvecs";
        gtI = "../data/sift1M/idx_1M.ivecs";
        gtD = "../data/sift1M/dis_1M.fvecs";
    } else if (param1 == "sift10M") {
        db = "/workspace/data/sift/sift10M/sift10M.fvecs";
        query = "/workspace/data/sift/sift10M/query.fvecs";
        gtI = "/workspace/data/sift/sift10M/idx.ivecs";
        gtD = "/workspace/data/sift/sift10M/dis.fvecs";
    } else if (param1 == "deep10M") {
        db = "../data/deep/deep10M.fvecs";
        query = "../data/deep/query.fvecs";
        gtI = "../data/deep/idx.ivecs";
        gtD = "../data/deep/dis.fvecs";
    } else if (param1 == "gist") {
        db = "../data/gist/gist_base.fvecs";
        query = "../data/gist/gist_query.fvecs";
        gtI = "../data/gist/idx.ivecs";
        gtD = "../data/gist/dis.fvecs";
    } else if (param1 == "glove_100") {
        db = "../data/glove/db.fvecs";
        query = "../data/glove/queries.fvecs";
        gtI = "../data/glove/indices-100.fvecs";
        gtD = "../data/glove/distances-100.fvecs";
    } else {
        printf("Your dataset name is illegal\n");
        return 0;
    }

    omp_set_num_threads(32);
    double t0 = elapsed();

    // this is typically the fastest one.
    const char *index_key = "IVF1024,Flat";

    faiss::IndexIVFFlat *index;

    size_t d;

    int nlist = 1024; // 1024 as per index_key
    if (param1.find("bert") != std::string::npos) {
        nlist = 128;
    }

    {
        printf("[%.3f s] Loading train set\n", elapsed() - t0);

        size_t nt;
        float *xt = fvecs_read(db.c_str(), &d, &nt);

        printf("[%.3f s] Preparing index \"%s\" d=%ld\n", elapsed() - t0,
               index_key, d);

        faiss::IndexFlatL2 *flat_index = new faiss::IndexFlatL2(d);
        index = new faiss::IndexIVFFlat(flat_index, d, nlist, faiss::METRIC_L2);

        index->nprobe = nlist;

        // train on half the dataset
        auto ntt = size_t(0.5 * nt);
        printf("[%.3f s] Training on %ld vectors\n", elapsed() - t0, ntt);

        index->train(ntt, xt);
        delete[] xt;
    }

    {
        printf("[%.3f s] Loading database\n", elapsed() - t0);

        size_t nb, d2;
        float *xb = fvecs_read(db.c_str(), &d2, &nb);
        assert(d == d2 || !"dataset does not have same dimension as train set");

        printf("[%.3f s] Indexing database, size %ld*%ld\n", elapsed() - t0, nb,
               d);

        index->add(nb, xb);

        delete[] xb;
    }

    size_t nq;
    float *xq;

    {
        printf("[%.3f s] Loading queries\n", elapsed() - t0);

        size_t d2;
        xq = fvecs_read(query.c_str(), &d2, &nq);
        assert(d == d2 || !"query does not have same dimension as train set");
    }

    size_t k;         // nb of results per query in the GT
    faiss::idx_t *gt; // nq * k matrix of ground-truth nearest-neighbors

    {
        printf("[%.3f s] Loading ground truth for %ld queries\n",
               elapsed() - t0, nq);

        // load ground-truth and convert int to long
        size_t nq2;
        int *gt_int = ivecs_read(gtI.c_str(), &k, &nq2);
        assert(nq2 == nq || !"incorrect nb of ground truth entries");

        gt = new faiss::idx_t[k * nq];
        for (int i = 0; i < k * nq; i++) {
            gt[i] = gt_int[i];
        }
        delete[] gt_int;
    }

    size_t kk;
    float *gt_v;

    {
        printf("[%.3f s] Loading groud truth vector\n", elapsed() - t0);
        size_t nq3;
        gt_v = fvecs_read(gtD.c_str(), &kk, &nq3);
        assert(kk == k ||
               !"gt distances does not have same dimension as gt IDs");
        assert(nq3 == nq || !"incorrect nb of ground truth entries");
    }

    float *gt_D;

    {
        printf("[%.3f s] Loading ground truth distance for %ld queries\n",
               elapsed() - t0, nq);

        // load ground-truth and convert int to long
        size_t nq2;
        gt_D = fvecs_read(gtD.c_str(), &k, &nq2);
        assert(nq2 == nq || !"incorrect nb of ground truth entries");
    }

    auto calib_nq = size_t((1 - calib_sz) * nq);
    int optimal_nprobe = 0;

    {

        printf("[%.3f s] Perform parameter search on %ld queries\n",
               elapsed() - t0, calib_nq);

        // Iterate over nprobe values
        for (size_t nprobe = 1; nprobe <= nlist; nprobe++) {
            index->nprobe = nprobe;
            printf("[%.3f s] Testing nprobe = %ld\n", elapsed() - t0, nprobe);

            // Perform knn search
            std::vector<faiss::idx_t> I(nq * k);
            std::vector<float> D(nq * k);
            index->search(calib_nq, xq, k, D.data(), I.data());

            // Calculate average FNR
            auto [avg_fnr, all_fnrs] = calculate_fnr(I.data(), gt, calib_nq, k);
            printf("[%.3f s] Average FNR = %.5f\n", elapsed() - t0, avg_fnr);

            if (avg_fnr <= alpha) {
                printf("[%.3f s] Stopping search at nprobe = %ld with FNR = "
                       "%.5f\n",
                       elapsed() - t0, nprobe, avg_fnr);
                optimal_nprobe = nprobe;
                break;
            }
        }
    }

    {
        size_t nq_remaining = nq - calib_nq;

        printf("[%.3f s] Evaluating on %ld queries\n", elapsed() - t0, nq_remaining);

        // Perform knn search with optimal nprobe on the remaining queries
        index->nprobe = optimal_nprobe;
        std::vector<faiss::idx_t> I_remaining(nq_remaining * k);
        std::vector<float> D_remaining(nq_remaining * k);

        index->search(nq_remaining, xq + calib_nq * d, k, D_remaining.data(),
                      I_remaining.data());

        // Calculate and print FNR for remaining queries
        auto [avg_fnr, all_fnrs] = calculate_fnr(
            I_remaining.data(), gt + calib_nq * k, nq_remaining, k);
        printf("Average FNR for remaining queries = %.5f\n", avg_fnr);

        std::ostringstream filename;
        filename << "../Faiss-error-" << param1 << "-" << k << "-" << alpha
                  << "-" << std::time(nullptr) << ".log";           
        write_to_file(all_fnrs, filename.str());


        std::ostringstream filename2;
        filename2 << "../Faiss-cls-" << param1 << "-" << k << "-" << alpha
                 << "-" << std::time(nullptr) << ".log";
        write_to_file(std::vector<int>{optimal_nprobe}, filename2.str());
    }

    delete[] xq;
    delete[] gt;
    delete[] gt_D;
    delete index;
    return 0;
}