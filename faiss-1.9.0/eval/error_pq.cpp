#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <algorithm>

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <omp.h>
#include <sys/time.h>

#include "faiss/AutoTune.h"
#include "faiss/IndexFlat.h"
#include "faiss/IndexIVFFlat.h"
#include "faiss/IndexIVFPQ.h"
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


// Updated main function
int main(int argc, char **argv) {
    if (argc < 4) {
        printf("You should input at least 3 params: dataset name, calib size percentage, and one or more alpha values\n");
        return 0;
    }

    std::string param1 = argv[1];
    std::string param2 = argv[2];
    std::string param3 = argv[3];
    std::string param4 = argv[4];
    std::string param5 = argv[5];

    float calib_sz = std::stof(param2);
    int input_nlist = std::stoi(param3);
    std::string selection_k = param4;
    std::string dataset_key = param1 + "_" + param3 + "_" + selection_k;
    int starting_nprobe = std::stoi(param5);

    std::vector<float> alphas;
    for (int i = 6; i < argc; i++) {
        alphas.push_back(std::stof(argv[i]));
    }

    std::sort(alphas.begin(), alphas.end(), std::greater<>()); // Sort in descending order

    std::string db, query, gtI, gtD;
    if (param1 == "bert") {
        db = "../data/bert/db.fvecs";
        query = "../data/bert/queries.fvecs";
        gtI = "../data/bert/indices-" + selection_k + ".fvecs";
        gtD = "../data/bert/distances-" + selection_k + ".fvecs";
    } else if (param1 == "gist30k") {
        db = "../data/gist30k/gist30k_base.fvecs";
        query = "../data/gist30k/queries.fvecs";
        gtI = "../data/gist30k/indices-" + selection_k + ".fvecs";
        gtD = "../data/gist30k/distances-" + selection_k + ".fvecs";
    } else if (param1 == "glove30k") {
        db = "../data/glove30k/glove30k_db.fvecs";
        query = "../data/glove30k/queries.fvecs";
        gtI = "../data/glove30k/indices-" + selection_k + ".fvecs";
        gtD = "../data/glove30k/distances-" + selection_k + ".fvecs";
    } else if (param1 == "sift1M") {
        db = "../data/sift1M/sift_base.fvecs";
        query = "../data/sift1M/queries.fvecs";
        gtI = "../data/sift1M/indices-" + selection_k + ".fvecs";
        gtD = "../data/sift1M/distances-" + selection_k + ".fvecs";
    } else if (param1 == "deep10M") {
        db = "../data/deep/deep10M.fvecs";
        query = "../data/deep/queries.fvecs";
        gtI = "../data/deep/indices-" + selection_k + ".fvecs";
        gtD = "../data/deep/distances-" + selection_k + ".fvecs";
    } else if (param1 == "gist") {
        db = "../data/gist/gist_base.fvecs";
        query = "../data/gist/queries.fvecs";
        gtI = "../data/gist/indices-" + selection_k + ".fvecs";
        gtD = "../data/gist/distances-" + selection_k + ".fvecs";
    } else if (param1 == "glove") {
        db = "../data/glove/db.fvecs";
        query = "../data/glove/queries.fvecs";
        gtI = "../data/glove/indices-" + selection_k + ".fvecs";
        gtD = "../data/glove/distances-" + selection_k + ".fvecs";
    } else if (param1 == "synth") {
        db = "../data/synthetic10/db.fvecs";
        query = "../data/synthetic10/queries.fvecs";
        gtI = "../data/synthetic10/indices-" + selection_k + ".fvecs";
        gtD = "../data/synthetic10/distances-" + selection_k + ".fvecs";
    } else if (param1 == "gauss5") {
        db = "../data/gauss05/db.fvecs";
        query = "../data/gauss5/queries.fvecs";
        gtI = "../data/gauss5/indices-" + selection_k + ".fvecs";
        gtD = "../data/gauss5/distances-" + selection_k + ".fvecs";
    } else if (param1 == "gauss10") {
        db = "../data/gauss10/db.fvecs";
        query = "../data/gauss10/queries.fvecs";
        gtI = "../data/gauss10/indices-" + selection_k + ".fvecs";
        gtD = "../data/gauss10/distances-" + selection_k + ".fvecs";
    } else if (param1 == "fasttext") {
        db = "../data/fasttext/db.fvecs";
        query = "../data/fasttext/queries.fvecs";
        gtI = "../data/fasttext/indices-" + selection_k + ".fvecs";
        gtD = "../data/fasttext/distances-" + selection_k + ".fvecs";
    }  else {
        printf("Your dataset name is illegal\n");
        return 1;
    }

    omp_set_num_threads(60);
    double t0 = elapsed();

    // faiss::IndexIVFFlat *index;
    faiss::IndexIVFPQ *index;

    size_t d;

    int nlist = input_nlist; // 1024 originally

    {
        printf("[%.3f s] Loading train set\n", elapsed() - t0);

        size_t nt;
        float *xt = fvecs_read(db.c_str(), &d, &nt);

        printf("[%.3f s] Preparing index IndexIVF_%i d=%ld\n", elapsed() - t0,
               nlist, d);

        faiss::IndexFlatL2 *flat_index = new faiss::IndexFlatL2(d);
        // index = new faiss::IndexIVFFlat(flat_index, d, nlist, faiss::METRIC_L2);
        index = new faiss::IndexIVFPQ(flat_index, d, nlist, 8, 8, faiss::METRIC_L2);
        // Make clustering seed explicit
        index->cp.seed = 420;
        index->pq.cp.seed = 420;

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

    auto calib_nq = size_t(calib_sz* nq);

    int optimal_nprobe = starting_nprobe;
    for (float alpha : alphas) {
        printf("[%.3f s] Processing alpha = %.5f. Starting from nprobe = %d\n", elapsed() - t0, alpha, optimal_nprobe);

        if (optimal_nprobe != nlist) {
            // check maximum nprobe first
            index->nprobe = nlist;
            std::vector<faiss::idx_t> I(calib_nq * k);
            std::vector<float> D(calib_nq * k);
            index->search(calib_nq, xq, k, D.data(), I.data());
            auto [avg_fnr, _] = calculate_fnr(I.data(), gt, calib_nq, k);

            if (avg_fnr > alpha) {
                printf("[%.3f s] Underlying quantizer is not accurate enough\n", elapsed() - t0);
                optimal_nprobe = nlist;
            } else {
                // Search for optimal nprobe
                for (size_t nprobe = optimal_nprobe; nprobe <= nlist; nprobe++) {
                    index->nprobe = nprobe;
                    std::vector<faiss::idx_t> I(calib_nq * k);
                    std::vector<float> D(calib_nq * k);
                    index->search(calib_nq, xq, k, D.data(), I.data());

                    auto [avg_fnr, _] = calculate_fnr(I.data(), gt, calib_nq, k);
                    std::cout << "Probed " << nprobe << " clusters; fnr=" << avg_fnr << "\n";

                    if (avg_fnr <= alpha) {
                        optimal_nprobe = nprobe;
                        break;
                    }
                }
            }
        }

        printf("[%.3f s] Optimal nprobe for alpha %.5f = %d\n", elapsed() - t0, alpha, optimal_nprobe);

        // Evaluate on remaining queries
        size_t nq_remaining = nq - calib_nq;
        index->nprobe = optimal_nprobe;
        std::vector<faiss::idx_t> I_remaining(nq_remaining * k);
        std::vector<float> D_remaining(nq_remaining * k);

        index->search(nq_remaining, xq + calib_nq * d, k, D_remaining.data(),
                      I_remaining.data());

        auto [avg_fnr, all_fnrs] = calculate_fnr(
            I_remaining.data(), gt + calib_nq * k, nq_remaining, k);
        printf("[%.3f s] Average FNR for alpha %.5f = %.5f\n", elapsed() - t0, alpha, avg_fnr);

        // Save results
        std::ostringstream filename;
        filename << "../Faiss-pq-error-" << dataset_key << "-" << alpha << ".log";
        write_to_file(all_fnrs, filename.str());

        std::ostringstream filename2;
        filename2 << "../Faiss-pq-efficiency-" << dataset_key << "-" << alpha << ".log";
        write_to_file(std::vector<int>{optimal_nprobe}, filename2.str());
    }
    std::cout << optimal_nprobe << std::endl;

    delete[] xq;
    delete[] gt;
    delete[] gt_D;
    delete index;
    return 0;
}
