#include <algorithm>
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

#include <omp.h>
#include <sys/time.h>

#include "faiss/AutoTune.h"
#include "faiss/IndexFlat.h"
#include "faiss/index_factory.h"
#include "faiss/index_io.h"
#include "faiss/profile.h"

#include <fstream>
#include <iostream>

/*****************************************************
 * I/O functions for fvecs and ivecs
 *****************************************************/

#define DC(classname) classname* ix = dynamic_cast<classname*>(index)

float* fvecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "r");
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
    float* x = new float[n * (d + 1)];
    size_t nr = fread(x, sizeof(float), n * (d + 1), f);
    assert(nr == n * (d + 1) || !"could not read whole file");

    // shift array to remove row headers
    for (size_t i = 0; i < n; i++)
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

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

template <typename T>
void write_to_file(const std::vector<T>& data, const std::string& filename) {
    std::ofstream file(filename);
    for (const auto& value : data) {
        file << value << '\n';
    }
    file.close();
}

std::pair<float, std::vector<float>> calculate_fnr(
        const faiss::idx_t* query_indices,
        const faiss::idx_t* ground_truth,
        size_t nq_sampled,
        size_t k) {
    int total_false_negatives = 0;
    std::vector<float> fnrs_per_query(nq_sampled);

    for (size_t i = 0; i < nq_sampled; i++) {
        // Create sets for the current query and ground truth
        std::unordered_set<faiss::idx_t> query_set(
                query_indices + i * k, query_indices + (i + 1) * k);
        std::unordered_set<faiss::idx_t> gt_set(
                ground_truth + i * k, ground_truth + (i + 1) * k);

        int local_fn = 0;
        // Measure the intersection between query set and ground truth set
        for (const auto& gt_idx : gt_set) {
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
int main(int argc, char** argv) {
    if (argc < 4) {
        printf("You should input at least 3 params: dataset name, calib size percentage, and one or more alpha values\n");
        return 0;
    }

    std::string param1 = argv[1];
    std::string param2 = argv[2];
    std::vector<float> alphas;
    for (int i = 3; i < argc; i++) {
        alphas.push_back(std::stof(argv[i]));
    }

    std::sort(
            alphas.begin(),
            alphas.end(),
            std::greater<>()); // Sort in descending order
    float training_fraction = std::stof(param2);

    int figureid = -1;
    std::string db, query, gtI, gtD;
    if (param1 == "bert_10") {
        db = "../../data/bert/db.fvecs";
        query = "../../data/bert/queries.fvecs";
        gtI = "../../data/bert/indices-10.fvecs";
        gtD = "../../data/bert/distances-10.fvecs";
        figureid = 11;
    } else if (param1 == "bert_100") {
        db = "../../data/bert/db.fvecs";
        query = "../../data/bert/queries.fvecs";
        gtI = "../../data/bert/indices-100.fvecs";
        gtD = "../../data/bert/distances-100.fvecs";
        figureid = 11;
    } else if (param1 == "bert_1000") {
        db = "../../data/bert/db.fvecs";
        query = "../../data/bert/queries.fvecs";
        gtI = "../../data/bert/indices-1000.fvecs";
        gtD = "../../data/bert/distances-1000.fvecs";
        figureid = 11;
    } else if (param1 == "sift10k") {
        db = "../../data/sift10k/siftsmall_base.fvecs";
        query = "../../data/sift10k/siftsmall_query.fvecs";
        gtI = "../../data/sift10k/sift10k_gt_indices_k10.fvecs";
        gtD = "../../data/sift10k/sift10k_gt_distances_k10.fvecs";
        figureid = 9;
    } else if (param1 == "sift1M") {
        db = "../../data/sift1M/sift_base.fvecs";
        query = "../../data/sift1M/sift_query.fvecs";
        gtI = "../../data/sift1M/sift_gt_index.fvecs";
        gtD = "../../data/sift1M/sift_gt_dis.fvecs";
        figureid = 9;
    } else if (param1 == "sift10M") {
        figureid = 9;
        db = "/workspace/data/sift/sift10M/sift10M.fvecs";
        query = "/workspace/data/sift/sift10M/query.fvecs";
        gtI = "/workspace/data/sift/sift10M/idx.fvecs";
        gtD = "/workspace/data/sift/sift10M/dis.fvecs";
    } else if (param1 == "deep10M_10") {
        figureid = 10;
        db = "../../data/deep/deep10M.fvecs";
        query = "../../data/deep/queries.fvecs";
        gtI = "../../data/deep/indices-10.fvecs";
        gtD = "../../data/deep/distances-10.fvecs";
    } else if (param1 == "deep10M_100") {
        figureid = 10;
        db = "../../data/deep/deep10M.fvecs";
        query = "../../data/deep/queries.fvecs";
        gtI = "../../data/deep/indices-100.fvecs";
        gtD = "../../data/deep/distances-100.fvecs";
    } else if (param1 == "deep10M_1000") {
        figureid = 10;
        db = "../../data/deep/deep10M.fvecs";
        query = "../../data/deep/queries.fvecs";
        gtI = "../../data/deep/indices-1000.fvecs";
        gtD = "../../data/deep/distances-1000.fvecs";
    } else if (param1 == "gist_10") {
        figureid = 11;
        db = "../../data/gist/gist_base.fvecs";
        query = "../../data/gist/queries.fvecs";
        gtI = "../../data/gist/indices-10.fvecs";
        gtD = "../../data/gist/distances-10.fvecs";
    } else if (param1 == "gist_100") {
        figureid = 11;
        db = "../../data/gist/gist_base.fvecs";
        query = "../../data/gist/queries.fvecs";
        gtI = "../../data/gist/indices-100.fvecs";
        gtD = "../../data/gist/distances-100.fvecs";
    } else if (param1 == "gist_1000") {
        figureid = 11;
        db = "../../data/gist/gist_base.fvecs";
        query = "../../data/gist/queries.fvecs";
        gtI = "../../data/gist/indices-1000.fvecs";
        gtD = "../../data/gist/distances-1000.fvecs";
    } else if (param1 == "spacev") {
        db = "/workspace/data/spacev/spacev10M.fvecs";
        query = "/workspace/data/spacev/query.fvecs";
        gtI = "/workspace/data/spacev/idx.fvecs";
        gtD = "/workspace/data/spacev/dis.fvecs";
    } else if (param1 == "glove_10") {
        figureid = 9;
        db = "../../data/glove/db.fvecs";
        query = "../../data/glove/queries.fvecs";
        gtI = "../../data/glove/indices-10.fvecs";
        gtD = "../../data/glove/distances-10.fvecs";
    } else if (param1 == "glove_100") {
        figureid = 9;
        db = "../../data/glove/db.fvecs";
        query = "../../data/glove/queries.fvecs";
        gtI = "../../data/glove/indices-100.fvecs";
        gtD = "../../data/glove/distances-100.fvecs";
    } else if (param1 == "glove_1000") {
        figureid = 9;
        db = "../../data/glove/db.fvecs";
        query = "../../data/glove/queries.fvecs";
        gtI = "../../data/glove/indices-1000.fvecs";
        gtD = "../../data/glove/distances-1000.fvecs";
    } else if (param1 == "text") {
        figureid = 12;
        db = "/workspace/data/text/text10M.fvecs";
        query = "/workspace/data/text/query.fvecs";
        gtI = "/workspace/data/text/idx.fvecs";
        gtD = "/workspace/data/text/dis.fvecs";
    } else {
        printf("Your dataset name is illegal\n");
        return 0;
    }

    omp_set_num_threads(32);
    double t0 = elapsed();

    const char* index_key;
    if (param1.find("bert") != std::string::npos) {
        index_key = "IVF128,Flat";
    } else {
        index_key = "IVF1024,Flat";
    }

    faiss::Index* index;
    size_t d;

    {
        printf("[%.3f s] Loading database\n", elapsed() - t0);

        size_t nb;
        float* xb = fvecs_read(db.c_str(), &d, &nb);

        printf("[%.3f s] Preparing index \"%s\" d=%ld\n",
               elapsed() - t0,
               index_key,
               d);

        index = faiss::index_factory(d, index_key);

        printf("Output index type: %d\n", index->type);

        // train on half the dataset
        auto nt = size_t(0.5 * nb);
        printf("[%.3f s] Training on %ld vectors\n", elapsed() - t0, nt);

        index->set_tune_mode();
        index->train(nt, xb);
        index->set_tune_off();

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

    size_t k;         // nb of results per query in the GT
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

    size_t kk;
    float* gt_v;

    {
        printf("[%.3f s] Loading groud truth vector\n", elapsed() - t0);
        size_t nq3;
        gt_v = fvecs_read(gtD.c_str(), &kk, &nq3);
        assert(kk == k ||
               !"gt distances does not have same dimension as gt IDs");
        assert(nq3 == nq || !"incorrect nb of ground truth entries");
    }

    // float *gt_D;

    // {
    //     printf("[%.3f s] Loading ground truth distance for %ld queries\n",
    //            elapsed() - t0, nq);

    //     // load ground-truth and convert int to long
    //     size_t nq2;
    //     gt_D = fvecs_read(gtD.c_str(), &k, &nq2);
    //     assert(nq2 == nq || !"incorrect nb of ground truth entries");
    // }

    size_t topk = k;
    size_t max_topk = k;

    // Round down to the nearest number divisible by ten (necessary for auncel,
    // don't know why)
    nq = (nq / 10) * 10;
    // get training and testing sizes
    int ts = nq * training_fraction;
    int ses = nq - ts;

    // Run error profile system
    {
        printf("[%.3f s] Preparing error profile system criterion 100-recall at 100 "
               "criterion, with k=%ld nq=%ld\n",
               elapsed() - t0,
               k,
               nq);
        faiss::Error_sys err_sys(index, nq, k);

        err_sys.set_gt(gt_v, gt);
        printf("[%.3f s] Start error profile system training\n",
               elapsed() - t0);
        err_sys.sys_train(ts, xq);
        printf("[%.3f s] Finish error profile system training\n",
               elapsed() - t0);

        // Set query topk val
        err_sys.set_topk(topk);

        for (float alpha : alphas) {
            // Initialize required accuracy and reset stopping conditions
            std::vector<float> acc(ses + ts, 1 - alpha);

            err_sys.set_queries(ses, xq, acc.data(), ts + ses);

            printf("[%.3f s] Start error profile system search for alpha: %.3f\n",
                   elapsed() - t0,
                   alpha);
            t0 = elapsed();
            // Set the infamous figure id for hyperparameter loading during
            // search
            if (DC(faiss::IndexIVF)) {
                assert(figureid >= 1 && figureid <= 12);
                ix->t->setparam(figureid);
                ix->t->profile = true;
            }

            int test_start_idx = ts;
            std::vector<double> latencies;

            // Run queries individually
            for (int i = test_start_idx; i < nq; ++i) {
                // iterate one query at a time
                // const float* xi = xq + i * index->d;
                std::vector<faiss::idx_t> nns(k);
                std::vector<float> dis(k);
                double t1 = elapsed();
                err_sys.search_latency(dis.data(), nns.data(), i, 1);
                latencies.push_back((elapsed() - t1) * 1000);
            }

            printf("Finish error profile system search: %.3f\n",
                elapsed() - t0);

            std::ostringstream filename;
            filename << "../../Auncel-latency-" << param1 << "-" << k << "-"
                     << alpha << "-" << std::time(nullptr) << ".log";
            write_to_file(latencies, filename.str());
        }
    }

    delete[] xq;
    delete[] gt;
    delete index;
    return 0;
}
