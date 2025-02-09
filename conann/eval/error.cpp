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

template <typename T> double computeAverage(const std::vector<T> &numbers) {
    if (numbers.empty())
        return 0.0;
    double sum = std::accumulate(numbers.begin(), numbers.end(), 0.0);
    return sum / numbers.size();
}

template <typename T>
void write_to_file(const std::vector<T> &data, const std::string &filename) {
    std::ofstream file(filename);
    for (const auto &value : data) {
        file << value << '\n';
    }
    file.close();
}

/// Command like this: ./error sift1M 0.5 0.1
int main(int argc, char **argv) {
    std::cout << argc << " arguments" << std::endl;
    if (argc - 1 != 3) {
        printf("You should at least input 3 params: the dataset name, calib "
               "size percentage, alpha\n");
        return 0;
    }
    std::string param1 = argv[1];
    std::string param2 = argv[2];
    std::string param3 = argv[3];
    float calib_sz = std::stof(param2);
    float alpha = std::stof(param3);

    float max_distance;

    float bert_max_dist = 200;
    float glove_max_dist = 100;
    float gist_max_dist = 200;
    float deep_max_dist = 100;
    float sift_max_dist = 100000;

    std::string db, query, gtI, gtD;
    if (param1 == "bert_10") {
        db = "../data/bert/db.fvecs";
        query = "../data/bert/queries.fvecs";
        gtI = "../data/bert/indices-10.fvecs";
        gtD = "../data/bert/distances-10.fvecs";
        max_distance = bert_max_dist;
    } else if (param1 == "bert_100") {
        db = "../data/bert/db.fvecs";
        query = "../data/small-bert/queries.fvecs";
        gtI = "../data/small-bert/indices-100.fvecs";
        gtD = "../data/small-bert/distances-100.fvecs";
        max_distance = bert_max_dist;
    } else if (param1 == "bert_1000") {
        db = "../data/bert/db.fvecs";
        query = "../data/bert/queries.fvecs";
        gtI = "../data/bert/indices-1000.fvecs";
        gtD = "../data/bert/distances-1000.fvecs";
        max_distance = bert_max_dist;
    } else if (param1 == "sift1M") {
        db = "../data/sift1M/sift1M.fvecs";
        query = "../data/sift1M/1M_query.fvecs";
        gtI = "../data/sift1M/idx_1M.ivecs";
        gtD = "../data/sift1M/dis_1M.fvecs";
        max_distance = sift_max_dist;
    } else if (param1 == "sift10M") {
        db = "/workspace/data/sift/sift10M/sift10M.fvecs";
        query = "/workspace/data/sift/sift10M/query.fvecs";
        gtI = "/workspace/data/sift/sift10M/idx.ivecs";
        gtD = "/workspace/data/sift/sift10M/dis.fvecs";
        max_distance = sift_max_dist;
    } else if (param1 == "deep10M") {
        db = "../data/deep/deep10M.fvecs";
        query = "../data/deep/query.fvecs";
        gtI = "../data/deep/idx.ivecs";
        gtD = "../data/deep/dis.fvecs";
        max_distance = deep_max_dist;
    } else if (param1 == "gist_10") {
        db = "../data/gist/gist_base.fvecs";
        query = "../data/gist/queries.fvecs";
        gtI = "../data/gist/indices-10.fvecs";
        gtD = "../data/gist/distances-10.fvecs";
        max_distance = gist_max_dist;
    } else if (param1 == "gist_100") {
        db = "../data/gist/gist_base.fvecs";
        query = "../data/gist/queries.fvecs";
        gtI = "../data/gist/indices-100.fvecs";
        gtD = "../data/gist/distances-100.fvecs";
        max_distance = gist_max_dist;
    } else if (param1 == "gist_1000") {
        db = "../data/gist/gist_base.fvecs";
        query = "../data/gist/queries.fvecs";
        gtI = "../data/gist/indices-1000.fvecs";
        gtD = "../data/gist/distances-1000.fvecs";
        max_distance = gist_max_dist;
    } else if (param1 == "glove_100") {
        db = "../data/glove/db.fvecs";
        query = "../data/glove/queries.fvecs";
        gtI = "../data/glove/indices-100.fvecs";
        gtD = "../data/glove/distances-100.fvecs";
        max_distance = glove_max_dist;
    } else if (param1 == "glove_1000") {
        db = "../data/glove/db.fvecs";
        query = "../data/glove/queries.fvecs";
        gtI = "../data/glove/indices-1000.fvecs";
        gtD = "../data/glove/distances-1000.fvecs";
        max_distance = glove_max_dist;
    } else {
        printf("Your dataset name is illegal\n");
        return 0;
    }

    omp_set_num_threads(8);
    double t0 = elapsed();

    // this is typically the fastest one.
    const char *index_key = "IVF1024,Flat";

    faiss::IndexIVFFlat *index;

    size_t d;

    {
        printf("[%.3f s] Loading train set\n", elapsed() - t0);

        size_t nt;
        float *xt = fvecs_read(db.c_str(), &d, &nt);

        printf("[%.3f s] Preparing index \"%s\" d=%ld\n", elapsed() - t0,
               index_key, d);

        int nlist = 1024; // 1024 as per index_key
        if (param1.find("bert") != std::string::npos) {
            nlist = 128;
        }

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

    printf("[%.3f s] ConANN Calibration\n", elapsed() - t0);
    auto lamhat = index->calibrate(alpha, k, calib_sz, xq, nq, gt, max_distance);
    printf("[%.3f s] ConANN Evaluation\n", elapsed() - t0);
    auto [fnr, cls] = index->evaluate_test(lamhat);
    std::cout << "alpha=" << alpha << ", test fnr=" << computeAverage(fnr)
              << ", avg cls searched=" << computeAverage(cls) << std::endl;

    // std::ostringstream fnr_filename;
    // fnr_filename << "../ConANN-error-" << param1 << "-" << k << "-" << alpha
    //              << "-" << num_bins << "-" << std::time(nullptr) <<".log";

    // write_to_file(fnr, fnr_filename.str());

    // std::ostringstream cls_filename;

    // cls_filename << "../ConANN-efficiency-" << param1 << "-" << k << "-"
    //              << alpha << "-" << num_bins << "-" << std::time(nullptr) << ".log";

    // write_to_file(cls, cls_filename.str());

    delete[] xq;
    delete[] gt;
    delete[] gt_v;
    delete index;
    return 0;
}