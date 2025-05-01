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
    double sum = 0.0;
    int negativeCount = 0;
    for (const auto &num : numbers) {
        if (num > 0) {
            sum += num;
        } else {
            ++negativeCount;
        }
    }
    if (negativeCount > 0) {
        std::cout << "\nWARNING! Number of negative values: " << negativeCount
                  << std::endl;
    }
    return sum / (numbers.size() - negativeCount);
}

template <typename T>
void write_to_file(const std::vector<T> &data, const std::string &filename) {
    std::ofstream file(filename);
    for (const auto &value : data) {
        file << value << '\n';
    }
    file.close();
}

void write_time_report_csv(const std::string& filename, const faiss::IndexIVF::TimeReport time_report) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    // Write header
    // file << "Metric,Time\n";
    
    // Write data with 6 decimal precision
    file << std::fixed << std::setprecision(6)
         << "ComputeScores," << time_report.computeScores << "\n"
         << "ComputeScoresCalib," << time_report.computeScoresCalib << "\n"
         << "ComputeScoresTune," << time_report.computeScoresTune << "\n"
         << "MemoryCopyPostCompute," << time_report.memoryCopyPostCompute << "\n"
         << "PickRegLambda," << time_report.pickRegLambda << "\n"
         << "RegularizeScores," << time_report.regularizeScores << "\n"
         << "Optimize," << time_report.optimize << "\n"
         << "ConfigureTotal," << time_report.configureTotal << "\n";

    file.close();
}

/// Command like this: ./error sift1M 0.5 0.1
int main(int argc, char **argv) {
    std::cout << argc - 1 << " arguments" << std::endl;
    if (argc - 1 <= 5) {
        printf("You should at least input 5 params: the dataset name, calib "
               "size (%), tune size (%), alpha, nlist, k\n");
        return 0;
    }
    std::string param1 = argv[1]; // dataset
    std::string param2 = argv[2]; // calibration size (%)
    std::string param3 = argv[3]; // tuning size (%)
    std::string param4 = argv[4]; // alpha
    std::string param5 = argv[5]; // nlist value
    std::string param6 = argv[6]; // optional: k

    std::string dataset_name = param1;
    float calib_sz = std::stof(param2);
    float tune_sz = std::stof(param3);
    float alpha = std::stof(param4);
    int input_nlist = std::stoi(param5);
    std::string selection_k =
        param6; // needs to be part of the dataset read in process and will be
                // determined on reading in GTs

    float max_distance;

    float bert_max_dist = 20;
    float glove_max_dist = 100;
    float fasttext_max_dist = 1000;
    float gist_max_dist = 200;
    float deep_max_dist = 100;
    float sift_max_dist = 1000000;

    std::string db, query, gtI, gtD;
    if (dataset_name == "bert") {
        db = "../data/bert/db.fvecs";
        query = "../data/bert/queries.fvecs";
        gtI = "../data/bert/indices-" + selection_k + ".fvecs";
        gtD = "../data/bert/distances-" + selection_k + ".fvecs";
        max_distance = bert_max_dist;
    } else if (dataset_name == "gist30k") {
        db = "../data/gist30k/gist30k_base.fvecs";
        query = "../data/gist30k/queries.fvecs";
        gtI = "../data/gist30k/indices-" + selection_k + ".fvecs";
        gtD = "../data/gist30k/distances-" + selection_k + ".fvecs";
        max_distance = gist_max_dist;
    } else if (dataset_name == "glove30k") {
        db = "../data/glove30k/glove30k_db.fvecs";
        query = "../data/glove30k/queries.fvecs";
        gtI = "../data/glove30k/indices-" + selection_k + ".fvecs";
        gtD = "../data/glove30k/distances-" + selection_k + ".fvecs";
        max_distance = glove_max_dist;
    } else if (dataset_name == "synth") {
        db = "../data/synthetic10/db.fvecs";
        query = "../data/synthetic10/queries.fvecs";
        gtI = "../data/synthetic10/indices-" + selection_k + ".fvecs";
        gtD = "../data/synthetic10/distances-" + selection_k + ".fvecs";
        max_distance = bert_max_dist;
    } else if (dataset_name == "sift1M") {
        db = "../data/sift1M/sift_base.fvecs";
        query = "../data/sift1M/queries.fvecs";
        gtI = "../data/sift1M/indices-" + selection_k + ".fvecs";
        gtD = "../data/sift1M/distances-" + selection_k + ".fvecs";
        max_distance = sift_max_dist;
    } else if (dataset_name == "deep10M") {
        db = "../data/deep/deep10M.fvecs";
        query = "../data/deep/queries.fvecs";
        gtI = "../data/deep/indices-" + selection_k + ".fvecs";
        gtD = "../data/deep/distances-" + selection_k + ".fvecs";
        max_distance = deep_max_dist;
    } else if (param1 == "gist") {
        db = "../data/gist/gist_base.fvecs";
        query = "../data/gist/queries.fvecs";
        gtI = "../data/gist/indices-" + selection_k + ".fvecs";
        gtD = "../data/gist/distances-" + selection_k + ".fvecs";
        max_distance = gist_max_dist;
    } else if (param1 == "glove") {
        db = "../data/glove/db.fvecs";
        query = "../data/glove/queries.fvecs";
        gtI = "../data/glove/indices-" + selection_k + ".fvecs";
        gtD = "../data/glove/distances-" + selection_k + ".fvecs";
        max_distance = glove_max_dist;
    } else if (param1 == "synth") {
        db = "../data/synthetic10/db.fvecs";
        query = "../data/synthetic10/queries.fvecs";
        gtI = "../data/synthetic10/indices-" + selection_k + ".fvecs";
        gtD = "../data/synthetic10/distances-" + selection_k + ".fvecs";
        max_distance = sift_max_dist;
    } else if (dataset_name == "gauss5") {
        db = "../data/gauss5/db.fvecs";
        query = "../data/gauss5/queries.fvecs";
        gtI = "../data/gauss5/indices-" + selection_k + ".fvecs";
        gtD = "../data/gauss5/distances-" + selection_k + ".fvecs";
        max_distance = sift_max_dist;
    } else if (param1 == "gauss10") {
        db = "../data/gauss10/db.fvecs";
        query = "../data/gauss10/queries.fvecs";
        gtI = "../data/gauss10/indices-" + selection_k + ".fvecs";
        gtD = "../data/gauss10/distances-" + selection_k + ".fvecs";
        max_distance = sift_max_dist;
    } else if (param1 == "fasttext") {
        db = "../data/fasttext/db.fvecs";
        query = "../data/fasttext/queries.fvecs";
        gtI = "../data/fasttext/indices-" + selection_k + ".fvecs";
        gtD = "../data/fasttext/distances-" + selection_k + ".fvecs";
        max_distance = fasttext_max_dist;
    } else {
        printf("Your dataset name is illegal\n");
        return 1;
    }

    omp_set_num_threads(60);
    double t0 = elapsed();

    faiss::IndexIVFFlat *index;

    size_t d;

    {
        printf("[%.3f s] Loading train set\n", elapsed() - t0);

        size_t nt;
        float *xt = fvecs_read(db.c_str(), &d, &nt);

        printf("[%.3f s] Preparing index IVFFlat_%i d=%ld\n", elapsed() - t0,
               input_nlist, d);

        int nlist = input_nlist; // 1024 originally

        faiss::IndexFlatL2 *flat_index = new faiss::IndexFlatL2(d);
        index = new faiss::IndexIVFFlat(flat_index, d, nlist, faiss::METRIC_L2);
        
        // Make clustering seed explicit
        index->cp.seed = 420;

        index->nprobe = nlist;
        // train on half the dataset
        // TODO: Are we sure the data is shuffled otherwise we train K-means out
        // of distribution.
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
    auto t1 = elapsed();
    auto calib_res = index->calibrate(alpha, k, calib_sz, tune_sz, xq, nq, gt,
                                      max_distance, dataset_name);
    std::cout << "Calibration-time=" << elapsed() - t1 << "\n";
    std::cout << "Found lamhat=" << calib_res.lamhat << "\n";

    // Around half of GT was mem_copied into calib_cx and calib_labels so we can
    // free up this memory here
    delete[] xq;
    delete[] gt;
    delete[] gt_v;

    printf("[%.3f s] ConANN Evaluation\n", elapsed() - t0);
    auto [fnr, cls] = index->evaluate_test(calib_res);
    float avg_fnr = std::accumulate(fnr.begin(), fnr.end(), 0.0) / fnr.size();
    printf(
        "[%.3f s] Finished: alpha=%.3f, test fnr=%.3f, avg cls searched=%.3f\n",
        elapsed() - t0, alpha, avg_fnr, computeAverage(cls));
    auto c = computeAverage(cls);
    std::cout << "alpha=" << alpha << ", test fnr=" << avg_fnr
              << ", avg cls searched=" << c << std::endl;

    std::ostringstream fnr_filename;
    // std::string dataset_key = param1 + "-" + std::to_string(input_nlist) +
    // "-" + selection_k;
    fnr_filename << "../ConANN-error-" << dataset_name << "-"
                 << std::to_string(input_nlist) << "-" << selection_k << "-"
                 << alpha << "-" << calib_sz << "-" << tune_sz << ".log";
    write_to_file(fnr, fnr_filename.str());

    std::ostringstream cls_filename;
    cls_filename << "../ConANN-efficiency-" << dataset_name << "-"
                 << std::to_string(input_nlist) << "-" << selection_k << "-"
                 << alpha << "-" << calib_sz << "-" << tune_sz << ".log";
    write_to_file(cls, cls_filename.str());

    std::ostringstream time_filename;
    time_filename << "../ConANN-timing-" << dataset_name << "-"
                << std::to_string(input_nlist) << "-" << selection_k << "-"
                << alpha << "-" << calib_sz << "-" << tune_sz << ".csv";
    write_time_report_csv(time_filename.str(), index->time_report);

    delete index;
    return 0;
}