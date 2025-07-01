#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <numeric>
#include <set>
#include <vector>
#include <random>

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

template <typename T>
void write_to_file(const std::vector<T> &data, const std::string &filename) {
    std::ofstream file(filename);
    for (const auto &value : data) {
        file << value << '\n';
    }
    file.close();
}

std::vector<float> compute_fnr_per_query(
    const std::vector<std::vector<faiss::idx_t>> &prediction_set,
    const std::vector<std::vector<faiss::idx_t>> &gt_labels) {

    int nq = prediction_set.size();  // number of queries
    int k = gt_labels[0].size();  // number of ground truth neighbors
    int total_false_negatives = 0;
    std::vector<float> fnrs_per_query(nq);

    for (size_t i = 0; i < nq; ++i) {
        const std::set<int> pred_set(prediction_set[i].begin(),
                                     prediction_set[i].end());
        const std::set<int> gt_set(gt_labels[i].begin(), gt_labels[i].end());

        // Calculate the intersection size (True Positives)
        int intersection_size = 0;
        for (int pred : pred_set) {
            if (gt_set.count(pred) > 0) {
                ++intersection_size;
            }
        }

        // FNR = 1 - (intersection_size / k)
        fnrs_per_query[i] = 1.0f - (static_cast<float>(intersection_size) /
                                    static_cast<float>(k));

        // Update total false negatives
        total_false_negatives += (k - intersection_size); // FN = k - intersection_size
    }
    return fnrs_per_query;
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

void write_variable_k_labels_to_file(
        const std::vector<std::vector<int64_t>>& labels,
        const std::string& filename) {
    std::ofstream file(filename);
    for (const auto& query_labels : labels) {
        for (size_t i = 0; i < query_labels.size(); ++i) {
            file << query_labels[i];
            if (i + 1 < query_labels.size())
                file << " ";
        }
        file << "\n";
    }
    file.close();
}

std::vector<std::vector<int64_t>> read_variable_k_labels_from_file(
        const std::string& filename) {
    std::vector<std::vector<int64_t>> labels;
    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::vector<int64_t> query_labels;
        int64_t idx;
        while (iss >> idx) {
            query_labels.push_back(idx);
        }
        labels.push_back(query_labels);
    }
    return labels;
}

void print_progress_bar(size_t i, size_t total) {
    // Print progress bar
    int barWidth = 50;
    float progress = float(i) / total;
    int pos = barWidth * progress;
    std::cout << "[";
    for (int j = 0; j < barWidth; ++j) {
        if (j < pos)
            std::cout << "=";
        else if (j == pos)
            std::cout << ">";
        else
            std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << " %\r";
    std::cout.flush();
}

/// Command like this: ./error sift1M 0.5 0.1 0.1 1024 1 100
int main(int argc, char **argv) {
    std::cout << argc - 1 << " arguments" << std::endl;
    if (argc - 1 <= 6) {
        printf("You should at least input 5 params: the dataset name, calib "
               "size (%), tune size (%), alpha, nlist, lower_bound_k, upper_bound_k\n");
        return 0;
    }

    std::string param1 = argv[1]; // dataset
    std::string param2 = argv[2]; // calibration size (%)
    std::string param3 = argv[3]; // tuning size (%)
    std::string param4 = argv[4]; // alpha
    std::string param5 = argv[5]; // nlist value
    std::string param6 = argv[6]; // k lower bound
    std::string param7 = argv[7]; // k upper bound

    std::string dataset_name = param1;
    float calib_sz = std::stof(param2);
    float tune_sz = std::stof(param3);
    float alpha = std::stof(param4);
    int input_nlist = std::stoi(param5);
    std::string lower_bound_k = param6;
    std::string upper_bound_k = param7;

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
        // gtI = "../data/bert/indices-" + selection_k + ".fvecs";
        // gtD = "../data/bert/distances-" + selection_k + ".fvecs";
        max_distance = bert_max_dist;
    } else if (dataset_name == "gist30k") {
        db = "../data/gist30k/gist30k_base.fvecs";
        query = "../data/gist30k/queries.fvecs";
        // gtI = "../data/gist30k/indices-" + selection_k + ".fvecs";
        // gtD = "../data/gist30k/distances-" + selection_k + ".fvecs";
        max_distance = gist_max_dist;
    } else if (dataset_name == "glove30k") {
        db = "../data/glove30k/glove30k_db.fvecs";
        query = "../data/glove30k/queries.fvecs";
        // gtI = "../data/glove30k/indices-" + selection_k + ".fvecs";
        // gtD = "../data/glove30k/distances-" + selection_k + ".fvecs";
        max_distance = glove_max_dist;
    } else if (dataset_name == "synth") {
        db = "../data/synthetic10/db.fvecs";
        query = "../data/synthetic10/queries.fvecs";
        // gtI = "../data/synthetic10/indices-" + selection_k + ".fvecs";
        // gtD = "../data/synthetic10/distances-" + selection_k + ".fvecs";
        max_distance = bert_max_dist;
    } else if (dataset_name == "sift1M") {
        db = "../data/sift1M/sift_base.fvecs";
        query = "../data/sift1M/queries.fvecs";
        // gtI = "../data/sift1M/indices-" + selection_k + ".fvecs";
        // gtD = "../data/sift1M/distances-" + selection_k + ".fvecs";
        max_distance = sift_max_dist;
    } else if (dataset_name == "deep10M") {
        db = "../data/deep/deep10M.fvecs";
        query = "../data/deep/queries.fvecs";
        // gtI = "../data/deep/indices-" + selection_k + ".fvecs";
        // gtD = "../data/deep/distances-" + selection_k + ".fvecs";
        max_distance = deep_max_dist;
    } else if (param1 == "gist") {
        db = "../data/gist/gist_base.fvecs";
        query = "../data/gist/queries.fvecs";
        // gtI = "../data/gist/indices-" + selection_k + ".fvecs";
        // gtD = "../data/gist/distances-" + selection_k + ".fvecs";
        max_distance = gist_max_dist;
    } else if (param1 == "glove") {
        db = "../data/glove/db.fvecs";
        query = "../data/glove/queries.fvecs";
        // gtI = "../data/glove/indices-" + selection_k + ".fvecs";
        // gtD = "../data/glove/distances-" + selection_k + ".fvecs";
        max_distance = glove_max_dist;
    } else if (param1 == "synth") {
        db = "../data/synthetic10/db.fvecs";
        query = "../data/synthetic10/queries.fvecs";
        // gtI = "../data/synthetic10/indices-" + selection_k + ".fvecs";
        // gtD = "../data/synthetic10/distances-" + selection_k + ".fvecs";
        max_distance = sift_max_dist;
    } else if (dataset_name == "gauss5") {
        db = "../data/gauss5/db.fvecs";
        query = "../data/gauss5/queries.fvecs";
        // gtI = "../data/gauss5/indices-" + selection_k + ".fvecs";
        // gtD = "../data/gauss5/distances-" + selection_k + ".fvecs";
        max_distance = sift_max_dist;
    } else if (param1 == "gauss10") {
        db = "../data/gauss10/db.fvecs";
        query = "../data/gauss10/queries.fvecs";
        // gtI = "../data/gauss10/indices-" + selection_k + ".fvecs";
        // gtD = "../data/gauss10/distances-" + selection_k + ".fvecs";
        max_distance = sift_max_dist;
    } else if (param1 == "fasttext") {
        db = "../data/fasttext/db.fvecs";
        query = "../data/fasttext/queries.fvecs";
        // gtI = "../data/fasttext/indices-" + selection_k + ".fvecs";
        // gtD = "../data/fasttext/distances-" + selection_k + ".fvecs";
        max_distance = fasttext_max_dist;
    } else {
        printf("Your dataset name is illegal\n");
        return 1;
    }

    omp_set_num_threads(60);
    double t0 = elapsed();

    faiss::IndexIVFFlat *index;

    size_t d;
    
    // read db vectors
    printf("[%.3f s] Loading train set\n", elapsed() - t0);
    size_t nt;
    float *xt = fvecs_read(db.c_str(), &d, &nt);
    {
        printf("[%.3f s] Preparing index IVFFlat_%i d=%ld\n", elapsed() - t0,
               input_nlist, d);

        int nlist = input_nlist; // 1024 originally

        faiss::IndexFlatL2 *flat_index = new faiss::IndexFlatL2(d);
        index = new faiss::IndexIVFFlat(flat_index, d, nlist, faiss::METRIC_L2);
        // Make clustering seed explicit
        index->cp.seed = 420;
        // index->pq.cp.seed = 420; // Use this when testing with quantization

        index->nprobe = nlist;
        // train on half the dataset
        // TODO: Are we sure the data is shuffled otherwise we train K-means out
        // of distribution.
        auto ntt = size_t(0.5 * nt);
        printf("[%.3f s] Training on %ld vectors\n", elapsed() - t0, ntt);

        index->train(ntt, xt);
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
        // assert(d == d2 || !"query does not have same dimension as train set");
    }

    // VARIABLE K BLOCK
    
    std::string filebase = db.substr(0, db.find_last_of("/\\"));
    std::string variable_k_distribution_key = lower_bound_k + "-" + upper_bound_k;
    std::string variable_k_filename = filebase + "/variable-k-labels-" + variable_k_distribution_key + ".txt";

    std::vector<std::vector<faiss::idx_t>> labels;
    // returns empty if file does not exist
    labels = read_variable_k_labels_from_file(variable_k_filename);

    // create list of ks
    std::vector<int> ks(nq);
    std::mt19937 rng(42); // fixed seed for reproducibility on same machine
    std::uniform_int_distribution<int> dist(
            std::stoi(lower_bound_k), std::stoi(upper_bound_k));
    for (size_t i = 0; i < nq; ++i) {
        ks[i] = dist(rng);
    }
    std::cout << "First 5 ks values: ";
    for (size_t i = 0; i < std::min<size_t>(5, ks.size()); ++i) {
        std::cout << ks[i] << " ";
    }
    std::cout << std::endl;

    if (labels.empty()) {
        printf("[%.3f s] Computing ground truth for %ld queries\n",
               elapsed() - t0, nq);
 
        faiss::IndexFlatL2 exact_index(d);
        exact_index.add(nt, xt);

        printf("[%.3f s] Computing gts...\n", elapsed() - t0);
    
        labels.resize(nq);
        for (size_t i = 0; i < nq; ++i) {
            // print_progress_bar(i, nq);

            // iterate one query at a time
            const float* xi = xq + i * d;
            labels[i].resize(ks[i]);
            exact_index.assign(1, xi, labels[i].data(), ks[i]);
        }
        
        write_variable_k_labels_to_file(labels, variable_k_filename);
    }

    delete[] xt;

    printf("[%.3f s] ConANN Calibration\n", elapsed() - t0);
    auto t1 = elapsed();
    auto calib_res = index->calibrate(alpha, ks, calib_sz, tune_sz, xq, nq, labels,
                                      max_distance, dataset_name);
    std::cout << "Calibration-time=" << elapsed() - t1 << "\n";
    std::cout << "Found lamhat=" << calib_res.lamhat << "\n";

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
    fnr_filename << "../ConANN-error-" << dataset_name << "-"
                 << std::to_string(input_nlist) << "-variable-k-" << variable_k_distribution_key << "-"
                 << alpha << "-" << calib_sz << "-" << tune_sz << ".log";
    write_to_file(fnr, fnr_filename.str());

    std::ostringstream cls_filename;
    cls_filename << "../ConANN-efficiency-" << dataset_name << "-"
                 << std::to_string(input_nlist) << "-variable-k-" << variable_k_distribution_key << "-"
                 << alpha << "-" << calib_sz << "-" << tune_sz << ".log";
    write_to_file(cls, cls_filename.str());

    std::ostringstream time_filename;
    time_filename << "../ConANN-timing-" << dataset_name << "-"
                << std::to_string(input_nlist) << "-variable-k-" << variable_k_distribution_key << "-"
                << alpha << "-" << calib_sz << "-" << tune_sz << ".csv";
    write_time_report_csv(time_filename.str(), index->time_report);

    delete[] xq;
    delete index;
    return 0;
}