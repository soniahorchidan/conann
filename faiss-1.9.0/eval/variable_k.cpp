#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <numeric>
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
    std::cout << "] " << "(" << i << "/" << total << ")" << "\r";
    std::cout.flush();
}

// Updated main function
int main(int argc, char** argv) {
    if (argc < 8) {
        printf("Not enough params.");
        return 0;
    }

    std::string param1 = argv[1];
    std::string param2 = argv[2];
    std::string param3 = argv[3];
    std::string param4 = argv[4];
    std::string param5 = argv[5];
    std::string param6 = argv[6];
    std::string param7 = argv[7];

    float calib_sz = std::stof(param2);
    int input_nlist = std::stoi(param3);
    std::string lower_bound_k = param4;
    std::string upper_bound_k = param5;
    int starting_nprobe = std::stoi(param6);
    float alpha = std::stof(param7);

    // std::vector<float> alphas;
    // for (int i = 7; i < argc; i++) {
    //     alphas.push_back(std::stof(argv[i]));
    // }

    // std::sort(alphas.begin(), alphas.end(), std::greater<>()); // Sort in
    // descending order

    std::string db, query, gtI, gtD;
    if (param1 == "bert") {
        db = "../data/bert/db.fvecs";
        query = "../data/bert/queries.fvecs";
        // gtI = "../data/bert/indices-" + selection_k + ".fvecs";
        // gtD = "../data/bert/distances-" + selection_k + ".fvecs";
    } else if (param1 == "gist30k") {
        db = "../data/gist30k/gist30k_base.fvecs";
        query = "../data/gist30k/queries.fvecs";
        // gtI = "../data/gist30k/indices-" + selection_k + ".fvecs";
        // gtD = "../data/gist30k/distances-" + selection_k + ".fvecs";
    } else if (param1 == "glove30k") {
        db = "../data/glove30k/glove30k_db.fvecs";
        query = "../data/glove30k/queries.fvecs";
        // gtI = "../data/glove30k/indices-" + selection_k + ".fvecs";
        // gtD = "../data/glove30k/distances-" + selection_k + ".fvecs";
    } else if (param1 == "sift1M") {
        db = "../data/sift1M/sift_base.fvecs";
        query = "../data/sift1M/queries.fvecs";
        // gtI = "../data/sift1M/indices-" + selection_k + ".fvecs";
        // gtD = "../data/sift1M/distances-" + selection_k + ".fvecs";
    } else if (param1 == "deep10M") {
        db = "../data/deep/deep10M.fvecs";
        query = "../data/deep/queries.fvecs";
        // gtI = "../data/deep/indices-" + selection_k + ".fvecs";
        // gtD = "../data/deep/distances-" + selection_k + ".fvecs";
    } else if (param1 == "gist") {
        db = "../data/gist/gist_base.fvecs";
        query = "../data/gist/queries.fvecs";
        // gtI = "../data/gist/indices-" + selection_k + ".fvecs";
        // gtD = "../data/gist/distances-" + selection_k + ".fvecs";
    } else if (param1 == "glove") {
        db = "../data/glove/db.fvecs";
        query = "../data/glove/queries.fvecs";
        // gtI = "../data/glove/indices-" + selection_k + ".fvecs";
        // gtD = "../data/glove/distances-" + selection_k + ".fvecs";
    } else if (param1 == "synth") {
        db = "../data/synthetic10/db.fvecs";
        query = "../data/synthetic10/queries.fvecs";
        // gtI = "../data/synthetic10/indices-" + selection_k + ".fvecs";
        // gtD = "../data/synthetic10/distances-" + selection_k + ".fvecs";
    } else if (param1 == "gauss5") {
        db = "../data/gauss05/db.fvecs";
        query = "../data/gauss5/queries.fvecs";
        // gtI = "../data/gauss5/indices-" + selection_k + ".fvecs";
        // gtD = "../data/gauss5/distances-" + selection_k + ".fvecs";
    } else if (param1 == "gauss10") {
        db = "../data/gauss10/db.fvecs";
        query = "../data/gauss10/queries.fvecs";
        // gtI = "../data/gauss10/indices-" + selection_k + ".fvecs";
        // gtD = "../data/gauss10/distances-" + selection_k + ".fvecs";
    } else if (param1 == "fasttext") {
        db = "../data/fasttext/db.fvecs";
        query = "../data/fasttext/queries.fvecs";
        // gtI = "../data/fasttext/indices-" + selection_k + ".fvecs";
        // gtD = "../data/fasttext/distances-" + selection_k + ".fvecs";
    } else {
        printf("Your dataset name is illegal\n");
        return 1;
    }

    omp_set_num_threads(60);
    double t0 = elapsed();

    faiss::IndexIVFFlat* index;

    size_t d;

    int nlist = input_nlist; // 1024 originally

    printf("[%.3f s] Loading train set\n", elapsed() - t0);

    size_t nt;
    float* xt = fvecs_read(db.c_str(), &d, &nt);

    {
        printf("[%.3f s] Preparing index IndexIVF_%i d=%ld\n",
               elapsed() - t0,
               nlist,
               d);

        faiss::IndexFlatL2* flat_index = new faiss::IndexFlatL2(d);
        index = new faiss::IndexIVFFlat(flat_index, d, nlist, faiss::METRIC_L2);

        // Make clustering seed explicit
        index->cp.seed = 420;

        index->nprobe = nlist;

        // train on half the dataset
        auto ntt = size_t(0.5 * nt);
        printf("[%.3f s] Training on %ld vectors\n", elapsed() - t0, ntt);

        index->train(ntt, xt);
        index->add(nt, xt);
        // delete[] xt;
    }

    size_t nq;
    float* xq;

    {
        printf("[%.3f s] Loading queries\n", elapsed() - t0);

        size_t d2;
        xq = fvecs_read(query.c_str(), &d2, &nq);
        assert(d == d2 || !"query does not have same dimension as train set");
    }

    // VARIABLE K BLOCK

    std::string filebase = db.substr(0, db.find_last_of("/\\"));
    std::string variable_k_distribution_key =
            lower_bound_k + "-" + upper_bound_k;
    std::string variable_k_filename = filebase + "/variable-k-labels-" +
            variable_k_distribution_key + ".txt";

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
               elapsed() - t0,
               nq);

        faiss::IndexFlatL2 exact_index(d);
        exact_index.add(nt, xt);

        printf("[%.3f s] Computing gts...\n", elapsed() - t0);

        labels.resize(nq);
        for (size_t i = 0; i < nq; ++i) {
            print_progress_bar(i, nq);

            // iterate one query at a time
            const float* xi = xq + i * d;
            labels[i].resize(ks[i]);
            exact_index.assign(1, xi, labels[i].data(), ks[i]);
        }

        write_variable_k_labels_to_file(labels, variable_k_filename);
    }

    delete[] xt;

    // calibration
    auto calib_nq = size_t(calib_sz * nq);

    printf("[%.3f s] Calibrating for alpha = %.5f. Starting from nprobe = %d.\n",
           elapsed() - t0,
           alpha,
           starting_nprobe);
    int optimal_nprobe = starting_nprobe;
    for (size_t nprobe = optimal_nprobe; nprobe <= nlist; ++nprobe) {
        index->nprobe = nprobe;

        std::vector<float> fnr_per_query{};
        for (int i = 0; i < calib_nq; ++i) {
            print_progress_bar(i, calib_nq);
            // iterate one query at a time
            const float* xi = xq + i * d;

            std::vector<faiss::idx_t> nns(ks[i]);
            std::vector<float> dis(ks[i]);
            index->search(1, xi, ks[i], dis.data(), nns.data());

            auto [fnr, _] =
                    calculate_fnr(nns.data(), labels[i].data(), 1, ks[i]);
            fnr_per_query.push_back(fnr);
        }

        float average_fnr =
                std::accumulate(
                        fnr_per_query.begin(), fnr_per_query.end(), 0.0f) /
                fnr_per_query.size();
        printf("[%.3f s] Processed batch. Current nprobe = %d. Current average fnr = %.3f\n",
           elapsed() - t0, nprobe, average_fnr);
        if (average_fnr <= alpha) {
            optimal_nprobe = nprobe;
            break;
        }
    }

    printf("[%.3f s] Optimal nprobe for alpha %.5f = %d\n", elapsed() - t0, alpha, optimal_nprobe);

    // Evaluation
    index->nprobe = optimal_nprobe;
    int test_start_idx = calib_nq;

    printf("[%.3f s] Evaluating alpha = %.5f, with nprobe = %d. Beginning with query number %d.\n",
           elapsed() - t0,
           alpha,
           optimal_nprobe,
           test_start_idx);

    std::vector<float> fnr_per_query{};
    for (int i = test_start_idx; i < nq; ++i) {
        print_progress_bar(i-test_start_idx, nq-test_start_idx);
        // iterate one query at a time
        const float* xi = xq + i * d;

        std::vector<faiss::idx_t> nns(ks[i]);
        std::vector<float> dis(ks[i]);
        index->search(1, xi, ks[i], dis.data(), nns.data());

        auto [fnr, _] = calculate_fnr(nns.data(), labels[i].data(), 1, ks[i]);
        fnr_per_query.push_back(fnr);
    }

    float average_fnr =
            std::accumulate(fnr_per_query.begin(), fnr_per_query.end(), 0.0f) /
            fnr_per_query.size();
    printf("[%.3f s] Average FNR for alpha %.5f = %.5f                        \n", elapsed() - t0, alpha, average_fnr);

    // Save results
    std::ostringstream filename;
    filename << "../Faiss-error-variable-k-" << param1 << "-" << param2 << "-"
             << param3 << "-" << lower_bound_k << "-" << upper_bound_k << "-"
             << alpha << ".log";
    write_to_file(fnr_per_query, filename.str());

    std::ostringstream filename2;
    filename2 << "../Faiss-efficiency-variable-k-" << param1 << "-" << param2
              << "-" << param3 << "-" << lower_bound_k << "-" << upper_bound_k
              << "-" << alpha << ".log";
    write_to_file(std::vector<int>{optimal_nprobe}, filename2.str());

    delete[] xq;
    delete index;
    return 0;
}
