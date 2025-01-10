#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>


using namespace std::chrono;

double computeAverage(const std::vector<int>& numbers) {
    if (numbers.empty()) return 0.0;
    double sum = std::accumulate(numbers.begin(), numbers.end(), 0);
    return sum / numbers.size();
}

int main(void) {
    // dimension of the vectors to index
    int d = 3;
    int K = 100;
    int nlist = 10;

    // size of the database we plan to index
    size_t nb = 10000;
    // size of the training dataset
    size_t nt = 4000;

    std::mt19937 rng(12345);

    // make the IVF index object and train it
    faiss::IndexFlatL2 quantizer(d); // The quantizer (flat index)
    faiss::IndexIVFFlat index(&quantizer, d, nlist, faiss::METRIC_L2); // IVF index
    index.nprobe = 100; // number of probes

    // train the index on some data
    std::vector<float> training_data(nt * d); // Random training data
    std::uniform_real_distribution<float> dist(
        0.0f, 1.0f); // Distribution between 0 and 1
    for (size_t i = 0; i < nt * d; i++) {
        training_data[i] = dist(rng); // Generate float between 0 and 1
    }
    index.train(nt, training_data.data());

    // generate random database
    std::vector<float> database(nb * d);
    for (size_t i = 0; i < nb * d; i++) {
        database[i] = dist(rng);
    }

    { // populate the database
        index.add(nb, database.data());
    }

    size_t nq = 1;

    { // searching the database
        printf("Searching ...\n");

        std::vector<float> queries(nq * d);
        for (size_t i = 0; i < nq * d; i++) {
            queries[i] = dist(rng);
        }

        int k = 5;
        std::vector<faiss::idx_t> nns(k * nq);
        std::vector<float> dis(k * nq);

        auto start = high_resolution_clock::now();
        index.search(nq, queries.data(), k, dis.data(), nns.data());

        auto end = high_resolution_clock::now();

        // Output results
        auto t = duration_cast<microseconds>(end - start).count();
        int qps = nq * 1000 * 1000 / t;

        printf("QPS: %d\n", qps);
    }

    // Test calibration
    auto alpa = 0.1;
    auto lamhat = index.calibrate(alpa);
    auto [fnr, cls] = index.evaluate_test(lamhat);
    std::cout << "alpha=" << alpa << ": lamhat= " << lamhat
              << ", test fnr=" << fnr << ", avg cls searched=" << computeAverage(cls) << std::endl;

    alpa = 0.2;
    lamhat = index.calibrate(alpa);
    auto res = index.evaluate_test(lamhat);
    fnr = res.first;
    cls = res.second;
    std::cout << "alpha=" << alpa << ": lamhat= " << lamhat
              << ", test fnr=" << fnr << ", avg cls searched=" << computeAverage(cls) << std::endl;

    alpa = 0.3;
    lamhat = index.calibrate(alpa);
    res = index.evaluate_test(lamhat);
    fnr = res.first;
    cls = res.second;
    std::cout << "alpha=" << alpa << ": lamhat= " << lamhat
              << ", test fnr=" << fnr << ", avg cls searched=" << computeAverage(cls) << std::endl;

    // nq = 10;

    // { // searching the database
    //     printf("Searching with error quantification ...\n");

    //     std::vector<float> queries(nq * d);
    //     for (size_t i = 0; i < nq * d; i++) {
    //         queries[i] = dist(rng);
    //     }

    //     int k = 5;
    //     std::vector<faiss::idx_t> nns(k * nq);
    //     std::vector<float> dis(k * nq);

    //     auto start = high_resolution_clock::now();

    //     std::unordered_map<faiss::idx_t, std::vector<float>> nonconf_list;
    //     std::unordered_map<faiss::idx_t, std::vector<std::vector<faiss::idx_t>>>
    //         all_preds_list;

    //     index.search_with_error_quantification(nq, queries.data(), k,
    //                                            dis.data(), nns.data(), 0.022,
    //                                            nonconf_list, all_preds_list);

    //     auto end = high_resolution_clock::now();

    //     // Output results
    //     auto t = duration_cast<microseconds>(end - start).count();
    //     int qps = nq * 1000 * 1000 / t;

    //     printf("QPS: %d\n", qps);
    // }

    return 0;
}
