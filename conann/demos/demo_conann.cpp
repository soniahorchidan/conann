#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>

using namespace std::chrono;

double computeAverage(const std::vector<int>& numbers) {
    if (numbers.empty()) return 0.0;
    double sum = std::accumulate(numbers.begin(), numbers.end(), 0);
    return sum / numbers.size();
}

int main(void) {
    // dimension of the vectors to index
    int d = 128;
    int K = 100;
    int nlist = 100;

    // size of the database we plan to index
    size_t nb = 10000;
    // size of the training dataset
    size_t nt = 4000;

    std::mt19937 rng(12345);

    // make the IVF index object and train it
    faiss::IndexFlatL2 quantizer(d);  // The quantizer (flat index)
    faiss::IndexIVFFlat index(&quantizer, d, nlist,
                              faiss::METRIC_L2);  // IVF index
    index.nprobe = 100;                           // number of probes

    // train the index on some data
    std::vector<float> training_data(nt * d);  // Random training data
    std::uniform_real_distribution<float> dist(
        0.0f, 1.0f);  // Distribution between 0 and 1
    for (size_t i = 0; i < nt * d; i++) {
        training_data[i] = dist(rng);  // Generate float between 0 and 1
    }
    index.train(nt, training_data.data());

    // generate random database
    std::vector<float> database(nb * d);
    for (size_t i = 0; i < nb * d; i++) {
        database[i] = dist(rng);
    }

    {  // populate the database
        index.add(nb, database.data());
    }

    // Test calibration
    // auto alpha = 0.1;
    // auto lamhat = index.calibrate(alpha, K);
    // auto [fnr, cls] = index.evaluate_test(lamhat);
    // std::cout << "alpha=" << alpha << ": lamhat= " << lamhat
    //           << ", test fnr=" << fnr
    //           << ", avg cls searched=" << computeAverage(cls) << std::endl;

    // alpha = 0.2;
    // lamhat = index.calibrate(alpha, K);
    // auto res = index.evaluate_test(lamhat);
    // fnr = res.first;
    // cls = res.second;
    // std::cout << "alpha=" << alpha << ": lamhat= " << lamhat
    //           << ", test fnr=" << fnr
    //           << ", avg cls searched=" << computeAverage(cls) << std::endl;

    // alpha = 0.3;
    // lamhat = index.calibrate(alpha, K);
    // res = index.evaluate_test(lamhat);
    // fnr = res.first;
    // cls = res.second;
    // std::cout << "alpha=" << alpha << ": lamhat= " << lamhat
    //           << ", test fnr=" << fnr
    //           << ", avg cls searched=" << computeAverage(cls) << std::endl;

    // size_t nq = 3;

    // std::vector<float> queries(nq * d);
    // for (size_t i = 0; i < nq * d; i++) {
    //     queries[i] = dist(rng);
    // }
    //  { // searching the database
    //     printf("Searching conann ...\n");

    //     std::vector<faiss::idx_t> nns(K * nq);
    //     std::vector<float> dis(K * nq);

    //     auto start = high_resolution_clock::now();
    //     index.search_conann(nq, queries.data(), lamhat, dis.data(),
    //     nns.data());

    //     for (int i = 0; i < nq; i ++) {
    //         std::cout << "[";
    //         for (int j = 0; j < K; j ++) {
    //             std::cout << nns[i * K + j] << ", ";
    //         }
    //         std::cout << "]\n\n";
    //     }

    //     auto end = high_resolution_clock::now();

    //     // Output results
    //     auto t = duration_cast<microseconds>(end - start).count();
    //     int qps = nq * 1000 * 1000 / t;

    //     printf("QPS: %d\n", qps);
    // }

    // auto index_flat = new faiss::IndexFlatL2(d);

    // { // populate the database
    //     index_flat->add(nb, database.data());
    // }

    //  { // searching the database
    //     printf("Searching index flat ...\n");

    //     std::vector<faiss::idx_t> nns(K * nq);
    //     std::vector<float> dis(K * nq);

    //     auto start = high_resolution_clock::now();
    //     index_flat->search(nq, queries.data(), K, dis.data(), nns.data());

    //     for (int i = 0; i < nq; i ++) {
    //         std::cout << "[";
    //         for (int j = 0; j < K; j ++) {
    //             std::cout << nns[i * K + j] << ", ";
    //         }
    //         std::cout << "]\n\n";
    //     }

    //     auto end = high_resolution_clock::now();

    //     // Output results
    //     auto t = duration_cast<microseconds>(end - start).count();
    //     int qps = nq * 1000 * 1000 / t;

    //     printf("QPS: %d\n", qps);
    // }


    // // Test Mondrian
    // float alpha = 0.1;
    // auto lamhats = index.calibrate_mondrian(alpha, K);
    // for (const auto& pair : lamhats) {
    //     std::cout << "Group: " << pair.first << ", lamhat: " << pair.second << std::endl;
    // }
    // std::cout << "\n";
    // auto [fnr, cls] = index.evaluate_test_mondrian(lamhats);
    // std::cout << " avg test fnr=" << fnr
    //           << ", avg cls searched=" << computeAverage(cls) << std::endl;
    // std::cout << "\n";

    // alpha = 0.2;
    // lamhats = index.calibrate_mondrian(alpha, K);
    // for (const auto& pair : lamhats) {
    //     std::cout << "Group: " << pair.first << ", lamhat: " << pair.second << std::endl;
    // }
    // std::cout << "\n";
    // auto res = index.evaluate_test_mondrian(lamhats);
    // fnr = res.first;
    // cls = res.second;
    // std::cout << " avg test fnr=" << fnr
    //           << ", avg cls searched=" << computeAverage(cls) << std::endl;
        
    return 0;
}
