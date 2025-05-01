#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <sys/time.h>
#include <omp.h>

#include "faiss/AutoTune.h"
#include "faiss/IndexIVF.h"
#include "faiss/index_io.h"
#include "faiss/profile.h"
#include "faiss/index_factory.h"

#include<iostream>
#include<fstream>

/*****************************************************
 * I/O functions for fvecs and ivecs
 *****************************************************/

#define DC(classname) classname* ix = dynamic_cast<classname*>(index)

// template <typename T> double computeAverage(const std::vector<T> &numbers) {
//     if (numbers.empty())
//         return 0.0;
//     double sum = std::accumulate(numbers.begin(), numbers.end(), 0.0);
//     return sum / numbers.size();
// }

// template <typename T>
// void write_to_file(const std::vector<T> &data, const std::string &filename) {
//     std::ofstream file(filename);
//     for (const auto &value : data) {
//         file << value << '\n';
//     }
//     file.close();
// }

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

/// Command like this: ./knn_script sift1M 100 0.66 0.1
int main(int argc, char **argv) {
    std::cout << argc << " arguments" <<std::endl;
    if(argc <= 4){
        printf("You should at least input 4 params: the dataset name, topk, train size fraction, alpha\n");
        return 0;
    }
    std::string param1 = argv[1];
    std::string param2 = argv[2];
    std::string param3 = argv[3];
    std::vector<float> alphas;
    for (int i = 4; i < argc; i++) {
        alphas.push_back(std::stof(argv[i]));
    }

    int input_k = std::stoi(param2);
    float training_fraction = std::stof(param3);
    
    int figureid = -1;
    // if(input_k>100 || input_k <0){
    //     printf("Input topk must be lower than or equal to 100 and greater than 0\n");
    //     return 0;
    // }
    std::string db, query, gtI, gtD;
    if (param1 == "bert") {
        db = "../data/bert/db.fvecs";
        query = "../data/bert/queries.fvecs";
        gtI = "../data/bert/indices-" + std::to_string(input_k) + ".fvecs";
        gtD = "../data/bert/distances-" + std::to_string(input_k) + ".fvecs";
        figureid = 11;
    // else if(param1 == "sift10k"){
    //     db = "../../data/sift10k/siftsmall_base.fvecs";
    //     query = "../../data/sift10k/siftsmall_query.fvecs";
    //     gtI = "../../data/sift10k/sift10k_gt_indices_k10.fvecs";
    //     gtD = "../../data/sift10k/sift10k_gt_distances_k10.fvecs";
    //     figureid = 9;
    } else if (param1 == "synth") {
        db = "../data/synthetic10/db.fvecs";
        query = "../data/synthetic10/queries.fvecs";
        gtI = "../data/synthetic10/indices-" + std::to_string(input_k) + ".fvecs";
        gtD = "../data/synthetic10/distances-" + std::to_string(input_k) + ".fvecs";
        figureid = 9;
    } else if (param1 == "sift1M") {
        db = "../data/sift1M/sift_base.fvecs";
        query = "../data/sift1M/queries.fvecs";
        gtI = "../data/sift1M/indices-" + std::to_string(input_k) + ".fvecs";
        gtD = "../data/sift1M/distances-" + std::to_string(input_k) + ".fvecs";
        figureid = 9;
    } else if (param1 == "deep10M") {
        figureid = 10;
        db = "../data/deep/deep10M.fvecs";
        query = "../data/deep/queries.fvecs";
        gtI = "../data/deep/indices-" + std::to_string(input_k) + ".fvecs";
        gtD = "../data/deep/distances-" + std::to_string(input_k) + ".fvecs";
    } else if (param1 == "gist") {
        figureid = 11;
        db = "../data/gist/gist_base.fvecs";
        query = "../data/gist/queries.fvecs";
        gtI = "../data/gist/indices-" + std::to_string(input_k) + ".fvecs";
        gtD = "../data/gist/distances-" + std::to_string(input_k) + ".fvecs";
    // else if(param1 == "spacev"){
    //     db = "/workspace/data/spacev/spacev10M.fvecs";
    //     query = "/workspace/data/spacev/query.fvecs";
    //     gtI = "/workspace/data/spacev/idx.fvecs";
    //     gtD = "/workspace/data/spacev/dis.fvecs";
    // }
    } else if (param1 == "glove") {
        figureid = 9;
        db = "../data/glove/db.fvecs";
        query = "../data/glove/queries.fvecs";
        gtI = "../data/glove/indices-" + std::to_string(input_k) + ".fvecs";
        gtD = "../data/glove/distances-" + std::to_string(input_k) + ".fvecs";
    // } else if(param1 == "text"){
    //     figureid = 12;
    //     db = "/workspace/data/text/text10M.fvecs";
    //     query = "/workspace/data/text/query.fvecs";
    //     gtI = "/workspace/data/text/idx.fvecs";
    //     gtD = "/workspace/data/text/dis.fvecs";
    // }
    } else{
        printf("Your dataset name is illegal\n");
        return 1;
    }

	omp_set_num_threads(32);
    double t0 = elapsed();


    // {
    //     // FOR DEBUG
    //     printf("[%.3f s] Loading ground truth queries for debug\n");
    //     size_t k;                // nb of results per query in the GT
    //     faiss::idx_t* gt; // nq * k matrix of ground-truth nearest-neighbors
    //     // CHANGED TO
    //     // faiss::idx_t* gt; // nq * k matrix of ground-truth nearest-neighbors
    //     //        elapsed() - t0,
    //     //        nq);

    //     // load ground-truth and convert int to long
    //     size_t nq2;
    //     int* gt_int = ivecs_read(gtI.c_str(), &k, &nq2);
    //     // assert(nq2 == nq || !"incorrect nb of ground truth entries");

    //     // gt = new faiss::Index::idx_t[k * nq];
    //     // CHANGED TO
    //     gt = new faiss::idx_t[k * nq2];
    //     for (int i = 0; i < k * nq2; i++) {
    //         gt[i] = gt_int[i];
    //     }
    //     delete[] gt_int;

    //     // Print the first 20 elements of the gt_int array
    //     std::cout << "First 20 elements of gt_int array: ";
    //     for (int i = 0; i < 20 && i < k * nq2; i++) {
    //         std::cout << gt[i] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // exit(0);

    // this is typically the fastest one.
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
        // if(param1 == "bert" || param1 == "sift10k" || param1 == "glove" || param1 == "sift1M" || param1 == "sift10M" || param1 == "deep10M" || param1 == "gist" || param1 == "spacev")
        index = faiss::index_factory(d, index_key);
        // else
            // index = faiss::index_factory(d, index_key
            // ,faiss::METRIC_INNER_PRODUCT
            // );

        // index->set_tune_mode();
        // if(DC(faiss::IndexIVF)){
        //     printf("Output tune type: %d %d\n", index->tune, ix->quantizer->tune);
        // }
        
        printf("Output index type: %d\n", index->type);

        printf("[%.3f s] Training on %ld vectors\n", elapsed() - t0, nb);

        // train on half the dataset
        auto nt = size_t(0.5 * nb);
        printf("[%.3f s] Training on %ld vectors\n", elapsed() - t0, nt);

        index->set_tune_mode();
        index->train(nt, xb);
        index->set_tune_off();
        // std::string filenameIn = "./trained_index/";
        // filenameIn += param1;
        // filenameIn += "_IVF1024,Flat_trained.index";
        // faiss::write_index(index, filenameIn.c_str());
    

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
    // faiss::Index::idx_t* gt; // nq * k matrix of ground-truth nearest-neighbors
    // CHANGED TO
    faiss::idx_t* gt; // nq * k matrix of ground-truth nearest-neighbors

    {
        printf("[%.3f s] Loading ground truth for %ld queries\n",
               elapsed() - t0,
               nq);

        // load ground-truth and convert int to long
        size_t nq2;
        int* gt_int = ivecs_read(gtI.c_str(), &k, &nq2);
        assert(nq2 == nq || !"incorrect nb of ground truth entries");

        // gt = new faiss::Index::idx_t[k * nq];
        // CHANGED TO
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
        // std::cout << "simi vector: ";
        // for (size_t index = 100*k; index < 100*k + k; index++) {
        //     std::cout << gt_v[index] << " ";
        // }
        // std::cout << std::endl;
        assert(kk == k || !"gt diatance does not have same dimension as gt IDs");
        assert(nq3 == nq || !"incorrect nb of ground truth entries");
    }

    size_t topk = k;
    size_t max_topk = k;

    // Round down to the nearest number divisible by ten (necessary for auncel, don't know why)
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
        faiss::Error_sys err_sys(index, nq , k);

        err_sys.set_gt(gt_v, gt);
        printf("[%.3f s] Start error profile system training\n",
               elapsed() - t0);
        err_sys.sys_train(ts, xq);
        printf("[%.3f s] Finish error profile system training\n",
               elapsed() - t0);

        // std::vector<float> alphas = {0.2, 0.1, 0.05};
        for (float alpha : alphas) {
            std::vector<float> D;
            std::vector<int64_t> I;
            std::vector<float> acc;
            size_t demo_size = ses;
            topk = input_k;
            // Set query topk val
            err_sys.set_topk(topk);
            D.resize(demo_size * k);
            I.resize(demo_size * k);
            // Set required recalls
            // std::vector<float> accs = {0.95, 0.9, 0.8};
            std::vector<float> accs = {1-alpha};
            for(int i = 0; i<demo_size+ts;i++){
                int index = i%accs.size();
                acc.push_back(accs[index]);
            }
            
            err_sys.set_queries(demo_size, xq, acc.data(), ts+ses);
            printf("[%.3f s] Start error profile system search for alpha: %.3f\n",
                elapsed() - t0, alpha);
            t0 = elapsed();
            if(DC(faiss::IndexIVF)){
                assert(figureid >= 1 && figureid <= 12);
                ix->t->setparam(figureid);
                ix->t->profile = true;
            }
            err_sys.search(D.data(), I.data(), ts);
            printf("Finish error profile system search: %.3f\n",
                elapsed() - t0);

            if(DC(faiss::IndexIVF)){
                /// log results
                {
                std::ostringstream fnr_filename;
                fnr_filename << "../Auncel-error-" << param1 << "-" << k << "-"
                            << alpha << ".log";
                std::string filename = fnr_filename.str();
                std::ofstream outfile;
                outfile.open(filename);
                for(int i = ts;i < (ts+ses); i++){
                    outfile << ix->t->t_fnrs[i] << " ";
                    outfile << std::endl;
                }
                }

                {
                std::ostringstream cls_filename;
                cls_filename << "../Auncel-efficiency-" << param1 << "-" << k << "-"
                            << alpha << ".log";
                std::string filename = cls_filename.str();
                std::ofstream outfile;
                outfile.open(filename);
                for(int i = ts;i < (ts+ses); i++){
                    outfile << ix->t->t_cls[i] << " ";
                    outfile << std::endl;
                }
                }
            }
        }
    }
    delete[] xq;
    delete[] gt;
    delete[] gt_v;
    delete index;
    return 0;
}
