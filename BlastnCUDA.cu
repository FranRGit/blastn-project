#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <fstream>
#include <chrono>
#include <omp.h>
#include <cuda_runtime.h>

using namespace std;
using namespace chrono;



// Definición de la estructura Seed
struct Seed {
    int pos;      // Posición del W-mero en la secuencia
    int score;    // Puntaje 
    int idxSeq;   // Identificador de la secuencia (índice de la secuencia en el archivo)
};

// 1- Entrada: Query y Secuencias BD 
void leer_fasta(const string& filename, vector<string>& names, vector<string>& sequences) {
    ifstream file(filename);
    if (!file) {
        cerr << "Error al abrir el archivo " << filename << endl;
        exit(1);
    }

    string line, sequence;
    string header;
    while (getline(file, line)) {
        if (line.empty()) continue;

        if (line[0] == '>') {
            if (!header.empty()) {
                sequences.push_back(sequence);
                sequence.clear();
            }
            names.push_back(line.substr(1));
            header = line;
        }
        else {
            sequence += line;
        }
    }

    if (!header.empty()) {
        sequences.push_back(sequence);
    }

    file.close();
}


//<=============BLAST SEQUENTIAL===============>
// 1- División de palabras - Creación de Wmers
void genrateWmers_sequential(vector <string>& sequences, vector <string>& queryWmers, int W) {
    for (int idxSeq = 0; idxSeq < sequences.size(); idxSeq++) {
        if (idxSeq == 0) {
            for (int i = 0; i <= sequences[idxSeq].size() - W; i++) {
                queryWmers.push_back(sequences[idxSeq].substr(i, W));
            }
        }
    }
}

//2-  Extensión de coincidencias
Seed extendSeed(const string& query, const string& sequence, int startQuery, int startDb, int X) {
    int score = 0, i = 0;
    while (startQuery + i < query.size() && startDb + i < sequence.size()) {
        if (query[startQuery + i] == sequence[startDb + i]) {
            score++;
        }
        else {
            score--;
        }
        if (score < X) break;
        i++;
    }
    return { startQuery, score };
}

//3- BlastN - Secuencial
Seed blastnSequential(vector<string>& dbSequences, int W, int X) {
    vector<string> queryWmers;

    genrateWmers_sequential(dbSequences, queryWmers, W);

    Seed best_seed = { 0, 0, 0 };

    for (int i = 0; i < queryWmers.size(); i++) {
        for (int idxSeq = 1; idxSeq < dbSequences.size(); idxSeq++) { // Excluyendo la consulta (dbSequences[0])
            for (int j = 0; j <= dbSequences[idxSeq].size() - W; j++) {
                if (queryWmers[i] == dbSequences[idxSeq].substr(j, W)) {
                    Seed seed = extendSeed(dbSequences[0], dbSequences[idxSeq], i, j, X);
                    if (seed.score > best_seed.score) {
                        best_seed = { seed.pos, seed.score, idxSeq };
                    }
                }
            }
        }
    }

    return best_seed;
}

//<=============BLAST PARALELO===============>
__device__ int computeScore(const char* query, const char* sequence, int startQuery, int startDb, int querySize, int sequenceSize, int X) {
    int score = 0, i = 0;
    while (startQuery + i < querySize && startDb + i < sequenceSize) {
        if (query[startQuery + i] == sequence[startDb + i]) {
            score += 2; 
        }
        else {
            score -= 1; 
        }
        if (score < X) break;
        i++;
    }
    return score;
}

__device__ bool strncmpCUDA(const char* str1, const char* str2, int n) {
    for (int i = 0; i < n; i++) {
        if (str1[i] != str2[i]) {
            return false;
        }
    }
    return true;
}

__global__ void blastnKernel(const char* query, const char* dbSequences, int* sequenceOffsets, int* sequenceLengths, int numSequences, int W, int X, Seed* bestSeed) {
    int idxSeq = blockIdx.x;
    int threadId = threadIdx.x;
    if (idxSeq == 0) return; // Evitar la consulta misma

    __shared__ Seed localBestSeed;
    if (threadId == 0) {
        localBestSeed = { 0, 0, idxSeq };
    }
    __syncthreads();

    int seqStart = sequenceOffsets[idxSeq];
    int seqLength = sequenceLengths[idxSeq];
    int queryLength = sequenceLengths[0];

    for (int j = threadId; j <= seqLength - W; j += blockDim.x) {
        for (int i = 0; i <= queryLength - W; i++) {
            if (strncmpCUDA(&query[i], &dbSequences[seqStart + j], W)) {
                int score = computeScore(query, &dbSequences[seqStart], i, j, queryLength, seqLength, X);
                if (score > localBestSeed.score) {
                    localBestSeed = { i, score, idxSeq };
                }
            }
        }
    }

    __syncthreads();
    if (threadId == 0) {
        atomicMax(&bestSeed->score, localBestSeed.score);
        if (bestSeed->score == localBestSeed.score) {
            bestSeed->pos = localBestSeed.pos;
            bestSeed->idxSeq = localBestSeed.idxSeq;
        }
    }
}

Seed blastnCUDA(const vector<string>& dbSequences, int W, int X) {
    char* d_dbSequences;
    int* d_sequenceOffsets;
    int* d_sequenceLengths;
    Seed* d_bestSeed;
    Seed bestSeed = { 0, 0, 0 };

    int numSequences = dbSequences.size();
    vector<int> sequenceOffsets(numSequences);
    vector<int> sequenceLengths(numSequences);
    int totalSize = 0;

    for (int i = 0; i < numSequences; i++) {
        sequenceOffsets[i] = totalSize;
        sequenceLengths[i] = dbSequences[i].size();
        totalSize += dbSequences[i].size();
    }

    cudaMalloc(&d_dbSequences, totalSize * sizeof(char));
    cudaMalloc(&d_sequenceOffsets, numSequences * sizeof(int));
    cudaMalloc(&d_sequenceLengths, numSequences * sizeof(int));
    cudaMalloc(&d_bestSeed, sizeof(Seed));

    cudaMemcpy(d_sequenceOffsets, sequenceOffsets.data(), numSequences * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sequenceLengths, sequenceLengths.data(), numSequences * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bestSeed, &bestSeed, sizeof(Seed), cudaMemcpyHostToDevice);

    string concatenatedSequences;
    for (const auto& seq : dbSequences) {
        concatenatedSequences += seq;
    }
    cudaMemcpy(d_dbSequences, concatenatedSequences.c_str(), totalSize * sizeof(char), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    blastnKernel << <numSequences, threadsPerBlock >> > (d_dbSequences, d_dbSequences, d_sequenceOffsets, d_sequenceLengths, numSequences, W, X, d_bestSeed);

    cudaMemcpy(&bestSeed, d_bestSeed, sizeof(Seed), cudaMemcpyDeviceToHost);

    cudaFree(d_dbSequences);
    cudaFree(d_sequenceOffsets);
    cudaFree(d_sequenceLengths);
    cudaFree(d_bestSeed);

    return bestSeed;
}

void printResult(const string& method, int index, const vector<string>& dbHeader) {
    if (index != -1) {
        cout << "Mejor coincidencia (" << method << "): " << dbHeader[index] << endl;
    }
    else {
        cout << "No se encontró coincidencia en la versión " << method << "." << endl;
    }
}


int main() {
    vector<string> dbHeader, dbSequences, queryHeader;
    string query_file = "";
    string database_file = "";
    int W = 11;  // Longitud del W-meros
    int X = -3; //Umbral

    leer_fasta(query_file, queryHeader, dbSequences);
    leer_fasta(database_file, dbHeader, dbSequences);

    if (dbSequences.empty()) {
        cerr << "Error: No se encontraron secuencias en los archivos FASTA" << endl;
        return 1;
    }

    string queryName = queryHeader[0];


    auto start_seq = high_resolution_clock::now();
    Seed resultSeq = blastnSequential(dbSequences, W, X);
    auto end_seq = high_resolution_clock::now();
    double tiempo_secuencial = duration<double>(end_seq - start_seq).count();
   
 
    
    auto start_par = high_resolution_clock::now();
    Seed resultPar = blastnCUDA(dbSequences, W, X);
    auto end_par = high_resolution_clock::now();
    double tiempo_paralelo = duration<double>(end_par - start_par).count();




    cout << "Tiempo Secuencial: " << tiempo_secuencial << " segundos" << endl;
    cout << "Tiempo Paralelo: " << tiempo_paralelo << " segundos" << endl;
    cout << "Speedup: " << tiempo_secuencial / tiempo_paralelo << "x" << endl;

    cout << "Query: " << queryName << endl;
    printResult("Secuencial", resultSeq.idxSeq - 1, dbHeader);
    printResult("Paralelo", resultPar.idxSeq - 1, dbHeader);

}

