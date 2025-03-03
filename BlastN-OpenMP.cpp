#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <fstream>
#include <chrono>
#include <omp.h>


using namespace std;
using namespace chrono;



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
        for (int idxSeq = 1; idxSeq < dbSequences.size(); idxSeq++) { 
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
//<=============BLAST PARALLEL===============>
//BlastN - Paralelizado con OpenMP
Seed blastnParallel(vector<std::string>& dbSequences, int W, int X) {
    std::vector<std::string> queryWmers;
    genrateWmers_sequential(dbSequences, queryWmers, W);

    Seed best_seed = { 0, 0, 0 };

    #pragma omp parallel
    {
        Seed local_best_seed = { 0, 0, 0 }; 

        #pragma omp for nowait
        for (int i = 0; i < queryWmers.size(); i++) {
            for (int idxSeq = 1; idxSeq < dbSequences.size(); idxSeq++) {
                for (int j = 0; j <= dbSequences[idxSeq].size() - W; j++) {
                    if (queryWmers[i] == dbSequences[idxSeq].substr(j, W)) {
                        Seed seed = extendSeed(dbSequences[0], dbSequences[idxSeq], i, j, X);
                        if (seed.score > local_best_seed.score) {
                            local_best_seed = { seed.pos, seed.score, idxSeq };
                        }
                    }
                }
            }
        }

        #pragma omp critical
        {
            if (local_best_seed.score > best_seed.score) {
                best_seed = local_best_seed;
            }
        }
    }

    return best_seed;
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
    int W = 5;  // Longitud del W-meros
    int X = -3; //Umbral

    // Leer los archivos FASTA
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
   Seed resultPar = blastnParallel(dbSequences, W, X);
   auto end_par = high_resolution_clock::now();
   double tiempo_paralelo = duration<double>(end_par - start_par).count();
   



    cout << "Tiempo Secuencial: " << tiempo_secuencial << " segundos" << endl;
    cout << "Tiempo Paralelo: " << tiempo_paralelo << " segundos" << endl;
    cout << "Speedup: " << tiempo_secuencial / tiempo_paralelo << "x" << endl;

    cout << "Query: " << queryName << endl;
    printResult("Secuencial", resultSeq.idxSeq - 1, dbHeader);
    printResult("Paralelo", resultPar.idxSeq - 1, dbHeader);

}

