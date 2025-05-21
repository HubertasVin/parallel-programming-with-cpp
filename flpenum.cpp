#include <iostream>
#include <cmath>
#include <time.h>
#include <sys/time.h>
#include <mpi.h>  // Include MPI header

using namespace std;

//===== Globalus kintamieji ===================================================

int numDP = 5000;               // Vietoviu skaicius (demand points, max 10000)
int numPF = 5;                  // Esanciu objektu skaicius (preexisting facilities)
int numCL = 50;                 // Kandidatu naujiems objektams skaicius (candidate locations)
int numX  = 3;                  // Nauju objektu skaicius

double **demandPoints;          // Geografiniai duomenys
double **distanceMatrix;        // Masyvas atstumu matricai saugoti

int *X = new int[numX];         // Naujas sprendinys
int *bestX = new int[numX];     // Geriausias rastas sprendinys
double u, bestU;                // Naujo sprendinio ir geriausio sprendinio naudingumas (utility)

//===== Funkciju prototipai ===================================================

double getTime();
void loadDemandPoints();
double HaversineDistance(double lat1, double lon1, double lat2, double lon2);
double HaversineDistance(int i, int j);
double evaluateSolution(int* X);
int increaseX(int* X, int index, int maxindex);

//=============================================================================

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    loadDemandPoints();

    //----- Atstumų matricos skaičiavimas -------------------------------------
    int totalElements = (numDP * (numDP + 1)) / 2;

    double* distanceMatrix_flat = new double[totalElements];
    // Inicializuojame masyvą nuliais
    memset(distanceMatrix_flat, 0, totalElements * sizeof(double));

    double t_start = getTime();
    int idx;
    for (int i = 0; i < numDP; i++) {
        if (i % size == rank) {
            for (int j = 0; j <= i; j++) {
                idx = i * (i + 1) / 2 + j;
                distanceMatrix_flat[idx] = HaversineDistance(demandPoints[i][0], demandPoints[i][1],
                                                            demandPoints[j][0], demandPoints[j][1]);
            }
        }
    }

    MPI_Allreduce(MPI_IN_PLACE, distanceMatrix_flat, totalElements, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    // Atstatome dvimatę atstumų matricą iš vienmačio masyvo
    distanceMatrix = new double*[numDP];
    for (int i = 0; i < numDP; i++) {
        distanceMatrix[i] = new double[i + 1];
        for (int j = 0; j <= i; j++) {
            idx = i * (i + 1) / 2 + j;
            distanceMatrix[i][j] = distanceMatrix_flat[idx];
        }
    }

    delete[] distanceMatrix_flat;
    double t_matrix = getTime();
    if (rank == 0) {
        printf("Matricos skaičiavimo trukmė: %f\n", t_matrix - t_start);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    //----- Pradinių naujo ir geriausio sprendinių reikšmės -------------------
    for (int i = 0; i < numX; i++) {
        X[i] = i;
        bestX[i] = i;
    }
    u = evaluateSolution(X);
    double localBestU = u;
    int localBestX[numX];
    for (int i = 0; i < numX; i++) localBestX[i] = X[i];

    //----- Visų galimų sprendinių perrinkimas --------------------------------
    int iteration = 0;
    while (increaseX(X, numX - 1, numCL) == true) {
        if (iteration % size == rank) {
            u = evaluateSolution(X);
            if (u > localBestU) {
                localBestU = u;
                for (int i = 0; i < numX; i++) localBestX[i] = X[i];
            }
        }
        iteration++;
    }

    struct {
        double value;
        int index;
    } in, out;

    in.value = localBestU;
    in.index = rank;

    MPI_Allreduce(&in, &out, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);

    if (rank == 0) {
        if (out.index != 0) {
            MPI_Recv(bestX, numX, MPI_INT, out.index, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
            for (int i = 0; i < numX; i++) bestX[i] = localBestX[i];
        }
        double t_finish = getTime();     // Skaičiavimų pabaigos laikas
        printf("Sprendinio paieškos trukmė: %f\n", t_finish - t_matrix);
        printf("Algoritmo vykdymo trukmė: %f\n", t_finish - t_start);
        printf("Geriausias sprendinys: ");
        for (int i = 0; i < numX; i++) printf("%d ", bestX[i]);
        printf("(%.2f procentai rinkos)\n", out.value);
    } else {
        if (rank == out.index) {
            MPI_Send(localBestX, numX, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
    }

    for (int i = 0; i < numDP; i++) {
        delete[] distanceMatrix[i];
    }
    delete[] distanceMatrix;

    MPI_Finalize();
    return 0;
}

//===== Funkciju implementacijos (siu funkciju LYGIAGRETINTI NEREIKIA) ========

void loadDemandPoints() {
    FILE *f;
    f = fopen("demandPoints.dat", "r");
    demandPoints = new double*[numDP];
    for (int i = 0; i < numDP; i++) {
        demandPoints[i] = new double[3];
        fscanf(f, "%lf%lf%lf", &demandPoints[i][0], &demandPoints[i][1], &demandPoints[i][2]);
    }
    fclose(f);
}

//=============================================================================

double HaversineDistance(double lat1, double lon1, double lat2, double lon2) {
    double dlat = fabs(lat1 - lat2);
    double dlon = fabs(lon1 - lon2);
    double aa = pow((sin((double)dlat / (double)2 * 0.01745)), 2) + cos(lat1 * 0.01745) *
        cos(lat2 * 0.01745) * pow((sin((double)dlon / (double)2 * 0.01745)), 2);
    double c = 2 * atan2(sqrt(aa), sqrt(1 - aa));
    double d = 6371 * c;
    return d;
}

double HaversineDistance(int i, int j) {
    if (i >= j)	return distanceMatrix[i][j];
    else return distanceMatrix[j][i];
}

//=============================================================================

double getTime() {
    struct timeval laikas;
    gettimeofday(&laikas, NULL);
    double rez = (double)laikas.tv_sec + (double)laikas.tv_usec / 1000000;
    return rez;
}

//=============================================================================

double evaluateSolution(int *X) {
    double U = 0;
    double totalU = 0;
    int bestPF;
    int bestX;
    double d;
    for (int i = 0; i < numDP; i++) {
        totalU += demandPoints[i][2];
        bestPF = 1e5;
        for (int j = 0; j < numPF; j++) {
            d = HaversineDistance(i, j);
            if (d < bestPF) bestPF = d;
        }
        bestX = 1e5;
        for (int j = 0; j < numX; j++) {
            d = HaversineDistance(i, X[j]);
            if (d < bestX) bestX = d;
        }
        if (bestX < bestPF) U += demandPoints[i][2];
        else if (bestX == bestPF) U += 0.3 * demandPoints[i][2];
    }
    return U / totalU * 100;
}

//=============================================================================

int increaseX(int *X, int index, int maxindex) {
    if (X[index] + 1 < maxindex - (numX - index - 1)) {
        X[index]++;
    }
    else {
        if ((index == 0) && (X[index] + 1 == maxindex - (numX - index - 1))) {
            return 0;
        }
        else {
            if (increaseX(X, index - 1, maxindex)) X[index] = X[index - 1] + 1;
            else return 0;
        }
    }
    return 1;
}
