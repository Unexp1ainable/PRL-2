/**
 * @file parkmeans.cpp
 * @author Samuel Repka (xrepka07@stud.fit.vutbr.cz)
 * @brief PRL project 2
 * @date 2023-04-15
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <mpi.h>


constexpr int ROOT             = 0;
constexpr int CLUSTER_COUNT    = 4;
constexpr int MAX_DIST         = 255;
constexpr double EPSILON       = 0.000000001;
constexpr const char* FILEPATH = "numbers";


using ByteVector = std::vector<unsigned char>; // got tired of writing "std::vector<unsigned char>" everywhere


/**
 * @brief Helper class for RAII implementation
 *
 * @tparam T
 */
template <class T> class RAII {
public:
    explicit RAII(T func)
        : m_func(func)
    {
    }
    ~RAII() { m_func(); }
    RAII(const RAII&)       = delete;
    RAII(RAII&&)            = delete;
    RAII& operator=(RAII&)  = delete;
    RAII& operator=(RAII&&) = delete;

private:
    T m_func;
};


/**
 * @brief Load numbers from file
 *
 * @param path Path to file
 * @return ByteVector Loaded numbers
 */
ByteVector loadNumbers(const char* path)
{
    std::ifstream inFile(path);

    // get length of file
    inFile.seekg(0, std::ios::end);
    const long length = inFile.tellg();
    inFile.seekg(0, std::ios::beg);

    ByteVector numbers {};
    std::vector<char> numbersSigned {};
    numbers.resize(length);
    numbersSigned.resize(length);
    // read file
    inFile.read(numbersSigned.data(), length);
    // read loads chars, I want unsigned chars so I do a quick conversion here
    memcpy(numbers.data(), numbersSigned.data(), length);

    return numbers;
}

/**
 * @brief Check if the process should calculate centroid or not
 *
 * @param rank Rank of the process
 * @return true Process should not calculate centroid
 * @return false Otherwise
 */
inline bool calculatesCentroid(const int rank) { return rank >= 0 && rank < CLUSTER_COUNT; }


/**
 * @brief Load numbers. Numbers will be trimmed to commSize if too large,
 *        or the program will be terminated, if number of numbers is insufficient
 *
 * @param dst Destination vector
 * @param commSize Number of processes available.
 */
void loadNumbersChecked(ByteVector& dst, int commSize)
{
    // load numbers
    dst = loadNumbers(FILEPATH);

    if (dst.size() < commSize || dst.size() < CLUSTER_COUNT) {
        std::cerr << "Not enough input dst.\n";
        exit(1);
    }
    if (dst.size() > commSize) {
        dst.resize(commSize);
    }
}

/**
 * @brief Calculate to which cluster myNumber belongs
 *
 * @param centroids Centroids
 * @param myNumber Number for which to find a centroid
 * @return int Index of the result centroid
 */
int assignToCluster(std::array<double, CLUSTER_COUNT>& centroids, uint8_t myNumber)
{
    // assign to clusters
    double smallestDistance = MAX_DIST;
    int count               = 0;
    int cluster             = 0;

    for (int i = 0; i < CLUSTER_COUNT; i++) {
        double dist = std::abs(myNumber - centroids[i]);
        if (dist < smallestDistance) {
            smallestDistance = dist;
            cluster          = i;
        }

        // this part makes sure, that if the number can be assigned to multiple clusters,
        // it is chosen at random which cluster it belongs to with same chance for all potential
        // clusters (in O(1) space - (surprisingly, this is TIN knowledge in practice))
        if (std::fabs(smallestDistance - dist) < EPSILON) {
            count++;
            if (std::rand() % count == (count - 1)) {
                cluster = i;
            }
        }
    }

    return cluster;
}

/**
 * @brief Gathers assignments of numbers to a cluster
 *
 * @param assignments Vector of 0 and 1, 0 if number does not belong to this cluster, 1 otherwise
 * @param cluster Cluster for which results should be gathered
 */
void distributeResults(ByteVector& assignments, int cluster)
{
    for (int i = 0; i < CLUSTER_COUNT; i++) {
        auto toSend = static_cast<uint8_t>(cluster == i);
        MPI_Gather(&toSend, 1, MPI_UINT8_T, assignments.data(), 1, MPI_UINT8_T, i, MPI::COMM_WORLD);
    }
}

/**
 * @brief Recalculate centroid belonging to this process (if applicable) according to new assignments
 *
 * @param rank Process rank
 * @param numbers Input numbers
 * @param centroids Reference to old centroids
 * @param assignments New centroid assignments
 */
void recalculateCentroid(
    int rank,
    ByteVector& numbers,
    std::array<double, CLUSTER_COUNT>& centroids,
    ByteVector& assignments
)
{
    if (calculatesCentroid(rank)) {
        double sum = 0.;
        int count  = 0;
        for (int i = 0; i < assignments.size(); i++) {

            if (assignments[i] == 1) {
                sum += numbers[i];
                count++;
            }
        }

        if (count != 0) {
            centroids[rank] = sum / count;
        }
    }
}

/**
 * @brief Function implementing kmeans calculation
 *
 * @param centroids Initialized centroids (result will be there)
 * @param numbers Input numbers
 * @param myNumber Number for this process
 * @return int Cluster to which myNumber was assigned
 */
int kmeansLoop(std::array<double, CLUSTER_COUNT>& centroids, ByteVector& numbers, uint8_t myNumber)
{
    // acquire data about the processes
    int commSize = 0;
    int rank     = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int cluster;

    while (true) {
        // number of processors should be the same as input count
        std::vector<uint8_t> assignments(commSize, 0);
        cluster = assignToCluster(centroids, myNumber);

        distributeResults(assignments, cluster);

        // calculate new centroids
        auto oldCentroids = centroids;
        recalculateCentroid(rank, numbers, centroids, assignments);

        // distribute new centroids
        for (int i = 0; i < CLUSTER_COUNT; i++) {
            MPI_Bcast(centroids.data() + i, 1, MPI_DOUBLE, i, MPI::COMM_WORLD);
        }

        // end if fixed point was reached
        if (centroids == oldCentroids) {
            break;
        }
    }

    return cluster;
}

/**
 * @brief Print final results to the console
 *
 * @param centroids Calculated centroids
 * @param finalAssignment Vector of numbers representing to which centroid number belongs
 * @param numbers Input numbers
 */
void printResults(
    std::array<double, CLUSTER_COUNT>& centroids,
    std::vector<int>& finalAssignment,
    ByteVector& numbers
)
{
    for (int i = 0; i < CLUSTER_COUNT; i++) {
        std::cout << "[" << centroids[i] << "] ";

        bool first = true;
        for (int j = 0; j < numbers.size(); j++) {
            if (i == finalAssignment[j]) {
                if (first) {
                    first = false;
                } else {
                    std::cout << ", ";
                }
                std::cout << +numbers[j];
            }
        }
        std::cout << "\n";
    }
}

int main(int argc, char** argv)
{
    // initialize and prepare finalization
    MPI_Init(&argc, &argv);
    auto raii = RAII([]() { MPI_Finalize(); });

    // acquire data about the processes
    int commSize = 0;
    int rank     = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // load numbers
    std::array<double, CLUSTER_COUNT> centroids {};
    auto numbers     = ByteVector {};
    uint8_t myNumber = 0;

    if (rank == 0) {
        loadNumbersChecked(numbers, commSize);
        std::copy(numbers.begin(), numbers.begin() + CLUSTER_COUNT, centroids.begin());
    }

    // distribute numbers to processes that will calculate centroids
    if (rank == ROOT) {
        for (int i = 1; i < CLUSTER_COUNT; i++) {
            MPI_Send(numbers.data(), commSize, MPI_UINT8_T, i, 0, MPI::COMM_WORLD);
        }
    } else if (calculatesCentroid(rank)) {
        numbers.resize(commSize);
        MPI_Recv(numbers.data(), commSize, MPI_UINT8_T, ROOT, 0, MPI::COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // broadcast centroids and assign numbers to processes
    MPI_Bcast(centroids.data(), centroids.size(), MPI_DOUBLE, ROOT, MPI::COMM_WORLD);
    MPI_Scatter(numbers.data(), 1, MPI_UINT8_T, &myNumber, 1, MPI_UINT8_T, ROOT, MPI::COMM_WORLD);

    // calculate final cluster for assigned number
    int cluster = kmeansLoop(centroids, numbers, myNumber);

    // gather results
    std::vector<int> finalAssignment(numbers.size(), 0);
    MPI_Gather(&cluster, 1, MPI_INT, finalAssignment.data(), 1, MPI_INT, ROOT, MPI::COMM_WORLD);

    if (rank == ROOT) {
        printResults(centroids, finalAssignment, numbers);
    }
    return 0;
}
