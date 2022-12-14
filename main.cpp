#include <iostream>
#include <cmath>
#include <random>
#include <omp.h>

static const int numPoints = 1000000;
static const int numCentroids = 20;
static const int seed = 27;

struct centroid {
    double x{0}, y{0};
    int numPointsAssigned{0};
};

struct point {
    double x{0}, y{0};
    int assignedCentroid{-1};
};

point *pointGenerator(std::default_random_engine &generator, std::uniform_real_distribution<double> &random);

centroid *centroidGenerator(std::default_random_engine &generator, std::uniform_real_distribution<double> &random);


int main() {
    std::default_random_engine generator(seed);
    std::uniform_real_distribution<double> random(0.0, 1000.0);

    point *points = pointGenerator(generator, random);

    double start = omp_get_wtime();
    int change;

    centroid *centroids = centroidGenerator(generator, random);
    printf("\n");

#pragma omp parallel default(none) shared(points, centroids, numPoints, numCentroids, change)
    {

        do {
            centroid privateCentroid[numCentroids];
#pragma omp barrier

#pragma omp single
            change = false;

#pragma omp for
            for (int i = 0; i < numPoints; i++) {
                double minDistance = INFINITY;
                int centroid = -1;

                for (int j = 0; j < numCentroids; j++) {
                    double distance = pow(points[i].x - centroids[j].x, 2) +
                                      pow(points[i].y - centroids[j].y, 2);

                    if (distance < minDistance) {
                        minDistance = distance;
                        centroid = j;
                    }
                }

                if (centroid != points[i].assignedCentroid) {
                    points[i].assignedCentroid = centroid;
                    change = true;
                }

                privateCentroid[centroid].numPointsAssigned++;
                privateCentroid[centroid].x += points[i].x;
                privateCentroid[centroid].y += points[i].y;

            }

            if (change) {
#pragma omp single
                centroids = new centroid[numCentroids];

                for (int j = 0; j < numCentroids; j++) {
#pragma omp atomic update
                    centroids[j].numPointsAssigned += privateCentroid[j].numPointsAssigned;
#pragma omp atomic update
                    centroids[j].x += privateCentroid[j].x;
#pragma omp atomic update
                    centroids[j].y += privateCentroid[j].y;
                }

#pragma omp barrier

#pragma omp for nowait
                for (int j = 0; j < numCentroids; j++) {
                    centroids[j].x /= centroids[j].numPointsAssigned;
                    centroids[j].y /= centroids[j].numPointsAssigned;
                }
            }

        } while (change);
    }

    double end = omp_get_wtime();

    for (int i = 0; i < numCentroids; i++)
        printf("%f, %f\n", centroids[i].x, centroids[i].y);

    delete[](points);
    delete[](centroids);

    printf("execution time: %f", end - start);
}


point *pointGenerator(std::default_random_engine &generator, std::uniform_real_distribution<double> &random) {
    auto *points = new point[numPoints];
    for (int i = 0; i < numPoints; i++) {
        points[i].x = random(generator);
        points[i].y = random(generator);
        points[i].assignedCentroid = -1;
    }
    return points;
}

centroid *centroidGenerator(std::default_random_engine &generator, std::uniform_real_distribution<double> &random) {
    auto *centroids = new centroid[numCentroids];
    for (int i = 0; i < numCentroids; i++) {
        centroids[i].x = random(generator);
        centroids[i].y = random(generator);
        printf("%f, %f\n", centroids[i].x, centroids[i].y);
    }
    return centroids;
}

