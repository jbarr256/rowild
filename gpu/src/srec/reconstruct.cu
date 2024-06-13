#include "hip/hip_runtime.h"
/*
 * MIT License
 *
 * Copyright (c) 2023 Carnegie Mellon University
 *
 * This file is part of RoWild.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <algorithm>
#include <hipblas/hipblas.h>
#include <hip/hip_runtime.h>
#include <hipsolver/hipsolver.h>
#include <iostream>
#include <math.h>
#include <vector>

#include "reconstruct.h"
#include "rowild_utils.h"
#include "transforms.h"

namespace reconstruct {

double **poseToTransformation(double *delta) {
    double *w = delta;
    double *u = delta + 3;

    double **T = new double *[4];
    for (int i = 0; i < 4; i++) {
        T[i] = new double[4];
    }

    T[0][0] = cos(w[2]) * cos(w[1]);
    T[0][1] = -sin(w[2]) * cos(w[0]) + cos(w[2]) * sin(w[1]) * sin(w[0]);
    T[0][2] = sin(w[2]) * sin(w[0]) + cos(w[2]) * sin(w[1]) * cos(w[1]);
    T[1][0] = sin(w[2]) * cos(w[1]);
    T[1][1] = cos(w[2]) * cos(w[0]) + sin(w[2]) * sin(w[1]) * sin(w[0]);
    T[1][2] = -cos(w[2]) * sin(w[0]) + sin(w[2]) * sin(w[1]) * cos(w[0]);
    T[2][0] = -sin(w[1]);
    T[2][1] = cos(w[1]) * sin(w[0]);
    T[2][2] = cos(w[1]) * cos(w[0]);

    T[0][3] = u[0];
    T[1][3] = u[1];
    T[2][3] = u[2];

    T[3][0] = T[3][1] = T[3][2] = 0.0;
    T[3][3] = 1.0;

    return T;
}

__global__ void transformAndProjectKernel(
    double *d_points, int numPoints, double *d_T, double fx, double fy,
    double cx, double cy, int *d_iuv, int *d_newNumPoints, double *d_vertexMap,
    double *d_normalMap, int h, int w, double *d_A, double *d_b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < numPoints) {
        double tPoint[3];
        int idx = 3 * i;
        tPoint[0] = (d_T[0] * d_points[idx]) + (d_T[1] * d_points[idx + 1]) +
                    (d_T[2] * d_points[idx + 2]) + d_T[3];
        tPoint[1] = (d_T[4] * d_points[idx]) + (d_T[5] * d_points[idx + 1]) +
                    (d_T[6] * d_points[idx + 2]) + d_T[7];
        tPoint[2] = (d_T[8] * d_points[idx]) + (d_T[9] * d_points[idx + 1]) +
                    (d_T[10] * d_points[idx + 2]) + d_T[11];

        int tU = round(fx * tPoint[0] / tPoint[2] + cx);
        int tV = round(fy * tPoint[1] / tPoint[2] + cy);
        double tD = tPoint[2];

        if ((tV >= 0) && (tV < h) && (tU >= 0) && (tU < w) && (tD >= 0)) {
            int mapIdx = (tV * w + tU) * 3;

            double dist = sqrt((tPoint[0] - d_vertexMap[mapIdx]) *
                                   (tPoint[0] - d_vertexMap[mapIdx]) +
                               (tPoint[1] - d_vertexMap[mapIdx + 1]) *
                                   (tPoint[1] - d_vertexMap[mapIdx + 1]) +
                               (tPoint[2] - d_vertexMap[mapIdx + 2]) *
                                   (tPoint[2] - d_vertexMap[mapIdx + 2]));

            if (dist < 0.07) {
                int newIdx = 3 * atomicAdd(d_newNumPoints, 1);
                d_iuv[newIdx] = i;
                d_iuv[newIdx + 1] = tU;
                d_iuv[newIdx + 2] = tV;
            }
        }
    }

    __syncthreads();

    if (i < *d_newNumPoints) {

        int idx = d_iuv[3 * i];
        int tU = d_iuv[3 * i + 1];
        int tV = d_iuv[3 * i + 2];

        int pointIdx = 3 * idx;
        int mapIdx = (tV * w + tU) * 3;

        double theSrcPoint[3] = {d_points[pointIdx], d_points[pointIdx + 1],
                                 d_points[pointIdx + 2]};
        double theTarPoint[3] = {d_vertexMap[mapIdx], d_vertexMap[mapIdx + 1],
                                 d_vertexMap[mapIdx + 2]};
        double theTarNormal[3] = {d_normalMap[mapIdx], d_normalMap[mapIdx + 1],
                                  d_normalMap[mapIdx + 2]};

        double sPointPrime[3] = {
            (d_T[0] * theSrcPoint[0]) + (d_T[1] * theSrcPoint[1]) +
                (d_T[2] * theSrcPoint[2]) + d_T[3],
            (d_T[4] * theSrcPoint[0]) + (d_T[5] * theSrcPoint[1]) +
                (d_T[6] * theSrcPoint[2]) + d_T[7],
            (d_T[8] * theSrcPoint[0]) + (d_T[9] * theSrcPoint[1]) +
                (d_T[10] * theSrcPoint[2]) + d_T[11]};

        int sIdx = 6 * i;
        d_A[sIdx + 0] =
            sPointPrime[1] * theTarNormal[2] - sPointPrime[2] * theTarNormal[1];
        d_A[sIdx + 1] =
            sPointPrime[2] * theTarNormal[0] - sPointPrime[0] * theTarNormal[2];
        d_A[sIdx + 2] =
            sPointPrime[0] * theTarNormal[1] - sPointPrime[1] * theTarNormal[0];
        d_A[sIdx + 3] = theTarNormal[0];
        d_A[sIdx + 4] = theTarNormal[1];
        d_A[sIdx + 5] = theTarNormal[2];
        d_b[i] = (theTarNormal[0] * (theTarPoint[0] - sPointPrime[0])) +
                 (theTarNormal[1] * (theTarPoint[1] - sPointPrime[1])) +
                 (theTarNormal[2] * (theTarPoint[2] - sPointPrime[2]));
    }
}

double **icp(Map *m, Frame *f, double **intrinsic, double **T) {
    int h = f->getHeight();
    int w = f->getWidth();
    double fx = intrinsic[0][0];
    double fy = intrinsic[1][1];
    double cx = intrinsic[0][2];
    double cy = intrinsic[1][2];

    hipsolverHandle_t cusolverH;
    hipblasHandle_t cublasH;
    hipsolverDnCreate(&cusolverH);
    hipblasCreate(&cublasH);

    int *d_iuv, *d_newNumPoints;
    double *d_points, *d_vertexMap, *d_normalMap, *d_T;
    double *d_A, *d_b, *d_delta, *d_tau, *d_work;
    int *devInfo;
    int lwork = 0;

    constexpr int vars = 3 + 3;
    int numPoints = m->getNumPoints();
    hipMalloc(&d_iuv, numPoints * 3 * sizeof(int));
    hipMalloc(&d_newNumPoints, sizeof(int));
    hipMalloc(&d_points, numPoints * 3 * sizeof(double));
    hipMalloc(&d_vertexMap, h * w * 3 * sizeof(double));
    hipMalloc(&d_normalMap, h * w * 3 * sizeof(double));
    hipMalloc(&d_T, 16 * sizeof(double));
    hipMalloc(&d_A, numPoints * vars * sizeof(double));
    hipMalloc(&d_b, numPoints * sizeof(double));
    hipMalloc(&d_delta, vars * sizeof(double));

    hipMalloc((void **)&d_tau, sizeof(double) * vars);
    hipMalloc((void **)&d_work, sizeof(double) * lwork);
    hipMalloc((void **)&devInfo, sizeof(int));

    hipsolverDnDgeqrf_bufferSize(cusolverH, numPoints, vars, d_A, numPoints,
                                &lwork);

    constexpr int threadsPerBlock = 256;

    for (int iter = 0; iter < 10; iter++) {
        hipMemcpy(d_points, m->getPoints(), numPoints * 3 * sizeof(double),
                   hipMemcpyHostToDevice);
        hipMemcpy(d_vertexMap, f->getVertexMap(), h * w * 3 * sizeof(double),
                   hipMemcpyHostToDevice);
        hipMemcpy(d_normalMap, f->getNormalMap(), h * w * 3 * sizeof(double),
                   hipMemcpyHostToDevice);
        hipMemcpy(d_T, T, 16 * sizeof(double), hipMemcpyHostToDevice);
        hipMemset(d_newNumPoints, 0, sizeof(int));

        int blocksPerGridTransform =
            (numPoints + threadsPerBlock - 1) / threadsPerBlock;
        transformAndProjectKernel<<<blocksPerGridTransform, threadsPerBlock>>>(
            d_points, numPoints, d_T, fx, fy, cx, cy, d_iuv, d_newNumPoints,
            d_vertexMap, d_normalMap, h, w, d_A, d_b);

        // QR factorization
        // QR provides numerical stability over methods like normal
        // equations, ensuring accurate solutions in each ICP iteration,
        // which is vital for the convergence of the ICP algorithm.
        hipsolverDnDgeqrf(cusolverH, numPoints, vars, d_A, numPoints, d_tau,
                         d_work, lwork, devInfo);

        hipMemcpy(d_delta, d_b, sizeof(double) * numPoints,
                   hipMemcpyDeviceToDevice);
        hipsolverDnDormqr(cusolverH, HIPBLAS_SIDE_LEFT, HIPBLAS_OP_T, numPoints, 1,
                         vars, d_A, numPoints, d_tau, d_delta, numPoints,
                         d_work, lwork, devInfo);
        double one = 1;
        hipblasDtrsm(cublasH, HIPBLAS_SIDE_LEFT, HIPBLAS_FILL_MODE_UPPER,
                    HIPBLAS_OP_N, HIPBLAS_DIAG_NON_UNIT, vars, 1, &one, d_A,
                    numPoints, d_delta, numPoints);

        double *delta = new double[vars];
        hipMemcpy(delta, d_delta, sizeof(double) * vars,
                   hipMemcpyDeviceToHost);

        double **update = poseToTransformation(delta);
        double **oldT = new double *[4];

        for (int i = 0; i < 4; i++) {
            oldT[i] = new double[4]{T[i][0], T[i][1], T[i][2], T[i][3]};
        }

        matrixMultiplication<double, 4, 4, 4>(update, oldT, T);

        delete[] oldT[0];
        delete[] oldT[1];
        delete[] oldT[2];
        delete[] oldT[3];
        delete[] oldT;
        delete[] delta;
    }

    hipFree(d_iuv);
    hipFree(d_newNumPoints);
    hipFree(d_points);
    hipFree(d_vertexMap);
    hipFree(d_normalMap);
    hipFree(d_T);
    hipFree(d_A);
    hipFree(d_b);
    hipFree(d_delta);
    hipFree(d_tau);
    hipFree(devInfo);

    hipblasDestroy(cublasH);
    hipsolverDnDestroy(cusolverH);

    return T;
}

} // namespace reconstruct
