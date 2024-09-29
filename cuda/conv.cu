#include "common.h"

__global__ void im2col_kernel(const float *__restrict__ im, float *__restrict__ col, int N, int C, int IH, int IW,
                              int KH, int KW, int OH, int OW, int PH, int PW) {
    const int numel = N * OH * OW * KH * KW * C;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < numel; idx += blockDim.x * gridDim.x) {
        const int c = idx % C;
        const int kw = (idx / C) % KW;
        const int kh = (idx / (C * KW)) % KH;
        const int ow = (idx / (C * KW * KH)) % OW;
        const int oh = (idx / (C * KW * KH * OW)) % OH;
        const int n = (idx / (C * KW * KH * OW * OH)) % N;

        // comply with cudnn convention: top left pixel is multiplied by bottom right filter element
        const int ih = oh - PH + (KH - 1 - kh);
        const int iw = ow - PW + (KW - 1 - kw);
        if (0 <= ih && ih < IH && 0 <= iw && iw < IW) {
            const int im_idx = ((n * IH + ih) * IW + iw) * C + c;
            col[idx] = im[im_idx];
        } else {
            col[idx] = 0.f;
        }
    }
}

cudaError_t im2col_cuda(const float *im, float *col, int N, int IC, int IH, int IW, int KH, int KW, int OH, int OW,
                        int PH, int PW) {
    // im: [N, IH, IW, IC]
    // col: [N, OH, OW, KH, KW, IC]
    const int block_size = 128;
    const int grid_size = (N * OH * OW * KH * KW * IC + block_size - 1) / block_size;
    im2col_kernel<<<grid_size, block_size>>>(im, col, N, IC, IH, IW, KH, KW, OH, OW, PH, PW);
    return cudaGetLastError();
}

cublasStatus_t conv_cublas(cublasHandle_t handle, const float *x, const float *w, float *y, void *workspace,
                           size_t workspace_size, int N, int IC, int OC, int IH, int IW, int KH, int KW, int OH, int OW,
                           int PH, int PW) {
    const size_t workspace_size_requested = N * OH * OW * KH * KW * IC * sizeof(float);
    CHECK(workspace_size_requested <= workspace_size) << workspace_size_requested << " vs " << workspace_size;
    float *col = (float *)workspace;
    CHECK_CUDA(im2col_cuda(x, col, N, IC, IH, IW, KH, KW, OH, OW, PH, PW));

    const float alpha = 1.f;
    const float beta = 0.f;
    cublasStatus_t status = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, OC, N * OH * OW, KH * KW * IC, &alpha, w,
                                        KH * KW * IC, col, KH * KW * IC, &beta, y, OC);

#if 0
    float *h_col, *h_im, *h_w, *h_y;
    CHECK_CUDA(cudaMallocHost(&h_col, workspace_size_requested));
    CHECK_CUDA(cudaMemcpy(h_col, col, workspace_size_requested, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMallocHost(&h_im, N * IH * IW * IC * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(h_im, x, N * IH * IW * IC * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMallocHost(&h_w, OC * KH * KW * IC * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(h_w, w, OC * KH * KW * IC * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMallocHost(&h_y, N * OH * OW * OC * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(h_y, y, N * OH * OW * OC * sizeof(float), cudaMemcpyDeviceToHost));

    printf("im:\n");
    for (int i = 0; i < N * IH; i++) {
        for (int iw = 0; iw < IW; iw++) {
            for (int c = 0; c < IC; c++) {
                printf("%s%.3f%s", (c == 0) ? "[" : ", ", h_im[(i * IW + iw) * IC + c], (c == IC - 1) ? "] " : "");
            }
        }
        printf("\n");
    }

    printf("col:\n");
    for (int i = 0; i < N * OH * OW; i++) {
        for (int j = 0; j < KH * KW * IC; j++) {
            printf("%.3f, ", h_col[i * KH * KW * IC + j]);
        }
        printf("\n");
    }

    printf("filter:\n");
    for (int i = 0; i < OC; i++) {
        for (int j = 0; j < KH * KW * IC; j++) {
            printf("%.3f, ", h_w[i * KH * KW * IC + j]);
        }
        printf("\n");
    }

    printf("cublas-output:\n");
    for (int oh = 0; oh < N * OH; oh++) {
        for (int ow = 0; ow < OW; ow++) {
            for (int c = 0; c < OC; c++) {
                printf("%s%.3f%s", (c == 0) ? "[" : ", ", h_y[(oh * OW + ow) * OC + c], (c == OC - 1) ? "] " : "");
            }
        }
        printf("\n");
    }

    CHECK_CUDA(cudaFreeHost(h_col));
    CHECK_CUDA(cudaFreeHost(h_im));
    CHECK_CUDA(cudaFreeHost(h_w));
    CHECK_CUDA(cudaFreeHost(h_y));
#endif

    return status;
}

cudnnStatus_t conv_cudnn(cudnnHandle_t handle, const float *x, const float *w, float *y, void *workspace,
                         size_t workspace_size, int N, int IC, int OC, int IH, int IW, int KH, int KW, int OH, int OW,
                         int PH, int PW) {
    cudnnTensorDescriptor_t x_desc, y_desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&x_desc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, N, IC, IH, IW));

    CHECK_CUDNN(cudnnCreateTensorDescriptor(&y_desc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(y_desc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, N, OC, OH, OW));

    cudnnFilterDescriptor_t w_desc;
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&w_desc));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NHWC, OC, IC, KH, KW));

    const int SH = 1; // stride h
    const int SW = 1; // stride w
    const int DH = 1; // dilation h
    const int DW = 1; // dilation w
    cudnnConvolutionDescriptor_t conv_desc;
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));
    CHECK_CUDNN(
        cudnnSetConvolution2dDescriptor(conv_desc, PH, PW, SH, SW, DH, DW, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT));

    int out_n, out_c, out_h, out_w;
    CHECK_CUDNN(cudnnGetConvolution2dForwardOutputDim(conv_desc, x_desc, w_desc, &out_n, &out_c, &out_h, &out_w));
    CHECK(out_n == N && out_c == OC && out_h == OH && out_w == OW);

    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;

    size_t workspace_size_requested;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(handle, x_desc, w_desc, conv_desc, y_desc, algo,
                                                        &workspace_size_requested));
    CHECK(workspace_size_requested <= workspace_size) << workspace_size_requested << " vs " << workspace_size;

    const float alpha = 1.f;
    const float beta = 0.f;
    cudnnStatus_t status = cudnnConvolutionForward(handle, &alpha, x_desc, x, w_desc, w, conv_desc, algo, workspace,
                                                   workspace_size, &beta, y_desc, y);

    CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(conv_desc));
    CHECK_CUDNN(cudnnDestroyFilterDescriptor(w_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(x_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(y_desc));

#if 0
    float *h_y;
    CHECK_CUDA(cudaMallocHost(&h_y, N * OH * OW * OC * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(h_y, y, N * OH * OW * OC * sizeof(float), cudaMemcpyDeviceToHost));

    printf("cudnn-output:\n");
    for (int oh = 0; oh < N * OH; oh++) {
        for (int ow = 0; ow < OW; ow++) {
            for (int c = 0; c < OC; c++) {
                printf("%s%.3f%s", (c == 0) ? "[" : ", ", h_y[(oh * OW + ow) * OC + c], (c == OC - 1) ? "] " : "");
            }
        }
        printf("\n");
    }

    CHECK_CUDA(cudaFreeHost(h_y));
#endif

    return status;
}

int main() {
    cudnnHandle_t cudnn_handle;
    CHECK_CUDNN(cudnnCreate(&cudnn_handle));

    cublasHandle_t cublas_handle;
    CHECK_CUBLAS(cublasCreate(&cublas_handle));

    const int N = 2;
    const int IC = 4;
    const int OC = 8;
    const int IH = 128;
    const int IW = 256;
    const int KH = 3; // kernel h
    const int KW = 3; // kernel w
    const int PH = 1; // pad h
    const int PW = 1; // pad w
    const int OH = (IH + 2 * PH - KH + 1);
    const int OW = (IW + 2 * PW - KW + 1);

    printf("input  [%d, %d, %d, %d]\n", N, IH, IW, IC);
    printf("filter [%d, %d, %d, %d]\n", OC, KH, KW, IC);
    printf("output [%d, %d, %d, %d]\n", N, OH, OW, OC);

    float *h_x, *h_w, *h_y_actual, *h_y_expect;
    CHECK_CUDA(cudaMallocHost(&h_x, N * IH * IW * IC * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&h_w, OC * KH * KW * IC * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&h_y_actual, N * OH * OW * OC * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&h_y_expect, N * OH * OW * OC * sizeof(float)));

    float *d_x, *d_w, *d_y;
    CHECK_CUDA(cudaMalloc(&d_x, N * IH * IW * IC * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_w, OC * KH * KW * IC * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, N * OH * OW * OC * sizeof(float)));

    for (int i = 0; i < N * IH * IW * IC; i++) {
        h_x[i] = uniform(-0.5, 0.5);
    }
    for (int i = 0; i < OC * KH * KW * IC; i++) {
        h_w[i] = uniform(-0.5, 0.5);
    }
    CHECK_CUDA(cudaMemcpyAsync(d_x, h_x, N * IH * IW * IC * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpyAsync(d_w, h_w, OC * KH * KW * IC * sizeof(float), cudaMemcpyHostToDevice));

    void *workspace;
    size_t workspace_size = 16 * 1024 * 1024;
    CHECK_CUDA(cudaMalloc(&workspace, workspace_size));

    CHECK_CUDA(cudaMemsetAsync(d_y, 0, N * OH * OW * OC * sizeof(float)));
    CHECK_CUBLAS(conv_cublas(cublas_handle, d_x, d_w, d_y, workspace, workspace_size, N, IC, OC, IH, IW, KH, KW, OH, OW,
                             PH, PW));
    CHECK_CUDA(cudaMemcpy(h_y_actual, d_y, N * OH * OW * OC * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaMemsetAsync(d_y, 0, N * OH * OW * OC * sizeof(float)));
    CHECK_CUDNN(
        conv_cudnn(cudnn_handle, d_x, d_w, d_y, workspace, workspace_size, N, IC, OC, IH, IW, KH, KW, OH, OW, PH, PW));
    CHECK_CUDA(cudaMemcpy(h_y_expect, d_y, N * OH * OW * OC * sizeof(float), cudaMemcpyDeviceToHost));

    check_is_close(h_y_expect, h_y_actual, N * OH * OW * OC, 1e-3f);

    {
        const float elapsed = timeit(
            [&] {
                CHECK_CUBLAS(conv_cublas(cublas_handle, d_x, d_w, d_y, workspace, workspace_size, N, IC, OC, IH, IW, KH,
                                         KW, OH, OW, PH, PW));
            },
            2, 10);
        printf("[im2col+cublas] elapsed %.3f us\n", elapsed * 1e6f);
    }
    {
        const float elapsed = timeit(
            [&] {
                CHECK_CUDNN(conv_cudnn(cudnn_handle, d_x, d_w, d_y, workspace, workspace_size, N, IC, OC, IH, IW, KH,
                                       KW, OH, OW, PH, PW));
            },
            2, 10);
        printf("[cudnn-conv] elapsed %.3f us\n", elapsed * 1e6f);
    }

    CHECK_CUDA(cudaFree(workspace));
    CHECK_CUDA(cudaFreeHost(h_x));
    CHECK_CUDA(cudaFreeHost(h_w));
    CHECK_CUDA(cudaFreeHost(h_y_expect));
    CHECK_CUDA(cudaFreeHost(h_y_actual));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_w));
    CHECK_CUDA(cudaFree(d_y));

    CHECK_CUBLAS(cublasDestroy(cublas_handle));
    CHECK_CUDNN(cudnnDestroy(cudnn_handle));

    return 0;
}