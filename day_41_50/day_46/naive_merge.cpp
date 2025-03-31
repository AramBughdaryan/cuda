void merge(int *A, int n, int *B, int m, int *C) {
    int i = 0, j = 0, k = 0;
    while ((i < n) && (j < m)) {
        if (A[i] < B[j]) {
            C[k++] = A[i++];
        } else {
            C[k++] = B[j++];
        }
    }
    while (i < n) {
        C[k++] = A[i++];
    }
    while (j < m) {
        C[k++] = B[j++];
    }
}