#include <vector>
#include <iostream>
#include <limits>
#include <cstring>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

using namespace std;

extern "C" {
EXPORT void align_sequences(const float* logProbs, const int64_t* targets, int64_t* paths, float* scores, int batch_size, int T,
    int num_classes, int L, int64_t blank){
    if (batch_size <= 0 || T <= 0 || num_classes <= 0 || L <= 0) {
        cerr << "Invalid dimensions provided to align_sequences function." << endl;
        return;
    }
    vector<vector<vector<float>>> logProbsVec(batch_size, vector<vector<float>>(T, vector<float>(num_classes)));
    vector<vector<int64_t>> targetsVec(batch_size, vector<int64_t>(L)), pathsVec(batch_size, vector<int64_t>(T));
    for (int b = 0; b < batch_size; ++b) {
        for (int t = 0; t < T; ++t) {
            for (int c = 0; c < num_classes; ++c) logProbsVec[b][t][c] = logProbs[b * T * num_classes + t * num_classes + c];
        }
        for (int l = 0; l < L; ++l) targetsVec[b][l] = targets[b * L + l];
    }
    // CTC alignment logic
    const float kNegInfinity = -numeric_limits<float>::infinity();
    const int batchIndex = 0, S = 2 * L + 1;
    vector<float> alphas(2 * S, kNegInfinity);
    vector<bool> backPtrBit0((S + 1) * (T - L), false), backPtrBit1((S + 1) * (T - L), false);
    vector<unsigned long long> backPtr_offset(T - 1), backPtr_seek(T - 1);
    int R = 0;
    for (int i = 1; i < L; i++)
        if (targetsVec[batchIndex][i] == targetsVec[batchIndex][i - 1]) ++R;
    if (T < L + R) { throw runtime_error("targets length is too long for CTC.");}
    int start = (T - (L + R) > 0) ? 0 : 1;
    int end = (S == 1) ? 1 : 2;
    for (int i = start; i < end; i++) {
        int labelIdx = (i % 2 == 0) ? blank : targetsVec[batchIndex][i / 2];
        alphas[i] = logProbsVec[batchIndex][0][labelIdx];
    }
    unsigned long long seek = 0;
    for (int t = 1; t < T; t++) {
        if (T - t <= L + R) {
            if ((start % 2 == 1) && targetsVec[batchIndex][start / 2] != targetsVec[batchIndex][start / 2 + 1]) {
                start += 1;
            } start += 1;
        }
        if (t <= L + R) {
            if (end % 2 == 0 && end < 2 * L && targetsVec[batchIndex][end / 2 - 1] != targetsVec[batchIndex][end / 2]) {
                end += 1;
            } end += 1;
        }
        int startloop = start;
        int curIdxOffset = t % 2;
        int prevIdxOffset = (t - 1) % 2;
        fill(alphas.begin() + curIdxOffset * S, alphas.begin() + (curIdxOffset + 1) * S, kNegInfinity);
        backPtr_seek[t - 1] = seek;
        backPtr_offset[t - 1] = start;
        if (start == 0) {
            alphas[curIdxOffset * S] = alphas[prevIdxOffset * S] + logProbsVec[batchIndex][t][blank];
            startloop += 1, seek += 1;
        }
        for (int i = startloop; i < end; i++) {
            float x0 = alphas[prevIdxOffset * S + i], x1 = alphas[prevIdxOffset * S + i - 1], x2 = kNegInfinity;
            int labelIdx = (i % 2 == 0) ? blank : targetsVec[batchIndex][i / 2];
            if (i % 2 != 0 && i != 1 && targetsVec[batchIndex][i / 2] != targetsVec[batchIndex][i / 2 - 1])
                x2 = alphas[prevIdxOffset * S + i - 2];
            float result = 0.0;
            if (x2 > x1 && x2 > x0) {
                result = x2, backPtrBit1[seek + i - startloop] = true;
            } else if (x1 > x0 && x1 > x2) {
                result = x1, backPtrBit0[seek + i - startloop] = true;
            } else result = x0;
            alphas[curIdxOffset * S + i] = result + logProbsVec[batchIndex][t][labelIdx];
        }
        seek += (end - startloop);
    }
    int idx1 = (T - 1) % 2, ltrIdx = alphas[idx1 * S + S - 1] > alphas[idx1 * S + S - 2] ? S - 1 : S - 2;
    for (int t = T - 1; t > -1; t--) {
        int lbl_idx = ltrIdx % 2 == 0 ? blank : targetsVec[batchIndex][ltrIdx / 2];
        pathsVec[batchIndex][t] = lbl_idx;
        int t_minus_one = t - 1 >= 0 ? t - 1 : 0;
        int backPtr_idx = backPtr_seek[t_minus_one] + ltrIdx - backPtr_offset[t_minus_one];
        ltrIdx -= (backPtrBit1[backPtr_idx] << 1) | backPtrBit0[backPtr_idx];
    }
    for (int b = 0; b < batch_size; ++b) {
        for (int t = 0; t < T; ++t) {
            paths[b * T + t] = pathsVec[b][t];
            scores[b * T + t] = logProbs[b * T * num_classes + t * num_classes + pathsVec[b][t]];
        }
    }
}
}
