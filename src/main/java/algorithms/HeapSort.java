package algorithms;

import metrics.PerformanceTracker;

public class HeapSort {
    public static void sort(int[] a, PerformanceTracker t) {
        if (a == null || a.length <= 1) return;
        int n = a.length;
        for (int i = parent(n - 1); i >= 0; i--) siftDown(a, i, n, t);
        for (int end = n - 1; end > 0; end--) {
            if (a[0] != a[end]) swap(a, 0, end, t);
            siftDown(a, 0, end, t);
        }
    }

    private static void siftDown(int[] a, int root, int size, PerformanceTracker t) {
        int temp = a[root]; t.access(1);
        while (true) {
            int left = 2 * root + 1;
            if (left >= size) break;
            int right = left + 1;
            int swapIdx = left;
            t.access(2);
            t.compare(1);
            if (right < size) {
                t.access(2);
                t.compare(1);
                if (a[right] > a[left]) swapIdx = right;
            }
            t.access(1);
            t.compare(1);
            if (temp >= a[swapIdx]) break;
            a[root] = a[swapIdx]; t.access(1); t.swap();
            root = swapIdx;
        }
        a[root] = temp; t.access(1);
    }

    private static void swap(int[] a, int i, int j, PerformanceTracker t) {
        int tmp = a[i]; t.access(1);
        a[i] = a[j]; t.access(1);
        a[j] = tmp; t.access(1);
        t.swap();
    }

    private static int parent(int i) { return (i - 1) / 2; }
}