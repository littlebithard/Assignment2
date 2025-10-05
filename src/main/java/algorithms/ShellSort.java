package algorithms;

import metrics.PerformanceTracker;
import java.util.ArrayList;
import java.util.List;

public class ShellSort {
    public enum GapSeq { SHELL, KNUTH, SEDGEWICK }

    public static void sort(int[] a, GapSeq seq, PerformanceTracker t) {
        if (a == null) return;
        int n = a.length;
        List<Integer> gaps = generateGaps(n, seq);
        for (int gap : gaps) {
            for (int i = gap; i < n; i++) {
                int temp = a[i]; t.access(1);
                int j = i;
                while (j >= gap) {
                    t.compare(1);
                    t.access(1);
                    if (a[j - gap] <= temp) break;
                    a[j] = a[j - gap]; t.access(2);
                    j -= gap;
                    t.swap();
                }
                a[j] = temp; t.access(1);
            }
        }
    }

    private static List<Integer> generateGaps(int n, GapSeq seq) {
        List<Integer> gaps = new ArrayList<>();
        if (n <= 1) { return gaps; }
        switch (seq) {
            case SHELL:
                for (int g = n / 2; g > 0; g /= 2) gaps.add(g);
                break;
            case KNUTH:
                int g = 1;
                while (g < n) g = 3 * g + 1;
                while (g > 0) { gaps.add(g); g = (g - 1) / 3; }
                break;
            case SEDGEWICK:
                List<Integer> tmp = new ArrayList<>();
                int k = 0;
                while (true) {
                    int gap = (int)(Math.pow(4, k) + 3 * Math.pow(2, k - 1) + 1);
                    if (gap >= n) break;
                    tmp.add(gap);
                    k++;
                }
                if (tmp.isEmpty() || tmp.get(tmp.size() - 1) != 1) tmp.add(1);
                for (int i = tmp.size() - 1; i >= 0; i--) gaps.add(tmp.get(i));
                break;
        }
        return gaps;
    }
}