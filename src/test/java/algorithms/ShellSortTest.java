package algorithms;

import metrics.PerformanceTracker;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class ShellSortTest {
    @Test
    public void testEmpty() {
        int[] a = new int[0];
        ShellSort.sort(a, ShellSort.GapSeq.KNUTH, new PerformanceTracker());
        assertArrayEquals(new int[0], a);
    }

    @Test
    public void testRandomMatchesArraysSort() {
        int[] a = {5, 3, 8, 3, 1, 9, 0};
        int[] expected = a.clone();
        java.util.Arrays.sort(expected);
        ShellSort.sort(a, ShellSort.GapSeq.SEDGEWICK, new PerformanceTracker());
        assertArrayEquals(expected, a);
    }
}