package algorithms;

import metrics.PerformanceTracker;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class HeapSortTest {
    @Test public void testEmpty() {
        int[] a = new int[0];
        HeapSort.sort(a, new PerformanceTracker());
        assertArrayEquals(new int[0], a);
    }
    @Test public void testRandomMatchesArraySort() {
        int[] a = {8, 5, 3, 9, 1};
        int[] expected = a.clone();
        java.util.Arrays.sort(expected);
        HeapSort.sort(a, new PerformanceTracker());
        assertArrayEquals(expected, a);
    }
}