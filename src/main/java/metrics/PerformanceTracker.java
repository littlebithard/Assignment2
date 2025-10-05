package metrics;

public class PerformanceTracker {
    private long startTime;
    public long comparisons = 0;
    public long swaps = 0;
    public long arrayAccesses = 0;
    public long timeNs = 0;

    public void start() { startTime = System.nanoTime(); }
    public void stop() { timeNs = System.nanoTime() - startTime; }

    public void access() { arrayAccesses++; }
    public void access(int count) { arrayAccesses += count; }
    public void compare(long count) { comparisons += count; }
    public void swap() { swaps++; arrayAccesses += 4; }

    @Override
    public String toString() {
        return String.format("Time: %d ns, Accesses: %d, Comparisons: %d, Swaps: %d",
                timeNs, arrayAccesses, comparisons, swaps);
    }

    public String toCsvRow(int n, String inputType) {
        return String.format("%d,%s,%d,%d,%d,%.6f",
                n, inputType, timeNs, comparisons, swaps, timeNs / 1e6);
    }
}