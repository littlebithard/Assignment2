package cli;

import algorithms.*;
import metrics.PerformanceTracker;

import java.util.Random;

public class BenchmarkRunner {
    public static void main(String[] args) {
        int n = 10000;
        String alg = "shell";
        String input = "random";
        ShellSort.GapSeq seq = ShellSort.GapSeq.KNUTH;
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "-n": n = Integer.parseInt(args[++i]); break;
                case "-alg": alg = args[++i]; break;
                case "-input": input = args[++i]; break;
                case "-seq": seq = ShellSort.GapSeq.valueOf(args[++i].toUpperCase()); break;
            }
        }
        int[] arr = generate(n, input);
        PerformanceTracker t = new PerformanceTracker();
        t.start();
        if ("shell".equalsIgnoreCase(alg)) ShellSort.sort(arr, seq, t);
        else if ("heap".equalsIgnoreCase(alg)) HeapSort.sort(arr, t);
        else { System.err.println("Unknown alg"); return; }
        t.stop();
        System.out.println("n,input,ns,comparisons,swaps,time_ms");
        System.out.println(t.toCsvRow(n, input));
    }

    private static int[] generate(int n, String type) {
        Random r = new Random(42);
        int[] a = new int[n];
        switch (type) {
            case "random": for (int i = 0; i < n; i++) a[i] = r.nextInt(); break;
            case "sorted": for (int i = 0; i < n; i++) a[i] = i; break;
            case "reversed": for (int i = 0; i < n; i++) a[i] = n - i; break;
            case "nearly":
                for (int i = 0; i < n; i++) a[i] = i;
                for (int i = 0; i < n / 20; i++) {
                    int x = r.nextInt(n), y = r.nextInt(n);
                    int tmp = a[x]; a[x] = a[y]; a[y] = tmp;
                }
                break;
            default: for (int i = 0; i < n; i++) a[i] = r.nextInt();
        }
        return a;
    }
}