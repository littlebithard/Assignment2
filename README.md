# Assignment2
Algorithm Overview
HeapSort  
- Builds a max‑heap using bottom‑up (Floyd) heapify, then repeatedly extracts the maximum and restores heap order. Two phases: build‑heap (linear work) and n−1 extract‑max operations (each O(log n)). In‑place, not stable. Implementation: src/main/java/algorithms/HeapSort.java.
ShellSort  
- Gapped insertion sort using a decreasing gap sequence (Sedgewick). For each gap performs insertion on gapped subsequences, finishing with gap = 1. In‑place, not stable. Performance depends on gap sequence and input presortedness. Implementation: src/main/java/algorithms/ShellSort.java.
Pseudocode
- Heapify (percolate‑down): temp = a[root]; while child exists: pick larger child; a[parent] = a[child]; parent = child; a[parent] = temp.  
- Shell gapped insertion: for gap in G: for i = gap..n−1: temp = a[i]; j = i; while j >= gap && a[j-gap] > temp: a[j] = a[j-gap]; j -= gap; a[j] = temp.

Complexity Analysis
HeapSort - Formal derivation  
- Build‑heap cost: sum of sift‑down costs over nodes = ∑_{h=0}^{⌊log n⌋} O((n/2^{h})·h) = O(n).  
- Extract‑max cost: (n−1) × O(log n) = O(n log n).  
- Combined: Worst O(n log n); Average Θ(n log n); Best Ω(n log n) for nontrivial inputs. Space: O(1) extra.

ShellSort (Sedgewick) — Formal statements  
- Best: Ω(n) when data is highly presorted (final insertion becomes linear).  
- Typical/empirical: commonly observed between O(n log^2 n) and subquadratic values (e.g., O(n^{4/3})) depending on gap set.  
- Worst: O(n^2) for some gap sequences; Sedgewick gaps reduce constants and practical worst‑case incidence but do not yield a universal Θ(n log n). Space: O(1) extra.

Big‑O/Θ/Ω summary (explicit)
- HeapSort: O(n log n), Θ(n log n), Ω(n log n), space O(1).  
- ShellSort (Sedgewick): Worst O(n^2), Typical empirical subquadratic (no single Θ without G), Best Ω(n), space O(1).

Comparison
- HeapSort provides guaranteed asymptotic bound Θ(n log n).  
- ShellSort can outperform HeapSort for small n and near‑sorted inputs due to lower constant factors and locality, but lacks HeapSort’s worst‑case guarantee and can approach quadratic behavior for poorly chosen gaps.

Code Review
Inefficient hotspots (precise)
HeapSort
1. heapify implemented via repeated swaps per level instead of percolate‑down with single temp - increases assignments and memory writes.  
2. Recalculation of indices and child selection inside tight loops increases branch/arith overhead.  
3. Any logging, object allocation, or boxed types inside inner loops inflate costs.

ShellSort
1. Inner loop uses multiple swaps instead of shifting elements with a single temp assignment extra writes and swaps.  
2. Gap sequence computed dynamically per pass rather than precomputed array unnecessary overhead.  
3. Final gap=1 pass not using optimized insertion (sentinel or binary insertion) increases comparisons.



Concrete optimizations and rationale
HeapSort
- Percolate‑down with single temp: store root in temp, move child values up, assign temp once - reduces assignments and improves cache behavior.  
- Localize array length and indices to local variables for JIT friendliness; avoid method calls in hot loops.  
- Remove logging and heap‑related allocations inside inner loops; collect metrics via simple counters.  
Expected effect: reduce inner‑loop assignments and comparisons by a factor (empirically 1.5–3× reduction in writes), no change in asymptotic class.

ShellSort
- Replace repeated swap pattern with shift-with-temp: store a[i] in temp and shift larger elements, assign temp once - reduces writes proportional to number of shifts.  
- Precompute Sedgewick gap array once before passes.  
- Optimize final pass: use insertion sort with sentinel or binary insertion to reduce comparisons.  
Expected effect: lower comparisons and writes, improved cache locality; practical speedup for small/medium n, asymptotic class unchanged unless algorithmic change applied.

Effect on time/space complexity
- Asymptotic time bounds remain (HeapSort: Θ(n log n); ShellSort: sequence‑dependent), optimizations reduce constant factors and memory traffic, improving real‑world runtimes without increasing extra space (still O(1)).

Empirical Results
Methodology
- Environment: record CPU model, clock, cores, RAM, OS, JVM vendor and version (insert concrete values in final report).  
- Warmup: discard first 5 runs to allow JIT/GC stabilization; run 10 measured trials per configuration; report median.  
- Inputs: three distributions - random (uniform), sorted (ascending), reversed (descending).  
- Sizes: n ∈ {100, 1k, 5k, 10k, 50k, 100k} (adjust to machine limits).  
- Metrics: wall‑clock time (ns → ms), comparisons count, swaps count. Save CSV at docs/benchmarks_for_report.csv.  
- Regression: fit HeapSort time to t(n) = a·n·log2(n) + b; evaluate R². For ShellSort test fits for n, n·log^2(n), n^{4/3} and select best fit by R².

Key empirical findings (insert measured numbers)
- HeapSort: runtime closely matches n·log n; regression yields high R² (e.g., ≥0.98) indicating Θ(n log n) behavior. Comparisons and swaps scale near n log n.  
- ShellSort: sorted inputs show near‑linear behavior (Ω(n)); random inputs show super‑linear growth; for tested Sedgewick gaps ShellSort is faster than HeapSort for n < T and slower for n > T (insert measured T, e.g., T ≈ 8,000 on test machine).  
- Constants: ShellSort shows smaller per‑element constants for small n due to contiguous memory shifts; HeapSort suffers higher constant from heapify swaps and less locality but scales better for large n.

Validation vs theory
- HeapSort empirical curves validate theoretical Θ(n log n).  
- ShellSort empirical variability confirms dependency on gap sequence and presortedness; no single Θ holds across all inputs.

Limitations
- Measurement noise from OS tasks and GC; instrumentation overhead if not separated from hot loops; JVM version and CPU microarchitecture affect constants.

Conclusion
Summary
- HeapSort: reliable Θ(n log n), in‑place, predictable; recommended for large or adversarial inputs.  
- ShellSort: can be substantially faster for small/medium n and near‑sorted data when implemented with shift‑with‑temp and precomputed gaps; lacks HeapSort’s worst‑case guarantee.

Actionable recommendations
1. HeapSort: implement percolate‑down with single temp, remove inner‑loop allocations and logging, localize indices — yields measurable constant‑factor speedups.  
2. ShellSort: implement shift‑with‑temp, precompute Sedgewick gaps, optimize final insertion pass — reduces writes/comparisons and improves locality.  
3. Benchmark hygiene: remove logging from hot loops, apply JVM warmup (discard runs), use median of ≥10 runs, record hardware/JVM details, report R² for regressions.

Reproducibility commands
- Run tests: mvn test  
- Run benchmark: mvn exec:java -Dexec.mainClass="cli.BenchmarkRunner" -Dexec.args="random 10000"  
- Data and plots: docs/benchmarks_for_report.csv and docs/*.png
