#include "cpu_reference.h"
#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: test_cpu_ref <model.json> <model.bin>\n");
        return 1;
    }

    rdna4::CpuReference cpuRef;
    if (!cpuRef.load(argv[1], argv[2])) {
        fprintf(stderr, "Failed to load CPU reference model\n");
        return 1;
    }

    fprintf(stderr, "Running CPU reference forward pass...\n");
    auto logits = cpuRef.forward(1);
    fprintf(stderr, "Forward returned %zu logits\n", logits.size());

    if (logits.size() > 0) {
        printf("CPU logits[0..9]: ");
        for (int i = 0; i < 10 && i < (int)logits.size(); ++i) {
            printf("%.6f ", logits[i]);
        }
        printf("\n");

        auto [argmax, max] = rdna4::CpuReference::argmax(logits);
        printf("CPU argmax: token %u = %.6f\n", argmax, max);

        cpuRef.printTopK(logits, 5);
    }

    fprintf(stderr, "Done, exiting normally\n");
    return 0;
}
