#include "physis_mpi.h"

#define N (4)

typedef void (*grid_update_client_t)();
grid_update_client_t *update_clients;

int *create_grid()
{
    return (int*)calloc(N * N * N, sizeof(int));
}

void init_grid(int *g) 
{
    int i;
    for (i = 0; i < N * N * N; i++) {
        g[i] = i;
    }
    return;
}

void print_grid(int *g, FILE *out) 
{
    int i;
    fprintf(out, "grid: ");
    for (i = 0; i < N * N * N; i++) {
        fprintf(out, "%d ", g[i]);
    }
    fprintf(out, "\n");
    return;
}

int main(int argc, char *argv[]) 
{
    PhysisInit(&argc, &argv);
    unsigned s[3] = {N, N, N};
    uvec_t halo = {1, 1, 1};
    grid *g = grid_new(3, sizeof(int), s, halo, halo);
    int *gin = create_grid();
    init_grid(gin);
    print_grid(gin, stdout);
    int *gout = create_grid();
    grid_copyin(g, gin);
    grid_copyout(g, gout);
    printf("copyout\n");
    print_grid(gout, stdout);
    grid_free(g);
    PhysisFinalize();
    return 0;
}

