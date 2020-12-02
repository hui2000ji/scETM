# Load packages
library(liger)
library(Seurat)
library(SeuratDisk)
library(SeuratWrappers)
library(aricode)
library(argparse)

print_memory_usage <- function() {
    for (line in readLines('/proc/self/status')) {
        if (substr(line, 1, 6) == 'VmPeak') {
            writeLines(line)
        }
        if (substr(line, 1, 5) == 'VmRSS') {
            writeLines(line)
        }
    }
    print(gc())
}

parser <- ArgumentParser()
parser$add_argument("--h5seurat-path", type = "character", help = "path to the h5seurat file to be processed")
parser$add_argument("--resolutions", type = "double", nargs = "+", default = c(0.002, 0.004, 0.006, 0.01, 0.015, 0.02, 0.025, 0.03), help = "resolution of leiden/louvain clustering")
parser$add_argument("--subset-genes", type = "integer", default = 3000, help = "number of features (genes) to select, 0 for don't select")
parser$add_argument("--no-draw", action = "store_true", help = "do not draw")
parser$add_argument("--no-eval", action = "store_true", help = "do not eval")
parser$add_argument('--ckpt-dir', type = "character", help='path to checkpoint directory', default = file.path('..', 'results'))
args <- parser$parse_args()

# Load dataset
fname <- substring(basename(args$h5seurat_path), 1, nchar(args$h5seurat_path) - 9)
dataset <- LoadH5Seurat(args$h5seurat_path)

args$ckpt_dir <- file.path(args$ckpt_dir, sprintf("%s_Liger%d_%s", fname, args$subset_genes, strftime(Sys.time(),"%m_%d-%H_%M_%S")))
if (!file.exists((args$ckpt_dir))) {
    mkdir(args$ckpt_dir)
}

# Run algo, print result and save images
start <- proc.time()
print_memory_usage()

dataset <- NormalizeData(dataset)
if (args$subset_genes)
    dataset <- FindVariableFeatures(dataset, nfeatures = args$subset_genes)
dataset <- ScaleData(dataset, split.by = "batch_indices", do.center = FALSE)
dataset <- RunOptimizeALS(dataset, k = 20, lambda = 5, split.by = "batch_indices")
dataset <- RunQuantileNorm(dataset, knn_k = 20, split.by = "batch_indices")

print(proc.time() - start)
print_memory_usage()

if (!args$no_eval) {
    dataset <- FindNeighbors(dataset, reduction = "iNMF", k.param = 20, dims = 1:20)
    for (res in args$resolutions) {
        dataset <- FindClusters(dataset, resolution = res)
        seurat <- dataset@meta.data$seurat_clusters
        nmi <- NMI(seurat, dataset@meta.data$cell_types)
        ari <- ARI(seurat, dataset@meta.data$cell_types)
        writeLines(sprintf("resolution: %.2f", res))
        writeLines(sprintf("ARI: %.4f", ari))
        writeLines(sprintf("NMI: %.4f", nmi))
        writeLines(sprintf("# clusters: %d", length(table(seurat))))
        if (!args$no_draw) {
            pdf(file.path(args$ckpt_dir, sprintf("%s_Liger_%.3f.pdf", fname, res)), width = 16, height = 8)
            dataset <- RunUMAP(dataset, dims = 1:50)
            p1 <- DimPlot(dataset, reduction = "umap", group.by = "cell_types")
            p2 <- DimPlot(dataset, reduction = "umap", group.by = "condition")
            p1 + p2
            dev.off()
        }
    }
}
