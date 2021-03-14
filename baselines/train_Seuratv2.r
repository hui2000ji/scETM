library(Seurat)
library(SeuratDisk)
library(dplyr)
library(cowplot)
library(argparse)
library(aricode)

print_memory_usage <- function() {
    if (file.exists('/proc/self/status')) {
        for (line in readLines('/proc/self/status')) {
            if (substr(line, 1, 6) == 'VmPeak') {
                writeLines(line)
            }
            if (substr(line, 1, 5) == 'VmRSS') {
                writeLines(line)
            }
        }
    }
    print(gc())
}

parser <- ArgumentParser()
parser$add_argument("--h5seurat-path", type = "character", help = "path to the h5seurat file to be processed")
parser$add_argument("--resolutions", type = "double", nargs = "+", default = c(0.02, 0.03, 0.04, 0.05, 0.08, 0.1, 0.2, 0.25, 0.3, 0.4), help = "resolution of leiden/louvain clustering")
parser$add_argument("--subset-genes", type = "integer", default = 3000, help = "number of features (genes) to select, 0 for don't select")
parser$add_argument("--no-draw", action = "store_true", help = "do not draw")
parser$add_argument("--no-eval", action = "store_true", help = "do not eval")
parser$add_argument('--ckpt-dir', type = "character", help='path to checkpoint directory', default = file.path('..', 'results'))

args <- parser$parse_args()

dataset_str <- basename(args$h5seurat_path)
dataset_str <- substring(dataset_str, 1, nchar(dataset_str) - 9)
dataset <- LoadH5Seurat(args$h5seurat_path)
args$ckpt_dir <- file.path(args$ckpt_dir, sprintf("%s_Seurat%d_%s", dataset_str, args$subset_genes, strftime(Sys.time(),"%m_%d-%H_%M_%S")))
if (!dir.exists((args$ckpt_dir))) {
    dir.create(args$ckpt_dir)
}

dataset <- NormalizeData(dataset)
dataset <- FindVariableFeatures(
    object = dataset,
    selection.method = "vst",
    nfeatures = args$subset_genes,
    verbose = FALSE
)
dataset <- ScaleData(dataset, features = rownames(dataset))

if (!args$no_eval) {
    dataset <- RunPCA(dataset, verbose = FALSE)
    dataset <- FindNeighbors(dataset, k.param = 20, dims = 1:20)
    best_ari <- -1
    best_res <- -1
    for (res in args$resolutions) {
        dataset <- FindClusters(dataset, resolution = res)
        seurat <- dataset@meta.data$seurat_clusters
        nmi <- NMI(seurat, dataset@meta.data$cell_types)
        ari <- ARI(seurat, dataset@meta.data$cell_types)
        writeLines(sprintf("resolution: %.3f", res))

        writeLines(sprintf("ARI: %.4f", ari))
        writeLines(sprintf("NMI: %.4f", nmi))
        writeLines(sprintf("# clusters: %d", length(table(seurat))))
        if (ari > best_ari) {
            dataset@meta.data$best_clusters <- seurat
            best_ari <- ari
            best_res <- res
        }
    }
    if (!args$no_draw) {
        dataset <- RunUMAP(dataset, dims = 1:50)
        pdf(
            file.path(args$ckpt_dir, sprintf("%s_Seurat_%.3f.pdf", dataset_str, best_res)),
            width = 24,
            height = 8
        )
        p1 <- DimPlot(dataset, reduction = "umap", group.by = "cell_types")
        p2 <- DimPlot(dataset, reduction = "umap", group.by = "best_clusters")
        p3 <- DimPlot(dataset, reduction = "umap", group.by = "batch_indices")
        print(p1 + p2 + p3)
        dev.off()
    }
}

fpath <- file.path(args$ckpt_dir, sprintf("%s_Seurat.h5seurat", dataset_str))
SaveH5Seurat(dataset, file = fpath, overwrite = T)
Convert(fpath, dest = "h5ad", overwrite = T)
