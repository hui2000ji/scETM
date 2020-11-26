library(Seurat)
library(SeuratDisk)
library(dplyr)
library(cowplot)
library(argparse)

print_memory_usage <- function() {
    for (line in readLines('/proc/self/status')) {
        if (substr(line, 1, 6) == 'VmPeak') {
            print(line)
        }
        if (substr(line, 1, 5) == 'VmRSS') {
            print(line)
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

args <- parser$parse_args()

fname <- substring(args$h5seurat_path, 1, nchar(args$h5seurat_path) - 9)
dataset <- LoadH5Seurat(args$h5seurat_path)
batches <- names(table(dataset@meta.data$batch_indices))
print(batches)

dataset_list <- list()
start_time <- proc.time()
print_memory_usage()

for (i in seq_len(length(batches))) {
    matrix_data <- dataset@assays$RNA@data[, dataset@meta.data$batch_indices == batches[[i]]]
    dataset_list[[i]] <- CreateSeuratObject(
        counts = matrix_data,
        min.cell = 0,
        min.features = 0
    )
    writeLines(sprintf("<%d> batch_name: %s; shape: %s", i, batches[[i]], paste(dim(dataset_list[[i]]), collapse = ' ')))
    dataset_list[[i]] <- NormalizeData(
        object = dataset_list[[i]],
        verbose = FALSE
    )
    if (args$subset_genes) {
        dataset_list[[i]] <- FindVariableFeatures(
            object = dataset_list[[i]],
            selection.method = "vst",
            nfeatures = args$subset_genes,
            verbose = FALSE
        )
    }
}

anchors <- FindIntegrationAnchors(
    object.list = dataset_list,
    dims = 1:30# ,
    # anchor.features = rownames(dataset_list[[1]]@assays$RNA@data)
)

integrated <- IntegrateData(
    anchorset = anchors,
    dims = 1:30
)

DefaultAssay(object = integrated) <- "integrated"

integrated <- ScaleData(
    object = integrated,
    verbose = FALSE
)
print(proc.time() - start_time)
print_memory_usage()

if (!args$no_eval) {
    dataset <- FindNeighbors(integrated, k.param = 20, dims = 1:20)
    for (res in args$resolutions) {
        dataset <- FindClusters(dataset, resolution = res)
        seurat <- dataset@meta.data$seurat_clusters
        nmi <- NMI(seurat, dataset@meta.data$cell_type)
        ari <- ARI(seurat, dataset@meta.data$cell_type)
        writeLines(sprintf("resolution: %.2f", res))
        writeLines(sprintf("ARI: %.4f", ari))
        writeLines(sprintf("NMI: %.4f", nmi))
        writeLines(sprintf("# clusters: %d", length(table(seurat))))
        if (!args$no_draw) {
            pdf(sprintf("figures/%s_Seurat_%.3f.pdf", fname, res), width = 16, height = 8)
            p1 <- DimPlot(dataset, reduction = "umap", group.by = "cell_types")
            p2 <- DimPlot(dataset, reduction = "umap", group.by = "condition")
            p1 + p2
            dev.off()
        }
    }
}