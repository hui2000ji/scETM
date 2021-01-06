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
batches <- names(table(dataset@meta.data$batch_indices))
print(batches)

args$ckpt_dir <- file.path(args$ckpt_dir, sprintf("%s_Seurat%d_%s", dataset_str, args$subset_genes, strftime(Sys.time(),"%m_%d-%H_%M_%S")))
if (!dir.exists((args$ckpt_dir))) {
    dir.create(args$ckpt_dir)
}

dataset_list <- list()
start_time <- proc.time()
print_memory_usage()

for (i in seq_len(length(batches))) {
    matrix_data <- dataset@assays$RNA@data[, dataset@meta.data$batch_indices == batches[[i]]]
    dataset_list[[i]] <- CreateSeuratObject(
        counts = matrix_data,
        min.cell = 0,
        min.features = 0,
        meta.data = dataset@meta.data[dataset@meta.data$batch_indices == batches[[i]],]
    )
    writeLines(sprintf("<%d> batch_name: %s; shape: %s", i, batches[[i]], paste(dim(dataset_list[[i]]), collapse = ' ')))
    dataset_list[[i]] <- NormalizeData(
        object = dataset_list[[i]],
        verbose = FALSE
    )
    # if (args$subset_genes) {
    #     dataset_list[[i]] <- FindVariableFeatures(
    #         object = dataset_list[[i]],
    #         selection.method = "vst",
    #         nfeatures = args$subset_genes,
    #         verbose = FALSE
    #     )
    # }
}
if (args$subset_genes) {
    anchor_features <- args$subset_genes
} else {
    anchor_features <- rownames(dataset@assays$RNA@data)
}

anchors <- FindIntegrationAnchors(
    object.list = dataset_list,
    dims = 1:30,
    anchor.features = anchor_features
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
    integrated <- RunPCA(integrated, verbose = FALSE)
    integrated <- FindNeighbors(integrated, k.param = 20, dims = 1:20)
    best_ari <- -1
    best_res <- -1
    for (res in args$resolutions) {
        integrated <- FindClusters(integrated, resolution = res)
        seurat <- integrated@meta.data$seurat_clusters
        nmi <- NMI(seurat, integrated@meta.data$cell_types)
        ari <- ARI(seurat, integrated@meta.data$cell_types)
        writeLines(sprintf("resolution: %.2f", res))
        writeLines(sprintf("ARI: %.4f", ari))
        writeLines(sprintf("NMI: %.4f", nmi))
        writeLines(sprintf("# clusters: %d", length(table(seurat))))
        if (ari > best_ari) {
            integrated@meta.data$best_clusters <- seurat
            best_ari <- ari
            best_res <- res
        }
    }
    if (!args$no_draw) {
        integrated <- RunUMAP(integrated, dims = 1:50)
        pdf(
            file.path(args$ckpt_dir, sprintf("%s_Seurat_%.3f.pdf", dataset_str, best_res)),
            width = 24,
            height = 8
        )
        p1 <- DimPlot(integrated, reduction = "umap", group.by = "cell_types")
        p2 <- DimPlot(integrated, reduction = "umap", group.by = "best_clusters")
        p3 <- DimPlot(integrated, reduction = "umap", group.by = "batch_indices")
        print(p1 + p2 + p3)
        dev.off()
    }
}

fpath <- file.path(args$ckpt_dir, sprintf("%s_Seurat.h5seurat", dataset_str))
SaveH5Seurat(integrated, file = fpath, overwrite = T)
Convert(fpath, dest = "h5ad", overwrite = T)
