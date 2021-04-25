library(Seurat)
library(SeuratDisk)
library(dplyr)
library(cowplot)
library(argparse)
library(aricode)
library(reticulate)
reticulate::use_python("python")

print_memory_usage <- function() {
    psutil <- import("psutil")
    pmem <- psutil$Process()$memory_info()
    for (name in names(pmem)) {
        if (name == "count" || name == "index") {
            next
        }
        writeLines(sprintf("%s\t%s", name, as.character(py_get_attr(pmem, name))))
    }
    return(pmem$rss)
}

parser <- ArgumentParser()
parser$add_argument("--h5seurat-path", type = "character", help = "path to the h5seurat file to be processed")
parser$add_argument("--resolutions", type = "double", nargs = "+", default = c(0.02, 0.03, 0.04, 0.05, 0.08, 0.1, 0.2, 0.25, 0.3, 0.4), help = "resolution of leiden/louvain clustering")
parser$add_argument("--subset-genes", type = "integer", default = 3000, help = "number of features (genes) to select, 0 for don't select")
parser$add_argument("--no-eval", action = "store_true", help = "do not eval")
parser$add_argument('--ckpt-dir', type = "character", help='path to checkpoint directory', default = file.path('..', 'results'))
parser$add_argument("--seed", type = "integer", default = -1, help = "random seed.")

args <- parser$parse_args()

if (args$seed >= 0) {
    set.seed(args$seed)
}

library(reticulate)
reticulate::use_python("python")
matplotlib <- import("matplotlib")
matplotlib$use("Agg")
sc <- import("scanpy")
sc$settings$set_figure_params(
    dpi=120,
    dpi_save=250,
    facecolor="white",
    fontsize=10,
    figsize=c(10, 10)
)

dataset_str <- basename(args$h5seurat_path)
dataset_str <- substring(dataset_str, 1, nchar(dataset_str) - 9)
dataset <- LoadH5Seurat(args$h5seurat_path)
batches <- names(table(dataset@meta.data$batch_indices))
print(batches)

ckpt_dir <- file.path(args$ckpt_dir, sprintf("%s_Seuratv3_%d_%s", dataset_str, args$subset_genes, strftime(Sys.time(),"%m_%d-%H_%M_%S")))
if (!dir.exists((ckpt_dir))) {
    dir.create(ckpt_dir)
}
scETM <- import("scETM")
scETM$initialize_logger(ckpt_dir = ckpt_dir)

dataset_list <- list()
for (i in seq_along(batches)) {
    matrix_data <- dataset@assays$RNA@data[, dataset@meta.data$batch_indices == batches[[i]]]
    dataset_list[[i]] <- CreateSeuratObject(
        counts = matrix_data,
        min.cell = 0,
        min.features = 0,
        meta.data = dataset@meta.data[dataset@meta.data$batch_indices == batches[[i]],]
    )
    writeLines(sprintf("<%d> batch_name: %s; shape: %s", i, batches[[i]], paste(dim(dataset_list[[i]]), collapse = ' ')))
}

start_time <- proc.time()[3]
start_mem <- print_memory_usage()

for (i in seq_along(batches)) {
    dataset_list[[i]] <- NormalizeData(
        object = dataset_list[[i]],
        verbose = FALSE
    )
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

time_cost <- proc.time()[3] - start_time
mem_cost <- print_memory_usage() - start_mem

fpath <- file.path(ckpt_dir, sprintf("%s_Seuratv3.h5seurat", dataset_str))
SaveH5Seurat(integrated, file = fpath, overwrite = T)
Convert(fpath, dest = "h5ad", overwrite = T)
file.remove(fpath)

if (!args$no_eval) {
    anndata <- import("anndata")
    fpath <- file.path(ckpt_dir, sprintf("%s_Seuratv3.h5ad", dataset_str))
    processed_data <- anndata$read_h5ad(fpath)
    sc$pp$pca(processed_data, n_comps = 50L)
    result <- scETM$evaluate(
        processed_data,
        embedding_key = "X_pca",
        resolutions = args$resolutions,
        plot_dir = ckpt_dir,
        n_jobs = 1L
    )
    line <- sprintf("%s\tSeuratv3\t%s\t%.4f\t%.4f\t%.5f\t%.5f\t%.2f\t%d\n",
        dataset_str, args$seed,
        result$ari, result$nmi, result$ebm, result$k_bet,
        time_cost, mem_cost)
    write(line, file = file.path(args$ckpt_dir, "table1.tsv"), append = T)
}
