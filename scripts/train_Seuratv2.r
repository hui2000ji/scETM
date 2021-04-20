library(Seurat)
library(SeuratDisk)
library(dplyr)
library(cowplot)
library(argparse)
library(aricode)
library(future)
plan("multiprocess", workers = parallel::detectCores(logical = F))
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
}

parser <- ArgumentParser()
parser$add_argument("--h5seurat-path", type = "character", help = "path to the h5seurat file to be processed")
parser$add_argument("--resolutions", type = "double", nargs = "+", default = c(0.02, 0.03, 0.04, 0.05, 0.08, 0.1, 0.2, 0.25, 0.3, 0.4), help = "resolution of leiden/louvain clustering")
parser$add_argument("--subset-genes", type = "integer", default = 3000, help = "number of features (genes) to select, 0 for don't select")
parser$add_argument("--no-eval", action = "store_true", help = "do not eval")
parser$add_argument('--ckpt-dir', type = "character", help='path to checkpoint directory', default = file.path('..', 'results'))

args <- parser$parse_args()

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

args$ckpt_dir <- file.path(args$ckpt_dir, sprintf("%s_Seuratv2_%d_%s", dataset_str, args$subset_genes, strftime(Sys.time(),"%m_%d-%H_%M_%S")))
if (!dir.exists((args$ckpt_dir))) {
    dir.create(args$ckpt_dir)
}
scETM <- import("scETM")
scETM$initialize_logger(ckpt_dir = args$ckpt_dir)

start_time <- proc.time()
print_memory_usage()

dataset <- NormalizeData(dataset)
dataset <- FindVariableFeatures(
    object = dataset,
    selection.method = "vst",
    nfeatures = args$subset_genes,
    verbose = FALSE
)
dataset <- ScaleData(dataset, features = rownames(dataset))
print(proc.time() - start_time)
print_memory_usage()

fpath <- file.path(args$ckpt_dir, sprintf("%s_Seuratv2.h5seurat", dataset_str))
SaveH5Seurat(dataset, file = fpath, overwrite = T)
Convert(fpath, dest = "h5ad", overwrite = T)
file.remove(fpath)

if (!args$no_eval) {
    anndata <- import("anndata")
    fpath <- file.path(args$ckpt_dir, sprintf("%s_Seuratv2.h5ad", dataset_str))
    processed_data <- anndata$read_h5ad(fpath)
    sc$pp$pca(processed_data, n_comps = 50L)
    scETM$evaluate(
        processed_data,
        embedding_key = "X_pca",
        resolutions = args$resolutions,
        plot_dir = args$ckpt_dir,
        n_jobs = 1L
    )
}
