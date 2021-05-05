# Load packages
library(rliger)
library(dplyr)
library(Seurat)
library(SeuratDisk)
library(SeuratWrappers)
library(aricode)
library(argparse)
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
parser$add_argument("--resolutions", type = "double", nargs = "+", default = c(0.002, 0.004, 0.006, 0.01, 0.015, 0.02, 0.025, 0.03), help = "resolution of leiden/louvain clustering")
parser$add_argument("--subset-genes", type = "integer", default = 3000, help = "number of features (genes) to select, 0 for don't select")
parser$add_argument("--no-eval", action = "store_true", help = "do not eval")
parser$add_argument("--ckpt-dir", type = "character", help="path to checkpoint directory", default = file.path("..", "results"))
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

# Load dataset
dataset_str <- basename(args$h5seurat_path)
dataset_str <- substring(dataset_str, 1, nchar(dataset_str) - 9)
seurat_obj <- LoadH5Seurat(args$h5seurat_path)
metadata <- seurat_obj@meta.data
batches <- names(table(metadata$batch_indices))
print(batches)
genes_use <- row.names(seurat_obj@assays$RNA@data)[rowSums(seurat_obj@assays$RNA@data) > 0]

ckpt_dir <- file.path(args$ckpt_dir, sprintf("%s_Liger%d_%s_seed%d", dataset_str, args$subset_genes, strftime(Sys.time(),"%m_%d-%H_%M_%S"), args$seed))
if (!dir.exists((ckpt_dir))) {
    dir.create(ckpt_dir)
}
scETM <- import("scETM")
scETM$initialize_logger(ckpt_dir = ckpt_dir)

# Run algo, print result and save images
start_time <- proc.time()[3]
start_mem <- print_memory_usage()

dataset_list <- list()
for (i in seq_along(batches)) {
    matrix_data <- seurat_obj@assays$RNA@data[genes_use, metadata$batch_indices == batches[[i]]]
    dataset_list[[i]] <- matrix_data
}
names(dataset_list) <- batches
dataset <- createLiger(dataset_list, remove.missing = F)
dataset <- normalize(dataset)
if (args$subset_genes)
    dataset <- selectGenes(dataset, num.genes = args$subset_genes)
dataset <- scaleNotCenter(dataset)
dataset <- optimizeALS(dataset, k = 20, lambda = 5)
dataset <- quantile_norm(dataset, knn_k = 20)

time_cost <- proc.time()[3] - start_time
mem_cost <- print_memory_usage() - start_mem

fpath <- file.path(ckpt_dir, sprintf("%s_Liger.h5ad", dataset_str))
anndata <- import("anndata")
processed_data <- anndata$AnnData(
    X = t(do.call(cbind, dataset@raw.data)),
    obs = metadata,
    obsm = list(H_norm = dataset@H.norm),
    uns = list(V = dataset@V, W = dataset@W)
)
processed_data$write_h5ad(fpath)

if (!args$no_eval) {
    scETM <- import("scETM")
    result <- scETM$evaluate(
        processed_data,
        embedding_key = "H_norm",
        resolutions = args$resolutions,
        plot_dir = ckpt_dir,
        n_jobs = 1L
    )
    line <- sprintf("%s\tLiger\t%s\t%.4f\t%.4f\t%.5f\t%.5f\t%.2f\t%d\n",
        dataset_str, args$seed,
        result$ari, result$nmi, result$ebm, result$k_bet,
        time_cost, mem_cost)
    write(line, file = file.path(args$ckpt_dir, "table1.tsv"), append = T)
}
