library(Seurat)
library(SeuratDisk)
library(dplyr)
library(cowplot)
library(argparse)
library(aricode)
library(reticulate)
reticulate::use_python("python")

print_memory_usage <- function() {
    library(reticulate)
    py_run_string('import psutil; pmem = psutil.Process().memory_info(); print(pmem); rss_mb = pmem.rss/1024')
    return(py$rss_mb)
}

parser <- ArgumentParser()
parser$add_argument("--h5seurat-path", type = "character", help = "path to the h5seurat file to be processed")
parser$add_argument("--resolutions", type = "double", nargs = "+", default = c(0.02, 0.03, 0.04, 0.05, 0.08, 0.1, 0.2, 0.25, 0.3, 0.4), help = "resolution of leiden/louvain clustering")
parser$add_argument("--subset-genes", type = "integer", default = 3000, help = "number of features (genes) to select, 0 for don't select")
parser$add_argument("--n-pcs", type = "integer", default = 30, help = "number of pcs to use during integration")
parser$add_argument("--no-eval", action = "store_true", help = "do not eval")
parser$add_argument("--reference", action = "store_true", help = "use the largest batch as reference")
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

ckpt_dir <- file.path(args$ckpt_dir, sprintf("%s_Seuratv3_%d_seed%d_%s", dataset_str, args$subset_genes, args$seed, strftime(Sys.time(),"%m_%d-%H_%M_%S")))
if (!dir.exists((ckpt_dir))) {
    dir.create(ckpt_dir)
}
scETM <- import("scETM")
scETM$initialize_logger(ckpt_dir = ckpt_dir)
anndata <- import("anndata")


start_time <- proc.time()[3]
start_mem <- print_memory_usage()

dataset_list <- SplitObject(dataset, split.by = "batch_indices")
batches <- names(dataset_list)
print(batches)

largest_batch <- NA
largest_batch_sample_size <- 0

for (i in seq_along(batches)) {
    dataset_list[[i]] <- NormalizeData(
        object = dataset_list[[i]],
        verbose = FALSE
    )
    ncells <- dim(dataset_list[[i]])[2]
    if (ncells > largest_batch_sample_size) {
        largest_batch <- i
        largest_batch_sample_size <- ncells
    }
}

if (args$subset_genes) {
    anchor_features <- args$subset_genes
} else {
    anchor_features <- rownames(dataset@assays$RNA@data)
}

if (args$reference){
    anchors <- FindIntegrationAnchors(
        object.list = dataset_list,
        dims = 1:args$n_pcs,
        anchor.features = anchor_features,
        reference = largest_batch
    )
} else {
    anchors <- FindIntegrationAnchors(
        object.list = dataset_list,
        dims = 1:args$n_pcs,
        anchor.features = anchor_features
    )
}

integrated <- IntegrateData(
    anchorset = anchors,
    dims = 1:args$n_pcs
)

DefaultAssay(object = integrated) <- "integrated"

integrated <- ScaleData(
    object = integrated,
    verbose = FALSE
)

time_cost <- proc.time()[3] - start_time
mem_cost <- print_memory_usage() - start_mem
writeLines(sprintf("Duration: %.1f s (%.1f min)", time_cost, time_cost / 60))

integrated <- RunPCA(integrated, npcs = args$n_pcs)
fpath <- file.path(ckpt_dir, sprintf("%s_Seuratv3_seed%d.h5ad", dataset_str, args$seed))
X <- integrated@reductions$pca@cell.embeddings
processed_data <- anndata$AnnData(
    X = X,
    obs = integrated@meta.data
)
processed_data$write_h5ad(fpath)

if (!args$no_eval) {
    result <- scETM$evaluate(
        processed_data,
        embedding_key = "X",
        resolutions = args$resolutions,
        plot_dir = ckpt_dir,
        plot_fname = sprintf("%s_Seuratv3_seed%d_eval", dataset_str, args$seed),
        n_jobs = 1L
    )
    line <- sprintf("%s\tSeuratv3\t%s\t%.4f\t%.4f\t%.4f\t%.5f\t%.5f\t%.2f\t%.0f",
        dataset_str, args$seed,
        result$ari, result$nmi, result$asw, result$ebm, result$k_bet,
        time_cost, mem_cost)
    write(line, file = file.path(args$ckpt_dir, "table1.tsv"), append = T)
}
