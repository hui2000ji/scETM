# load packages
library(SeuratDisk)
library(Seurat)
library(aricode)
library(SingleCellExperiment)
library(scAlign)
library(argparse)

# Parse arguments
parser <- ArgumentParser()
parser$add_argument("--h5seurat-path", type = "character", default = "../data/HumanPancreas/HumanPancreas.h5seurat", help = "path to the h5seurat file to be processed")
parser$add_argument("--supervised", type = "character", default = "none", help = "supervision setting")
parser$add_argument("--emb-dim", type = "integer", default = 128L, help = "latent cell embedding dimension")
parser$add_argument("--updates", type = "integer", default = 10000L, help = "number of updates for ")
parser$add_argument("--log-every", type = "integer", default = 5000L, help = "number of steps between checkpointing")
parser$add_argument("--train-decoder", action = "store_true", help = "train the decoder")
parser$add_argument("--model", choices = c("small", "medium", "large"), default = "medium", help = "model architecture")
parser$add_argument("--louvain-resolutions", type = "double", nargs = "*", default = c(0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1), help = "louvain resolution")
parser$add_argument("--leiden-resolutions", type = "double", nargs = "*", default = c(0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1), help = "leiden resolution")
parser$add_argument("--n-neighbors", type = "integer", default = 15, help = "number of neighbors for computing SNN graph")
parser$add_argument("--ckpt-dir", type = "character", default = "../results", help = "path to store logs and results in")
args <- parser$parse_args()

fname <- basename(args$h5seurat_path)
fname <- substring(fname, 0, nchar(fname) - 9)

dataset <- LoadH5Seurat(args$h5seurat_path)

colnames(dataset@meta.data) <- tolower(colnames(dataset@meta.data))
colnames(dataset@meta.data)[which(colnames(dataset@meta.data) == "cell_type")] <- "cell_types"
colnames(dataset@meta.data)[which(colnames(dataset@meta.data) == "batch_id")] <- "batch_indices"

batch_table <- table(dataset@meta.data$batch_indices)
batches <- names(batch_table)
largest_batch <- batches[as.integer(which.max(batch_table))]
data_list <- list()


for (batch in batches) {
    data_list[[batch]] <- SingleCellExperiment(assays = list(
        scale.data = dataset@assays$RNA@data
                     [, dataset@meta.data$batch_indices == batch]
    ))
}
labels <- list()
for (batch in batches) {
    labels[[batch]] <- (dataset@meta.data$cell_types
                        [dataset@meta.data$batch_indices == batch])
}

scAlign_obj <- scAlignCreateObject(
    sce.objects = data_list,
    labels = labels,
    project.name = fname
)
options <- scAlignOptions(
    steps = args$updates,
    batch.size = 800,
    log.every = args$log_every,
    architecture = args$model,
    batch.norm.layer = T,
    num.dim = args$emb_dim,
    norm = T,
    full.norm = F
)

if (length(batches) <= 2) {
    scAlign_obj <- scAlign(scAlign_obj,
        options = options,
        encoder.data = "scale.data",
        supervised = "none",
        run.encoder = T,
        run.decoder = F,
        log.dir = sprintf("%s/%s_scAlign", args$ckpt_dir, fname),
        log.results = T
    )
} else {
    scAlign_obj <- scAlignMulti(scAlign_obj,
        reference.data = largest_batch,
        options = options,
        encoder.data = "scale.data",
        supervised = "none",
        run.encoder = T,
        run.decoder = F,
        log.dir = sprintf("%s/%s_scAlign", args$ckpt_dir, fname),
        log.results = T
    )
}

seurat_obj <- as.Seurat(scAlign_obj,
    counts = "scale.data",
    data = NULL,
    project = sprintf("scAlign_%s", fname)
)

seurat_obj <- FindNeighbors(seurat_obj,
    reduction = "ALIGNED.GENE",
    dims = 1:args$emb_dim,
    k.param = args$n_neighbors
)
log_file <- file(sprintf("%s/log.txt", args$ckpt_dir))
do_clustering <- function(seurat_obj, algorithm, resolutions, log_file) {
    algorithm_dict <- list(louvain = 1, leiden = 4)

    writeLines(sprintf("========== %s ==========", algorithm), log_file)
    result <- list()
    best_resolution <- 0
    best_ari <- 0
    for (resolution in resolutions) {
        seurat_obj <- FindClusters(seurat_obj,
            resolution = resolution,
            n.iter = 15,
            algorithm = algorithm_dict[[algorithm]],
            verbose = F
        )
        result[[as.character(resolution)]] <- seurat_obj@active.ident

        ari <- ARI(seurat_obj@active.ident, seurat_obj@meta.data$scAlign.labels)
        n_clusters <- length(table(seurat_obj@active.ident))
        writeLines(sprintf("res: %4.2f, ARI: %7.4f, # clusters: %3d", resolution, ari, n_clusters), log_file)
        if (ari > best_ari) {
            best_ari <- ari
            best_resolution <- resolution
        }
    }
    writeLines(sprintf("Best %s res: %f", algorithm, best_resolution), log_file)
    seurat_obj@meta.data[[algorithm]] <- result[[as.character(best_resolution)]]
    return(seurat_obj)
}
seurat_obj <- do_clustering(seurat_obj,
    algorithm = "louvain",
    resolutions = args$louvain_resolutions,
    log_file = log_file
)
seurat_obj <- do_clustering(seurat_obj,
    algorithm = "leiden",
    resolutions = args$leiden_resolutions,
    log_file = log_file
)
close(log_file)

seurat_obj <- RunUMAP(seurat_obj,
    reduction = "ALIGNED.GENE",
    dims = 1:args$emb_dim
)
pdf(sprintf("%s/%s_scAlign.pdf", args$ckpt_dir, fname), width = 18, height = 16)
p1 <- UMAPPlot(seurat_obj,
    group.by = "group.by",
    reduction = "umap",
    pt.size = 1L
)
p2 <- UMAPPlot(seurat_obj,
    group.by = "louvain",
    reduction = "umap",
    pt.size = 1L
)
p3 <- UMAPPlot(seurat_obj,
    group.by = "leiden",
    reduction = "umap",
    pt.size = 1L
)
p4 <- UMAPPlot(seurat_obj,
    group.by = "scAlign.labels.orig",
    reduction = "umap",
    pt.size = 1L
)
p1 + p2 + p3 + p4
dev.off()
