# Load packages
library(BAMMSC)
library(SeuratDisk)
library(Seurat)
library(aricode)
library(argparse)

# Parse arguments
parser <- ArgumentParser()
parser$add_argument("--h5ad-path", type = "character", help = "path to the h5ad file to be processed")
parser$add_argument("--K", type = "integer", help = "number of clusters for BAMMSC algorithm")
args <- parser$parse_args()

fname <- basename(args$h5ad_path)
fname <- substring(fname, 0, nchar(fname) - 5)

# Load dataset
# Convert(args$h5ad_path, dest = "h5seurat", overwrite = TRUE)
dataset <- LoadH5Seurat(paste(substring(args$h5ad_path, 0, nchar(args$h5ad_path) - 5), ".h5seurat", sep = ""))

# Construct algorithm input
if ("batch_id" %in% names(dataset@meta.data)) {
    dataset@meta.data$batch_indices <- dataset@meta.data$batch_id
} else if ("Batch_id" %in% names(dataset@meta.data)) {
    dataset@meta.data$batch_indices <- dataset@meta.data$Batch_id
} else if ("batch" %in% names(dataset@meta.data)) {
    dataset@meta.data$batch_indices <- dataset@meta.data$batch
}

if ("cell_type" %in% names(dataset@meta.data)) {
    dataset@meta.data$cell_types <- dataset@meta.data$cell_type
} else if ("Cell_type" %in% names(dataset@meta.data)) {
    dataset@meta.data$cell_types <- dataset@meta.data$Cell_type
} else if ("celltype" %in% names(dataset@meta.data)) {
    dataset@meta.data$cell_types <- dataset@meta.data$celltype
}

batches <- names(table(dataset@meta.data$batch_indices))
print(batches)

data_list <- list()
for (i in seq_len(length(batches))) {
    matrix_data = dataset@assays$RNA@data[, dataset@meta.data$batch_indices == batches[[i]]]
    data_list[[i]] <- matrix(
        data = matrix_data,
        nrow = nrow(matrix_data),
        ncol = ncol(matrix_data)
    )
    writeLines(sprintf("<%d> batch_name: %s; shape: %s", i, batches[[i]], paste(dim(data_list[[i]]), collapse = ' ')))
}

# Preprocess dataset for visualization
dataset <- NormalizeData(dataset)
all.genes <- rownames(dataset)
dataset <- ScaleData(dataset, features = all.genes)
dataset <- RunPCA(dataset, features = all.genes)
dataset <- RunUMAP(dataset, dims = 1:50)

# Run algo, print result and save images
writeLines(sprintf("\n\n========== %d ==========\n", args$K))

start <- proc.time()
result <- BAMMSC(data_list, K = args$K)
print(proc.time() - start)

mem <- unlist(result$mem)
dataset[["BAMMSC"]] <- mem
nmi <- NMI(mem, dataset@meta.data$cell_types)
ari <- ARI(mem, dataset@meta.data$cell_types)
writeLines(sprintf("ARI: %.4f", ari))
writeLines(sprintf("NMI: %.4f", nmi))

pdf(sprintf("figures/BAMMSC_%d.pdf",  args$K), width = 16, height = 8)
p1 <- DimPlot(dataset, reduction = "umap", group.by = "cell_types")
p2 <- DimPlot(dataset, reduction = "umap", group.by = "BAMMSC")
p1 + p2
dev.off()