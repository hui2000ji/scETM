library(ComplexHeatmap)
library(circlize)
library(cluster)
library(svglite)

setwd('D:/1998c/Documents/Research/202002scRNA/package_release/scETM/scripts/heatmap_fig/AD')

mat <- read.table('perm_md_onesided_celltype.csv', row.names = 1,sep=',',check.names = F,
                   header = TRUE, stringsAsFactors = FALSE)
mat = t(mat)
aster <- read.table("perm_p_onesided_celltype.csv",row.names = 1,sep=',',
                    header = TRUE, stringsAsFactors = FALSE)

aster = t(aster)
hmap <- Heatmap(
  mat,
  cell_fun = function(j, i, x, y, w, h, fill) {
    if (aster[i, j] < 0.05) {
      grid.text("*", x, y)}},
  name='cell type',
  column_dend_side = "bottom",
  show_row_names = TRUE,
  show_column_names = TRUE,
  cluster_rows = FALSE,
  cluster_columns = FALSE,
  row_names_gp = gpar(fontsize = 7),
  column_names_gp = gpar(fontsize = 10),
  raster_quality = 5,
  rect_gp = gpar(col = "grey", lwd = 0.3))


mat2 <- read.table('perm_md_onesided_condition.csv', row.names = 1, sep=',', check.names = F,
                  header = TRUE, stringsAsFactors = FALSE)
mat2 = t(mat2)

aster2 <- read.table("perm_p_onesided_condition.csv", row.names = 1, sep=',',
                     header = TRUE, stringsAsFactors = FALSE)
aster2 = t(aster2)
hmap2 <- Heatmap(
  mat2,
  cell_fun = function(j, i, x, y, w, h, fill) {
    if(aster2[i, j] < 0.05) {
      grid.text("*", x, y)}},
  name='condition',
  column_dend_side = "bottom",
  show_row_names = TRUE,
  show_column_names = TRUE,
  cluster_rows = FALSE,
  cluster_columns = FALSE,
  row_names_gp = gpar(fontsize = 7),
  column_names_gp = gpar(fontsize = 10),
  raster_quality = 5,
  rect_gp = gpar(col = "grey", lwd = 0.3))

pdf(file = "panel_c_permtest.pdf",width=3,height=6)
draw(hmap+hmap2, heatmap_legend_side="right", annotation_legend_side="right")
dev.off()


svglite("panel_c_permtest.svg",width=3,height=6)
draw(hmap+hmap2, heatmap_legend_side="left", annotation_legend_side="right")
dev.off()
