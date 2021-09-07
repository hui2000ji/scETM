library(ComplexHeatmap)
library(circlize)
library(cluster)
library(svglite)

setwd('D:/1998c/Documents/Research/202002scRNA/package_release/scETM/scripts/heatmap_fig/AD')
 
# user defined set of topics

mat <- read.table('beta_top10genes_selected_topics.csv', row.names = 1,sep=',',check.names = F,
                  header = TRUE, stringsAsFactors = FALSE)
mat <- t(mat)

colorscheme=colorRamp2(c(min(mat),0,max(mat)), c( "blue","white", "red"))

hmap <- Heatmap(
  mat,
  col=colorscheme,
  name='Topic Intensity',
  column_dend_side = "bottom",
  show_row_names = TRUE,
  show_column_names = TRUE,
  cluster_rows = FALSE,
  cluster_columns = FALSE,
  column_names_gp = gpar(fontsize = 6),
  row_names_gp = gpar(fontsize = 7),
  rect_gp = gpar(col = "grey", lwd = 0.1))


pdf(file = "panel_a_beta.pdf",width=10,height=2.5)
draw(hmap, heatmap_legend_side="left", annotation_legend_side="right")
dev.off()


svglite("panel_a_beta.svg",width=8,height=3)
draw(hmap, heatmap_legend_side="left", annotation_legend_side="right")
dev.off()
