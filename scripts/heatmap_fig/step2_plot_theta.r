library(ComplexHeatmap)
library(circlize)
library(cluster)
library(svglite)


setwd('D:/1998c/Documents/Research/202002scRNA/package_release/scETM/scripts/heatmap_fig/AD')
mat <- read.table('delta_sampled.csv', row.names = 1,sep=',',
                  header = TRUE, stringsAsFactors = FALSE,check.names=F)
mat<-t(mat)
metadata  <- read.table('meta_sampled.csv',sep=',', row.names = 1,
                        header = TRUE, stringsAsFactors = FALSE)
colorscheme = colorRamp2(c(min(mat) ,0, max(mat)), c("blue", "white", "red"))

ann <- data.frame(metadata$condition, metadata$cell_types, metadata$batch_indices)
colnames(ann) <- c('condition','cell_type','batch_id')

colours <- list(
  'condition' = c('0' = 'black', '1' = 'green'),
  'cell_type' = c('Ex'='red','Per'='grey',
                  'Ast' = 'blue','In' = 'purple',
                  'Opc' = 'pink', 'Mic' = 'yellow',
                  'Oli' = 'cyan', 'End' = 'limegreen'
  ))


colAnn <- HeatmapAnnotation(df = ann,
                            which = 'column',
                            annotation_width = unit(c(1, 2), 'cm'),
                            gap = unit(1, 'mm'),
                            col = colours)
hmap <- Heatmap(
  mat,
  col=colorscheme,
  column_title = "AD",
  name='Topic Intensity',
  column_dend_side = "bottom",
  column_dend_height = unit(10, "mm"),
  clustering_method_rows='average',
  show_row_names = TRUE,
  show_column_names = FALSE,
  cluster_rows = TRUE,
  cluster_columns = TRUE,
  show_column_dend = FALSE,
  show_row_dend = TRUE,
  row_dend_reorder = TRUE,
  column_dend_reorder = TRUE,
  clustering_method_columns = "average",
  row_names_gp = gpar(fontsize = 7),
  top_annotation=colAnn)

pdf(file = "panel_b_theta_sampled.pdf",width=8,height=6)
draw(hmap, heatmap_legend_side="left", annotation_legend_side="right")
dev.off()

svglite("panel_b_theta_sampled.svg",width=8,height=6)
draw(hmap, heatmap_legend_side="left", annotation_legend_side="right")
dev.off()

#extract order
row_order_hmap = row_order(hmap)
actual_topic_name = dimnames(hmap@matrix)[1]
mylist = list() 
mylist[["order"]] = row_order_hmap
mylist[["topic"]] = actual_topic_name
write.table(as.data.frame(mylist),file="order.csv", quote=F,sep=",",row.names=F)

