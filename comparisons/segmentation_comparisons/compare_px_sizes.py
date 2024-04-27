import pandas as pd


px_sizes_csv_path = '/Users/austin/Downloads/drive-download-20240426T200128Z-001/pxsizes_all_vals.csv'
px_sizes = pd.read_csv(px_sizes_csv_path)

# group by "std" and "px_size" columns
grouped = px_sizes.groupby(['std', 'px_size']).size().reset_index(name='count')
# get the "nellie_f1" mean and std for each group
grouped['nellie_f1_mean'] = px_sizes.groupby(['std', 'px_size'])['nellie_f1'].mean().values
grouped['nellie_f1_std'] = px_sizes.groupby(['std', 'px_size'])['nellie_f1'].std().values
grouped['mitograph_f1_mean'] = px_sizes.groupby(['std', 'px_size'])['mitograph_f1'].mean().values
grouped['mitograph_f1_std'] = px_sizes.groupby(['std', 'px_size'])['mitograph_f1'].std().values
grouped['mitometer_f1_mean'] = px_sizes.groupby(['std', 'px_size'])['mitometer_f1'].mean().values
grouped['mitometer_f1_std'] = px_sizes.groupby(['std', 'px_size'])['mitometer_f1'].std().values
grouped['otsu_f1_mean'] = px_sizes.groupby(['std', 'px_size'])['otsu_f1'].mean().values
grouped['otsu_f1_std'] = px_sizes.groupby(['std', 'px_size'])['otsu_f1'].std().values
grouped['triangle_f1_mean'] = px_sizes.groupby(['std', 'px_size'])['triangle_f1'].mean().values
grouped['triangle_f1_std'] = px_sizes.groupby(['std', 'px_size'])['triangle_f1'].std().values

grouped['nellie_iou_mean'] = px_sizes.groupby(['std', 'px_size'])['nellie_iou'].mean().values
grouped['nellie_iou_std'] = px_sizes.groupby(['std', 'px_size'])['nellie_iou'].std().values
grouped['mitograph_iou_mean'] = px_sizes.groupby(['std', 'px_size'])['mitograph_iou'].mean().values
grouped['mitograph_iou_std'] = px_sizes.groupby(['std', 'px_size'])['mitograph_iou'].std().values
grouped['mitometer_iou_mean'] = px_sizes.groupby(['std', 'px_size'])['mitometer_iou'].mean().values
grouped['mitometer_iou_std'] = px_sizes.groupby(['std', 'px_size'])['mitometer_iou'].std().values
grouped['otsu_iou_mean'] = px_sizes.groupby(['std', 'px_size'])['otsu_iou'].mean().values
grouped['otsu_iou_std'] = px_sizes.groupby(['std', 'px_size'])['otsu_iou'].std().values
grouped['triangle_iou_mean'] = px_sizes.groupby(['std', 'px_size'])['triangle_iou'].mean().values
grouped['triangle_iou_std'] = px_sizes.groupby(['std', 'px_size'])['triangle_iou'].std().values

grouped.to_csv('/Users/austin/Downloads/drive-download-20240426T200128Z-001/pxsizes_grouped.csv', index=False)

