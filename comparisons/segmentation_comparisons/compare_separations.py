import pandas as pd


separation_csv_path = '/Users/austin/Downloads/drive-download-20240426T200128Z-001/separation_all_vals.csv'
separation = pd.read_csv(separation_csv_path)

# get the last string portion of the "key" column after splitting by "_"
separation_val = separation['key'].str.split('_').str[-1]
# replace 'p' with '.' and convert to float
separation_val = separation_val.str.replace('p', '.').astype(float)
# add the new column to the dataframe
separation['separation'] = separation_val

# thickness column is in pixels, px_size column is px/micron, separation column is in pixels
# convert thickness and separation to microns
separation['thickness_um'] = separation['thickness'] / separation['px_size']
separation['separation_um'] = separation['separation'] / separation['px_size']
# divide separation by thickness to get the ratio
separation['separation_ratio'] = separation['separation_um'] / separation['thickness_um']
# round to closets multiple of 0.25
separation['separation_ratio'] = separation['separation_ratio'] / 0.25
separation['separation_ratio'] = separation['separation_ratio'].round() * 0.25

# group by "std" and "sep_ratio" columns
grouped = separation.groupby(['std', 'separation_ratio']).size().reset_index(name='count')
# get the f1 and iou mean and std for each group
grouped['nellie_f1_mean'] = separation.groupby(['std', 'separation_ratio'])['nellie_f1'].mean().values
grouped['nellie_f1_std'] = separation.groupby(['std', 'separation_ratio'])['nellie_f1'].std().values
grouped['nellie_f1_num'] = separation.groupby(['std', 'separation_ratio'])['nellie_f1'].count().values
grouped['mitometer_f1_mean'] = separation.groupby(['std', 'separation_ratio'])['mitometer_f1'].mean().values
grouped['mitometer_f1_std'] = separation.groupby(['std', 'separation_ratio'])['mitometer_f1'].std().values
grouped['mitometer_f1_num'] = separation.groupby(['std', 'separation_ratio'])['mitometer_f1'].count().values
grouped['mitograph_f1_mean'] = separation.groupby(['std', 'separation_ratio'])['mitograph_f1'].mean().values
grouped['mitograph_f1_std'] = separation.groupby(['std', 'separation_ratio'])['mitograph_f1'].std().values
grouped['mitograph_f1_num'] = separation.groupby(['std', 'separation_ratio'])['mitograph_f1'].count().values
grouped['otsu_f1_mean'] = separation.groupby(['std', 'separation_ratio'])['otsu_f1'].mean().values
grouped['otsu_f1_std'] = separation.groupby(['std', 'separation_ratio'])['otsu_f1'].std().values
grouped['otsu_f1_num'] = separation.groupby(['std', 'separation_ratio'])['otsu_f1'].count().values
grouped['triangle_f1_mean'] = separation.groupby(['std', 'separation_ratio'])['triangle_f1'].mean().values
grouped['triangle_f1_std'] = separation.groupby(['std', 'separation_ratio'])['triangle_f1'].std().values
grouped['triangle_f1_num'] = separation.groupby(['std', 'separation_ratio'])['triangle_f1'].count().values

grouped['nellie_iou_mean'] = separation.groupby(['std', 'separation_ratio'])['nellie_iou'].mean().values
grouped['nellie_iou_std'] = separation.groupby(['std', 'separation_ratio'])['nellie_iou'].std().values
grouped['nellie_iou_num'] = separation.groupby(['std', 'separation_ratio'])['nellie_iou'].count().values
grouped['mitometer_iou_mean'] = separation.groupby(['std', 'separation_ratio'])['mitometer_iou'].mean().values
grouped['mitometer_iou_std'] = separation.groupby(['std', 'separation_ratio'])['mitometer_iou'].std().values
grouped['mitometer_iou_num'] = separation.groupby(['std', 'separation_ratio'])['mitometer_iou'].count().values
grouped['mitograph_iou_mean'] = separation.groupby(['std', 'separation_ratio'])['mitograph_iou'].mean().values
grouped['mitograph_iou_std'] = separation.groupby(['std', 'separation_ratio'])['mitograph_iou'].std().values
grouped['mitograph_iou_num'] = separation.groupby(['std', 'separation_ratio'])['mitograph_iou'].count().values
grouped['otsu_iou_mean'] = separation.groupby(['std', 'separation_ratio'])['otsu_iou'].mean().values
grouped['otsu_iou_std'] = separation.groupby(['std', 'separation_ratio'])['otsu_iou'].std().values
grouped['otsu_iou_num'] = separation.groupby(['std', 'separation_ratio'])['otsu_iou'].count().values
grouped['triangle_iou_mean'] = separation.groupby(['std', 'separation_ratio'])['triangle_iou'].mean().values
grouped['triangle_iou_std'] = separation.groupby(['std', 'separation_ratio'])['triangle_iou'].std().values
grouped['triangle_iou_num'] = separation.groupby(['std', 'separation_ratio'])['triangle_iou'].count().values

# export the dataframe to a csv file
grouped.to_csv('/Users/austin/Downloads/drive-download-20240426T200128Z-001/separation_grouped.csv', index=False)
