# @Author: Shounak Ray <Ray>
# @Date:   28-Sep-2020 10:09:50:507  GMT-0600
# @Email:  rijshouray@gmail.com
# @Filename: extracontent.py
# @Last modified by:   Ray
# @Last modified time: 24-Feb-2021 00:02:23:238  GMT-0700
# @License: [Private IP]


# pdf_HTML = weasyprint.HTML(file_name + ".html")
# pdf_CSS = weasyprint.CSS(string = """
#             @page {
#               size: A1;
#               margin: 1.0 cm;
#             }""")
# pdf_HTML.write_pdf(file_name + ".pdf", stylesheets = [css])
#
# open(file_name + ".pdf", "wb").write(pdf)
#
# total_file = 'ID,Diagnosis,'
# total_file += str([[ for i in range(29 + 1)]
#                    for n in range(3 + 1)]).replace("[", "").replace("]", "").replace("'", "")
# feature_list += ['radius', 'texture', 'perimeter','area', 'smoothness', 'compactness', 'concavity', 'concave_points',
#                  'symmetry','fractal_dimension']
# for i in range(num_lines):
#     x = file.readline() #reads one line ...
#     total_file += x
#
# # df = pd.DataFrame(list(reader(total_file)))
# StringData = StringIO(total_file)
#
# df = pd.read_csv(StringData, sep = ",")
# df.iloc[0]


# a = np.array(list(zip(x_top_dist, y_top_dist)))
# row_sums = a.sum(axis = 1)
# new_matrix = a / row_sums[:, np.newaxis]
# x_top_dist, y_top_dist = list(zip(*new_matrix))

# if(curve_df.empty):
#     curve_df = pd.DataFrame(zip(x_top_dist, y_top_dist, [name_val] * len(y_top_dist)), columns = [name_val + "_X", name_val + "_Y", 'model_id'])
# else:
# curve_df.columns = [name_val + "_X", name_val + "_Y"]
# curve_df['model_id'] = name_val
