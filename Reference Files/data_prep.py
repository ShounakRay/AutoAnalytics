import pandas as pd
import os
import pdfkit
import numpy as np

import tabula
import camelot

__location__ = '/Users/Ray/Documents/Python/Data Preparation'
file_name = 'Market Survey Template - Broadway Center - High Rise.xlsm'
# num_lines = sum(1 for line in open(os.path.join(__location__, file_name)))
file = open(os.path.join(__location__, file_name))
# f = open('{}.csv'.format(file.name), 'r', newline = '')

df = pd.read_excel(file_name, sheet_name = 1).replace(np.NaN, "    ")
df.index = pd.Series(df.index).fillna(method='ffill', axis = 0)
df.to_csv(file_name + ".csv")
df.to_html(file_name + ".html")
options = {'page-size':'A1', 'orientation':'portrait', 'dpi':400}
pdfkit.from_file(file_name + ".html", file_name + ".pdf", options = options)

tables = tabula.read_pdf(file_name + '.pdf', pages = "all", multiple_tables = True)
tables_1 = camelot.read_pdf(file_name + '.pdf', pages = "1")



#
