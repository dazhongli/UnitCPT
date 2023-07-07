import re
with open('./CPT_correlations.md','r') as fin:
    data = fin.read()
data_modified = data.replace(r'$$',' ')
pattern = r'\[label@(.*)\]'
replacement = r'{\\label{\1}}'
data_modified = re.sub(pattern, replacement, data_modified)
with open('./temp.md','w') as fin:
    fin.write(data_modified)
