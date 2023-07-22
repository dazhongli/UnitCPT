import argparse
import re
def main(input_file,output_file):
    with open(input_file,'r') as fin:
        data = fin.read()
    data_modified = data.replace(r'$$',' ')
    pattern = r'\[label@(.*)\]'
    replacement = r'{\\label{\1}}'
    data_modified = re.sub(pattern, replacement, data_modified)
    with open(output_file,'w') as fin:
        fin.write(data_modified)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help='path to input file')
    parser.add_argument('output_file', help='Path to output file')
    args = parser.parse_args()
    main(args.input_file,args.output_file)
