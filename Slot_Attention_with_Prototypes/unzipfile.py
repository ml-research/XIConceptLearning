import zipfile
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--input-file", type=str, default='.', help="input file")
parser.add_argument("--output-path", type=str, default='.', help="output path")
args = parser.parse_args()

with zipfile.ZipFile(args.input_file, "r") as z:
  z.extractall(args.output_path)
