import sys


def convert_dataset(input_file: str, output_file: str) -> None:
    """
    Read and convert dataset to plain text file.
    This is needed for all frameworks to work with standard file types instead
    of arff ones, which are not always supported.
    This is done in the following order:
      1. Load arff dataset
      2. Read dataset as pandas DataFrame
      3. Store dataframe into a csv file, keeping its name.

    Args:
        input_path: Input directory.
        output_path: Output directory.
    """
    with open(input_file, "r", encoding="utf-8") as inFile:
        text = inFile.readlines()

    data = False
    header = ""
    new_content = []
    for line in text:
        if not data:
            if "@ATTRIBUTE" in line or "@attribute" in line:
                attributes = line.split()
                if ("@attribute" in line):
                    attri_case = "@attribute"
                else:
                    attri_case = "@ATTRIBUTE"
                column_name = attributes[attributes.index(attri_case) + 1]
                header = header + column_name + ","
            elif "@DATA" in line or "@data" in line:
                data = True
                header = header[:-1]
                header += '\n'
                new_content.append(header)
        else:
            new_content.append(line)

    with open(output_file, "w") as outfile:
        outfile.writelines(new_content)


if __name__ == '__main__':
    input_file, output_file = sys.argv[1], sys.argv[2]
    convert_dataset(input_file, output_file)
