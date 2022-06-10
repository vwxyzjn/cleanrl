import os


def add_header(dirname: str):
    """
    Add a header string with documentation link
    to each file in the directory `dirname`.
    """

    for filename in os.listdir(dirname):
        if filename.endswith(".py"):
            with open(os.path.join(dirname, filename)) as f:
                lines = f.readlines()

            # hacky bit
            exp_name = filename.split(".")[0]
            algo_name = exp_name.split("_")[0]
            header_string = f"# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/{algo_name}/#{exp_name}py"

            if not lines[0].startswith(header_string):
                print(f"adding headers for {filename}")
                lines.insert(0, header_string + "\n")
                with open(os.path.join(dirname, filename), "w") as f:
                    f.writelines(lines)


if __name__ == "__main__":
    add_header("cleanrl")
