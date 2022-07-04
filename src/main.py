import argparse

from src.tools import execution

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run pipeline (pipeline_name).")

    parser.add_argument('pipeline_name',
                        type=str,
                        help='Pipeline to run')

    args = parser.parse_args()
    execution.Execution(args.pipeline_name).run()
