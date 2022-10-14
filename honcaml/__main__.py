#!/usr/bin/env python
import argparse

from honcaml.tools import execution


def cli():
    """
    Command-line interface for HONCAML.
    """
    parser = argparse.ArgumentParser(
        description="HONCAML pipeline command-line interface.",
        usage="""
honcaml [<args>]
""")

    parser.add_argument(
        "-c",
        "--config",
        default="base_pipeline.yaml",
        type=str,
        help='YAML configuration file specifying pipeline options')

    args = parser.parse_args()
    pipeline_conf = args.config

    execution.Execution(pipeline_conf).run()


if __name__ == '__main__':
    cli()
