#!/usr/bin/env python
import argparse
import pkg_resources

from honcaml.tools import execution


__version__ = pkg_resources.get_distribution('honcaml').version


def cli():
    """
    Command-line interface for HONCAML.
    """
    parser = argparse.ArgumentParser(
        prog="honcaml",
        description="HONCAML pipeline command-line interface.",
        usage="""
honcaml [<args>]
""")

    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version="%(prog)s " + __version__,
        help='HONCAML current version')

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help='YAML configuration file specifying pipeline options')

    args = parser.parse_args()
    pipeline_conf = args.config

    execution.Execution(pipeline_conf).run()


if __name__ == '__main__':
    cli()
