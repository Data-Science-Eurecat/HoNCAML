#!/usr/bin/env python
import argparse
import pkg_resources
from honcaml.tools import startup, utils
from honcaml.config import user

__version__ = pkg_resources.get_distribution('honcaml').version

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

parser.add_argument(
    "-l",
    "--log",
    type=str,
    help='File path in which to store execution log')

parser.add_argument(
    "-b",
    "--generate-basic-config",
    type=str,
    help='Generate most basic YAML configuration file')

parser.add_argument(
    "-a",
    "--generate-advanced-config",
    type=str,
    help='Generate advanced YAML configuration file')

args = parser.parse_args()


def main():
    """Main execution function."""
    if args.generate_basic_config or args.generate_advanced_config:
        conf_type, conf_value = utils.get_config_generation_argname_value(args)
        user.export_config(conf_type, conf_value)
    else:
        if args.log:
            startup.setup_file_logging(args.log)
        from honcaml.tools import execution
        execution.Execution(args.config).run()


if __name__ == '__main__':
    main()
