#!/usr/bin/env python
import argparse
import pkg_resources
from subprocess import Popen
from honcaml.tools import startup, utils
from honcaml.config import user
from honcaml.tools import logger_option

__version__ = pkg_resources.get_distribution('honcaml').version
TYPE_CHOICES = ['train', 'predict', 'benchmark']

parser = argparse.ArgumentParser(
    prog="honcaml",
    description="HoNCAML pipeline command-line interface.",
    usage="""
    honcaml [<args>]
    """)

parser.add_argument(
    "-v",
    "--version",
    action="version",
    version="%(prog)s " + __version__,
    help='HoNCAML current version')

parser.add_argument(
    "-c",
    "--config",
    type=str,
    help='YAML configuration file specifying pipeline options')

parser.add_argument(
    "-l",
    "--log",
    type=str,
    help='file path in which to store execution log')

parser.add_argument(
    "-b",
    "--generate-basic-config",
    type=str,
    help='generate most basic YAML configuration file. Requires -t argument.')

parser.add_argument(
    "-a",
    "--generate-advanced-config",
    type=str,
    help='generate advanced YAML configuration file. Requires -t argument.')

parser.add_argument(
    "-t",
    "--pipeline-type",
    type=str,
    choices=TYPE_CHOICES,
    help='type of execution used while creating YAML configuration. '
    'Only makes sense together with -a or -b arguments.')

parser.add_argument(
    "-g",
    "--gui",
    action='store_true',
    help='open GUI in a web browser tab'
)

args = parser.parse_args()


def main():
    """Main execution function."""
    if args.generate_basic_config or args.generate_advanced_config:
        conf_options = utils.get_configuration_arguments(args)
        user.export_config(*conf_options)
    elif args.gui:
        Popen(f'cd honcaml/visualization && streamlit run gui.py', shell=True)
    else:
        if args.config:
            config = logger_option.yaml_reader(args.config)
            startup.setup_logging_config(config)
        if args.log:
            startup.setup_file_logging(args.log)
        from honcaml.tools import execution
        execution.Execution(args.config).run()


if __name__ == '__main__':
    main()
