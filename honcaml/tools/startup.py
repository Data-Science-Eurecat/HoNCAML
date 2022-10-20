import datetime
import logging
import sys

from honcaml.config import params
from honcaml.__main__ import args

# Load settings
params['exec_name'] = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

# Set logger
log_params = params['logging']
logger = logging.getLogger(__name__)
hdlr_out = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(log_params['formatter']['format'],
                              log_params['formatter']['time_format'])
hdlr_out.setFormatter(formatter)
logger.addHandler(hdlr_out)
if args.log:
    hdlr_file = logging.FileHandler(args.log)
    hdlr_file.setFormatter(formatter)
    logger.addHandler(hdlr_file)
logger.setLevel(getattr(logging, log_params['level']))
logger.propagate = False
