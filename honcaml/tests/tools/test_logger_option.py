import unittest

from honcaml.tools.logger_option import logger_opt


class LoggerOptionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.default = 'INFO'

    def test_logger_opt_info(self):
        expected = self.default
        log_config = logger_opt({
            'global': {'problem_type': 'classification'},
            'logging': {'level': 'INFO'}
        })
        self.assertEqual(expected, log_config)

    def test_logger_opt_no_camp(self):
        expected = None
        log_config = logger_opt({
            'global': {'problem_type': 'classification'}
        })
        self.assertEqual(expected, log_config)

    def test_logger_opt_lower_case(self):
        expected = self.default
        log_config = logger_opt({
            'global': {'problem_type': 'classification'},
            'logging': {'level': 'info'}

        })
        self.assertEqual(expected, log_config)

    def test_logger_opt_no_option(self):
        expected = None
        log_config = logger_opt({
            'global': {'problem_type': 'classification'},
            'logging': {'level': 'ENFO'}

        })
        self.assertEqual(expected, log_config)
