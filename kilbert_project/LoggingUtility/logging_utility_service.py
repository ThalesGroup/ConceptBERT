""" Global application logging_utility.

All modules use the same global logging_utility configuration.
"""
import os
import yaml
import logging.config
import logging
import coloredlogs

__all__ = ["LoggingUtilityService"]

"""
Call setup_logging prior to all logger.X call in order to have configuration from logging_utility.yaml as default.
Messages less than the given priority level will be ignored. Call setLevel() to
change the logger's priority level after it has been stared. Available
levels and their suggested meanings:

    DEBUG - output useful for developers
    INFO - trace normal program flow, especially external interactions
    WARN - an abnormal condition was detected that might need attention
    ERROR - an error was detected but execution continued
    CRITICAL - an error was detected and execution was halted

"""
class LoggingUtilityService():
    @staticmethod
    def setup_logging(
        default_path='/assets/config/logging.yaml',
        default_level=logging.INFO,
        env_key='LOG_CFG'
    ):
        """
        Logging setup.

        Args:
            default_path: path to the yaml configuration file
            default_level: default log level to the newly created logger
            env_key: name of the environment variable to use for logger configuration

        """
        path = default_path
        value = os.getenv(env_key, None)
        if value:
            path = value
        if os.path.exists(path):
            with open(path, 'rt') as f:
                try:
                    config = yaml.safe_load(f.read())
                    logging.config.dictConfig(config)
                    coloredlogs.install()
                except Exception as e:
                    print(e)
                    print('Error in Logging Configuration. Using default configs')
                    LoggingUtilityService._setup_basic_configuration(default_level)
        else:
            print('Failed to load configuration file. Using default configs')
            LoggingUtilityService._setup_basic_configuration(default_level)

    @staticmethod
    def _setup_basic_configuration(default_level):
        logging.basicConfig(level=default_level)
        coloredlogs.install(level=default_level)
