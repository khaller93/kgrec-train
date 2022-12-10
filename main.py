import logging
import os
import typer

from kgrec.cmd import train, compute

app = typer.Typer()
app.add_typer(train.app, name='train')
app.add_typer(compute.app, name='compute')


def _configure_logging():
    """ configures the logger for this application. The log level can be changed
     by setting the environment variable `LOG_LEVEL` in the execution
     environment of this application. """
    log_level = os.getenv('LOG_LEVEL', default='INFO')
    logging.basicConfig(format='%(levelname)-8s :: %(message)s',
                        level=log_level)


if __name__ == '__main__':
    _configure_logging()
    app()
