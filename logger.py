import logging
import sys


logging.basicConfig(level=logging.INFO,
                    format="PID %(process)5s: %(message)s",
                    stream=sys.stderr)
log = logging.getLogger()
