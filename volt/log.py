import logging

logger = logging.getLogger(__name__)


class Task(object):
    def __init__(self, name):
        self.name = name
        self.started = False
        self.isdone = False

    def start(self):
        logger.info(self.name + '...')
        self.started = True
        return self

    def done(self):
        if not self.started:
            raise Exception(f"Task '{self.name}' done before it started.")
        if self.isdone:
            raise Exception(f"Task '{self.name}' already done!")
        logger.info(self.name + '...' + 'Done')
        self.isdone = True
        return self


def task(name):
    def decorator(function):
        def wrapper(*args, **kwargs):
            t = Task(name).start()
            result = function(*args, **kwargs)
            t.done()
            return result

        return wrapper

    return decorator
