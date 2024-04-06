import warnings
from collections import defaultdict


class FreqCounter:
    def __init__(self):
        self._call_count = defaultdict(int)

    def __call__(self, key, freq, update_count=True):
        """Updates frequency counter :param key: Key value to update :param freq: How often it
        should trigger :param update_count: Should we update counter.

        If True updates counter, and False wouldn't. In False case may trigger several times
        :return:
        """
        if update_count:
            self._call_count[key] += 1

            if self._call_count[key] >= freq:
                self._call_count[key] = 0
                return True
            else:
                return False
        else:
            return self._call_count[key] + 1 >= freq

    def get_counter(self, freq, name=None, exist_name_ok=False):
        if name is None:
            i = 0
            while True:
                temp_name = 'counter_' + str(i)
                if temp_name not in self._call_count:
                    name = temp_name
                    warnings.warn('Name for counter was not specified. Defaulting to {}'.format(name))
                    break
                i += 1
        else:
            if name in self._call_count and not exist_name_ok:
                raise ValueError('Counter with name {} already exists'.format(name))

        def counter_fn(update_count=True):
            return self(name, freq, update_count=update_count)

        return counter_fn
