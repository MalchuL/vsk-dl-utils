from collections import defaultdict

from lightning_utilities.core.rank_zero import rank_zero_only


class FreqCounter:
    def __init__(self):
        self._call_count = defaultdict(int)

    @rank_zero_only
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
