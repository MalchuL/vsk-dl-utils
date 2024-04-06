from lightning_utilities.core.rank_zero import rank_zero_only

from vsk_dl_utils.utils.frequency_counter import FreqCounter as _FreqCounter


class FreqCounter(_FreqCounter):

    @rank_zero_only
    def __call__(self, key, freq, update_count=True):
        return super().__call__(key, freq, update_count)
