from dowel import LogOutput
from dowel.tabular_input import TabularInput
import datetime
import dateutil.tz
import sys
import time

class StdLogger(LogOutput):
    def __init__(self, log_interval, with_timestamp=True):
        self._with_timestamp = with_timestamp
        self.log_interval = log_interval
        self.accum_str = ""
        self.prev_n_steps = 0
        self.prev_time = time.time()

    def _calc_fps(self, d):
        n_steps = d._dict['TotalEnvSteps']
        cur_time = time.time()
        self.fps = int((n_steps - self.prev_n_steps) / (cur_time - self.prev_time))
        self.prev_time = cur_time
        self.prev_n_steps = n_steps

    @property
    def types_accepted(self):
        return (str, TabularInput)

    def record(self, data, prefix=''):
        if isinstance(data, str):
            out = prefix + data
            if self._with_timestamp:
                now = datetime.datetime.now(dateutil.tz.tzlocal())
                timestamp = now.strftime('%Y-%m-%d %H:%M:%S')
                out = '%s | %s' % (timestamp, out)
        elif isinstance(data, TabularInput):
            self._calc_fps(data)
            out = str(data)
            data.mark_str()
        else:
            raise ValueError('Unacceptable type')

        self.accum_str += out + '\n'


    def dump(self, step=None):
        sys.stdout.flush()

        tmp_accum_str = self.accum_str
        self.accum_str = ""
        if (step) % self.log_interval != 0:
            return
        print('FPS: %i' % self.fps)
        print(tmp_accum_str)
