from loggers import BasicLogger
from tracking import Tracking
from utils.import_util import get_dataset

class VoxTracking:
    def __init__(self, args):
        self.args = args
        # logger (optional)
        self.logger = BasicLogger(args)
        # data stream
        self.data_stream = get_dataset(args)
        # tracker 
        self.tracker = Tracking(args, self.data_stream, self.logger, None)
        self.tracker.process_first_frame_t()

    def start(self):
        self.tracker.spin_for_pure_tracking()
