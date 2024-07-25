from rlf.algos.base_algo import BaseAlgo

class BaseTabular(BaseAlgo):
    def get_add_args(self, parser):
        super().get_add_args(parser)
        #########################################
        # Overrides
        parser.add_argument('--num-processes', type=int, default=1)
        parser.add_argument('--num-steps', type=int, default=1)

        ADJUSTED_INTERVAL = 50
        parser.add_argument('--log-interval', type=int,
                            default=ADJUSTED_INTERVAL)
        # Will probably be very fast so no need to save or evaluate.
        parser.add_argument('--save-interval', type=int,
                            default=-1)
        parser.add_argument('--eval-interval', type=int,
                            default=-1)
        parser.add_argument('--normalize-env', type=bool, default=False)
