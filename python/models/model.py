from hardware import chip


class Model:
    def __init__(self, args, config):
        self.args = args
        self.config = config
        # get the chip message from the chip module
        self.chip = chip.chip_map[args.chip_type]
