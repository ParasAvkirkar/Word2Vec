class ModelParams:
    def __init__(self):
        self.batch_size = 128
        self.skip_window = 4  # How many words to consider left and right.
        self.num_skips = 8  # How many times to reuse an input to generate a label.
        self.num_sampled = 64  # Number of negative examples to sample.
        self.max_num_steps = 200001
        self.learning_rate = 1.0

    def get_tuning_params(self):
        return self.batch_size, self.skip_window, self.num_skips, self.num_sampled, self.max_num_steps, self.learning_rate