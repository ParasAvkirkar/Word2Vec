class ModelParams:
    def __init__(self):
        self.batch_size = 128
        self.skip_window = 8  # How many words to consider left and right.
        self.num_skips = 16 # How many times to reuse an input to generate a label.
        self.num_sampled = 8  # Number of negative examples to sample.
        self.max_num_steps = 200001
        self.learning_rate = 1.0

    def get_tuning_params(self):
        return self.batch_size, self.skip_window, self.num_skips, self.num_sampled, self.max_num_steps, self.learning_rate

    def __str__(self):
        return "batch_size={0} skip_window={1} num_skips={2} num_sampled={3} max_num_steps={4} learning_rate={5}".format(
            str(self.batch_size), str(self.skip_window), str(self.num_skips), str(self.num_sampled),
            str(self.max_num_steps), str(self.learning_rate)
        )


