import numpy as np


class DataSet(object):
    """Basic DataSet class.
    """
    def __init__(self, data_list, nb_ref=1):
        self.d_size = data_list[0].shape[0]
        self.data_list = data_list
        self.nb_data = len(data_list)
        self.epochs_completed = 0
        self.i_in_epoch = 0
        assert nb_ref >= 1
        self.ref_data = [0] * self.nb_data
        for idx_dt in range(self.nb_data):
            self.ref_data[idx_dt] = np.copy(self.data_list[idx_dt][0:nb_ref])

    @property
    def nb_entries(self):
        return self.d_size

    @property
    def fixed_sample(self):
        return self.ref_data

    def shuffle(self):
        perm = np.arange(self.d_size)
        np.random.shuffle(perm)

        for idx_dt in range(self.nb_data):
            self.data_list[idx_dt] = self.data_list[idx_dt][perm]

    def next_batch(self, batch_size, batch_wrap=True, shuffle=True):
        """Return the next `batch_size` examples from this data set.
        """
        start = self.i_in_epoch
        if self.epochs_completed == 0 and start == 0 and shuffle:
            self.shuffle()

        data_batch = [0] * self.nb_data
        if start + batch_size >= self.d_size:
            # Finished epoch
            self.epochs_completed += 1
            self.i_in_epoch = 0
            for idx_dt in range(self.nb_data):
                data_batch[idx_dt] = self.data_list[idx_dt][start:self.d_size]
            if shuffle:
                self.shuffle()

            if batch_wrap:
                # Start next epoch
                self.i_in_epoch = batch_size - (self.d_size - start)
                end = self.i_in_epoch

                for idx_dt in range(self.nb_data):
                    data_new_part = self.data_list[idx_dt][0:end]
                    # e.g.shape of two inputs: (58, 12), (70, 12)
                    data_batch[idx_dt] = np.vstack([data_batch[idx_dt], data_new_part])
            return data_batch
        else:
            self.i_in_epoch += batch_size
            end = self.i_in_epoch
            for idx_dt in range(self.nb_data):
                data_batch[idx_dt] = self.data_list[idx_dt][start:end]
            return data_batch


class DataList():
    def __init__(self, data):
        self.data = data
        self.d_size = len(data)
        self.perm = np.arange(self.d_size)
        self.i_in_epoch = 0

    @property
    def nb_entries(self):
        return self.d_size

    def shuffle(self):
        np.random.shuffle(self.perm)

    def random_sample(self, shuffle=True):
        if self.i_in_epoch == 0 and shuffle:
            self.shuffle()

        sample = self.data[self.perm[self.i_in_epoch]]
        self.i_in_epoch = (self.i_in_epoch + 1) % self.d_size
        return sample

    def fixed_sample(self, idx=0):
        assert idx < self.d_size
        return self.data[idx]


if __name__ == '__main__':
    a = np.arange(6).reshape([-1, 1])
    b = a * (-1)
    c = a * 10
    d = DataSet([a, b, c])
    for _ in range(3):
        d_a, d_b, d_c = d.next_batch(4)
        assert np.all(d_b == (d_a * -1))
        assert np.all(d_c == (d_a * 10))
        print(d_a.flatten(), d_b.flatten(), d_c.flatten())
    print(d_a)

    a = np.arange(3).reshape([-1, 1])
    d = DataSet([a, b])
    rand = [d.next_batch(1)[0][0, 0] for i in range(8)]
    print(d.fixed_sample[0][0, 0], rand, d.fixed_sample[0][0, 0])

    a = np.arange(3)
    d = DataList(a)
    rand = [d.random_sample() for i in range(8)]
    print(d.fixed_sample(), rand, d.fixed_sample())
