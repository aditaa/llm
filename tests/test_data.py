import unittest

from llm.data import TokenWindowDataset, collate_batch, iter_batches, split_indices, split_token_ids


class TokenWindowDatasetTests(unittest.TestCase):
    def test_dataset_edge_cases_and_length(self) -> None:
        self.assertEqual(len(TokenWindowDataset([1, 2, 3, 4], context_length=4)), 0)

        ds = TokenWindowDataset([0, 1, 2, 3, 4], context_length=4)
        self.assertEqual(len(ds), 1)
        self.assertEqual(ds[0], ([0, 1, 2, 3], [1, 2, 3, 4]))

        stride_ds = TokenWindowDataset(list(range(8)), context_length=3, stride=2)
        self.assertEqual(len(stride_ds), 3)
        self.assertEqual(stride_ds[1], ([2, 3, 4], [3, 4, 5]))

    def test_next_token_shift_correctness(self) -> None:
        ds = TokenWindowDataset(list(range(10)), context_length=4)
        for x, y in ds:
            self.assertEqual(x[1:], y[:-1])

    def test_split_helpers_and_reproducibility(self) -> None:
        train_ids, val_ids = split_token_ids(list(range(10)), train_ratio=0.8)
        self.assertEqual(train_ids, list(range(8)))
        self.assertEqual(val_ids, [8, 9])

        train_1, val_1 = split_indices(20, train_ratio=0.75, seed=7, shuffle=True)
        train_2, val_2 = split_indices(20, train_ratio=0.75, seed=7, shuffle=True)
        train_3, val_3 = split_indices(20, train_ratio=0.75, seed=19, shuffle=True)

        self.assertEqual((train_1, val_1), (train_2, val_2))
        self.assertNotEqual((train_1, val_1), (train_3, val_3))
        self.assertEqual(sorted(train_1 + val_1), list(range(20)))

    def test_collate_and_iter_batches(self) -> None:
        ds = TokenWindowDataset(list(range(12)), context_length=3)
        batch_x, batch_y = next(iter_batches(ds, batch_size=2, shuffle=False))
        self.assertEqual(len(batch_x), 2)
        self.assertEqual(len(batch_y), 2)
        self.assertEqual(batch_x[0][1:], batch_y[0][:-1])

        examples = [([1, 2, 3], [2, 3, 4]), ([4, 5, 6], [5, 6, 7])]
        collated_x, collated_y = collate_batch(examples)
        self.assertEqual(collated_x[1], [4, 5, 6])
        self.assertEqual(collated_y[1], [5, 6, 7])


if __name__ == "__main__":
    unittest.main()
