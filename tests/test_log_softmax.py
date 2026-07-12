import unittest

import numpy

from ctc_forced_aligner import SAMPLING_FREQ, _log_softmax_in_place, generate_emissions


def reference_log_softmax(values):
    values = values.astype(numpy.float64)
    return values - numpy.logaddexp.reduce(values, axis=-1, keepdims=True)


class FakeSession:
    def __init__(self, logits):
        self.logits = logits

    def run(self, output_names, inputs):
        if output_names != ["logits"]:
            raise AssertionError(f"unexpected outputs: {output_names}")
        batch_size = inputs["input_values"].shape[0]
        return [numpy.repeat(self.logits, batch_size, axis=0)]


class LogSoftmaxTest(unittest.TestCase):
    def test_extreme_finite_logits_remain_finite_and_normalized(self):
        logits = numpy.array(
            [[1000.0, 999.0, -1000.0], [-1000.0, -999.0, 1000.0]], dtype=numpy.float32
        )
        expected = reference_log_softmax(logits)

        with numpy.errstate(over="raise", invalid="raise", divide="raise"):
            actual = _log_softmax_in_place(logits)

        self.assertIs(actual, logits)
        self.assertTrue(numpy.isfinite(actual).all())
        numpy.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-5)
        numpy.testing.assert_allclose(
            numpy.exp(actual).sum(axis=-1), 1.0, rtol=1e-6, atol=1e-6
        )

    def test_random_logits_match_float64_reference(self):
        generator = numpy.random.default_rng(0)
        logits = generator.normal(0.0, 40.0, size=(4, 7, 31)).astype(numpy.float32)
        logits += numpy.array([1000.0, -1000.0, 500.0, -500.0], dtype=numpy.float32)[
            :, None, None
        ]
        expected = reference_log_softmax(logits)

        actual = _log_softmax_in_place(logits.copy())

        self.assertEqual(actual.shape, logits.shape)
        self.assertEqual(actual.dtype, numpy.float32)
        numpy.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-4)

    def test_negative_infinity_class_is_preserved(self):
        logits = numpy.array([[0.0, -numpy.inf, -1.0]], dtype=numpy.float32)
        expected = reference_log_softmax(logits)

        actual = _log_softmax_in_place(logits)

        self.assertTrue(numpy.isneginf(actual[0, 1]))
        self.assertTrue(numpy.isfinite(actual[0, [0, 2]]).all())
        numpy.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
        numpy.testing.assert_allclose(
            numpy.exp(actual).sum(axis=-1), 1.0, rtol=1e-6, atol=1e-6
        )

    def test_generate_emissions_uses_stable_float32_normalization(self):
        logits = numpy.broadcast_to(
            numpy.array([1000.0, 999.0, -1000.0], dtype=numpy.float32), (1, 101, 3)
        ).copy()
        session = FakeSession(logits)
        waveform = numpy.zeros(SAMPLING_FREQ, dtype=numpy.float32)

        emissions, stride = generate_emissions(
            session,
            waveform,
            window_length=1,
            context_length=1,
            batch_size=1,
        )

        self.assertEqual(emissions.shape, (2, 4))
        self.assertEqual(emissions.dtype, numpy.float32)
        self.assertEqual(stride, 500)
        self.assertTrue(numpy.isfinite(emissions).all())
        numpy.testing.assert_allclose(
            numpy.exp(emissions[:, :-1]).sum(axis=-1), 1.0, rtol=1e-6, atol=1e-6
        )
        numpy.testing.assert_array_equal(emissions[:, -1], 0.0)


if __name__ == "__main__":
    unittest.main()
