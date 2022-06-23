from concurrent.futures import ThreadPoolExecutor
from math import log2, ceil
from os import urandom
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from vcgencmd import Vcgencmd
import time

import pyRAPL

from ecc.cipher import ElGamal
from ecc.curve import Curve
from ecc.curve import (
    ShortWeierstrassCurves,
    MontgomeryCurves,
    BrainpoolCurvesR1,
    BrainpoolCurvesT1,
    EdwardsCurves
)
from ecc.key import gen_keypair

SIZES = [
    1,
    2,
    4,
    8,
    16,
    32,
]

N = 100
MAX_CURVE_THREADS = 25
MAX_CURVE_TYPE_THREADS = 5
MAX_MESSAGE_THREADS = 6
MAX_TESTER_THREADS = 5


def get_curves():
    return [
        {
            'name': "ShortWeierstrassCurves",
            'curves': ShortWeierstrassCurves
        },
        {
            'name': "MontgomeryCurves",
            'curves': MontgomeryCurves
        },
        {
            'name': "BrainpoolCurvesT1",
            'curves': BrainpoolCurvesT1
        },
        {
            'name': "BrainpoolCurvesR1",
            'curves': BrainpoolCurvesR1
        },
        {
            'name': "EdwardsCurves",
            'curves': EdwardsCurves
        }
    ]


def average(list: List[float]) -> float:
    if list is None or len(list) == 0:
	    return
    return sum(list) / len(list)


class Benchmarker:
    def __init__(self, is_laptop: bool, n: int):
        self.n = n
        self.is_laptop = is_laptop
        self.raw_result = {}
        self.result = {}

    def benchmark(self):
        # Measure
        self._measure()
        # Process
        self._process_data()
        # Plot
        self._plot_data()

    def _measure(self):
        executor = ThreadPoolExecutor(max_workers=MAX_TESTER_THREADS)
        for size in SIZES:
            executor.submit(self._test_message(message_size=size))
        print("Finished measuring")

    def _process_data(self):
        # The data comes in as { message-size: { curve-type: { key-size: { process: { measurement-type: value }}}}}}
        # It needs to become: { message-size: { process: { measurement: { curve-type: { key-size: { value }}}}}}

        # This is done by first converting the dict in an array with lists with the desired order
        # Then that is converted into the dict
        result_array = []
        for message_size, message_size_values in self.raw_result.items():
            for curve_type, curve_type_values in message_size_values.items():
                for key_size, key_size_values in curve_type_values.items():
                    if key_size_values is not None:
                        for process, process_values in key_size_values.items():
                            for measurement, value in process_values.items():
                                result_array.append([message_size, process, measurement, curve_type, key_size, value])

        # Convert this large array to a dict
        for message_size, process, measurement, curve_type, key_size, value in result_array:
            if message_size not in self.result:
                self.result[message_size] = {}

            if process not in self.result[message_size]:
                self.result[message_size][process] = {}

            if measurement not in self.result[message_size][process]:
                self.result[message_size][process][measurement] = {}

            if curve_type not in self.result[message_size][process][measurement]:
                self.result[message_size][process][measurement][curve_type] = {}

            self.result[message_size][process][measurement][curve_type][key_size] = value
            "Finished Processing"

    def _plot_data(self):
        # The data is in the form: { message-size: { process: { measurement: { curve-type: { key-size: { value }}}}}}
        for message_size, message_size_values in self.result.items():
            for process, process_values in message_size_values.items():
                for measurement, measurement_values in process_values.items():
                    title = f"{measurement} usage in {process} with message size {message_size}, n={self.n}"
                    fig, ax = plt.subplots()
                    for curve_type, curve_type_values in measurement_values.items():

                        xs = []
                        ys = []
                        for key_size, value in curve_type_values.items():
                            xs.append(key_size)
                            ys.append(value)

                        x = np.array(xs)
                        y = np.array(ys)
                        ax.scatter(x, y, label=curve_type)
                        ax.plot(x, y)
                    ax.legend()
                    ax.set_title(title)
                    ax.set_xlabel('Key size')
                    ax.set_ylabel(measurement)

        plt.show()


    def _test_message(self, message_size):
        message_result = MessageTester(message_size=message_size, is_laptop=self.is_laptop, n=self.n).test_message()
        self.raw_result[message_size] = message_result


class MessageTester:
    def __init__(self, message_size: int, is_laptop: bool, n: int):
        self.n = n
        self.message = urandom(message_size)
        self.is_laptop = is_laptop
        self.result = {}

        print(f"Testing message size: {message_size}")

    def test_message(self):
        curve_types = get_curves()
        executor = ThreadPoolExecutor(max_workers=MAX_MESSAGE_THREADS)
        for curve_combi in curve_types:
            name = curve_combi['name']
            curves = curve_combi['curves']
            executor.submit(self._test_curve_type(name, curves))
        return self.result

    def _test_curve_type(self, name, curves):
        curve_type_result = CurveTypeTester(name=name, curves=curves, message=self.message,
                                            is_laptop=self.is_laptop, n=self.n).test_curve_type()
        self.result[name] = curve_type_result


class CurveTypeTester:
    def __init__(self, name: str, curves: List[Curve], message: bytes, is_laptop: bool, n: int):
        # Set the variables
        self.n = n
        self.name = name
        self.curves = curves
        self.message = message
        self.is_laptop = is_laptop
        self.result = {}

        print(f"Testing curve type: {name}")

    def test_curve_type(self):
        executor = ThreadPoolExecutor(max_workers=MAX_CURVE_TYPE_THREADS)
        for curve in self.curves:
            executor.submit(self._test_curve(curve))
        return self.result

    def _test_curve(self, curve):
        curve_size = ceil(log2(curve.p))
        curve_result = CurveTester(curve=curve, message=self.message, is_laptop=self.is_laptop, n=self.n).test_curve()
        self.result[curve_size] = curve_result


class CurveTester:
    def __init__(self, curve: Curve, message: bytes, is_laptop: bool, n: int):
        self.n = n
        self.curve = curve
        self.message = message
        self.is_laptop = is_laptop
        self.size = ceil(log2(curve.p))
        self.result = {
            'encryption': {
                'time': [],
                'power': [],
            },
            'decryption': {
                'time': [],
                'power': [],
            },
            'key_generation': {
                'time': [],
                'power': [],
            }
        }

    def test_curve(self):
        message_size = len(self.message)
        if not self.curve.fits_size(message_size):
            return

        if self.is_laptop:
            executor = ThreadPoolExecutor(max_workers=MAX_CURVE_THREADS)
            for i in range(self.n):
                executor.submit(self._test_laptop())
        else:
            executor = ThreadPoolExecutor(max_workers=MAX_CURVE_THREADS)
            for i in range(self.n):
                executor.submit(self._test_rpi())

        for process_type, measurements in self.result.items():
            for measurement_type, results in measurements.items():
                self.result[process_type][measurement_type] = round(average(results), 3)

        return self.result

    def _test_laptop(self):
        pyRAPL.setup()
        meter = pyRAPL.Measurement('bar')
        cipher_elg = ElGamal(self.curve)

        # Measure key generation
        meter.begin()
        private_key, public_key = gen_keypair(self.curve)
        meter.end()
        self.result['key_generation']['time'].append(round(meter.result.duration, 3))
        self.result['key_generation']['power'].append(round(average(meter.result.pkg), 3))

        # Measure encryption
        meter.begin()
        C1, C2 = cipher_elg.encrypt(self.message, public_key)
        meter.end()
        self.result['encryption']['time'].append(round(meter.result.duration, 3))
        self.result['encryption']['power'].append(round(average(meter.result.pkg), 3))

        # Measure decryption
        meter.begin()
        try:
            cipher_elg.decrypt(private_key, C1, C2)
        except ValueError:
            return
        meter.end()
        self.result['decryption']['time'].append(round(meter.result.duration, 3))
        self.result['decryption']['power'].append(round(average(meter.result.pkg), 3))

    def _test_rpi(self):
        vcgm = Vcgencmd()
        cipher_elg = ElGamal(self.curve)

        # Measure key generation
        key_generation_voltage_start = vcgm.measure_volts('core')
        key_generation_time_start = time.time()
        private_key, public_key = gen_keypair(self.curve)
        key_generation_time_end = time.time()
        key_generation_voltage_end = vcgm.measure_volts('core')
        self.result['key_generation']['time'].append(round(key_generation_time_end - key_generation_time_start, 3))
        self.result['key_generation']['power'].append(round(key_generation_voltage_end - key_generation_voltage_start, 3))

        # Measure decryption
        decryption_voltage_start = vcgm.measure_volts('core')
        decryption_time_start = time.time()
        C1, C2 = cipher_elg.encrypt(self.message, public_key)
        decryption_time_end = time.time()
        decryption_voltage_end = vcgm.measure_volts('core')
        self.result['decryption']['time'].append(round(decryption_time_end - decryption_time_start, 3))
        self.result['decryption']['power'].append(round(decryption_voltage_end - decryption_voltage_start, 3))

        # Measure encryption
        encryption_voltage_start = vcgm.measure_volts('core')
        encryption_time_start = time.time()
        try:
            cipher_elg.decrypt(private_key, C1, C2)
        except ValueError:
            return
        encryption_time_end = time.time()
        encryption_voltage_end = vcgm.measure_volts('core')
        self.result['encryption']['time'].append(round(encryption_time_end - encryption_time_start, 3))
        self.result['encryption']['power'].append(round(encryption_voltage_end - encryption_voltage_start, 3))


if __name__ == '__main__':
    is_laptop = not input("Laptop: y/n:\n") == "n"
    Benchmarker(is_laptop=is_laptop, n=N).benchmark()
