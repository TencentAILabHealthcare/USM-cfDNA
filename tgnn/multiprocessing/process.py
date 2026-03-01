# -*- coding: utf-8 -*-
# Copyright (c) 2022, Tencent Inc. All rights reserved.

import multiprocessing as mp
from signal import signal, SIGINT
from threading import Thread

from .thread import BackgroundIterator
from .utils import get_cpu_cores


def process_cancel():
    """
    Register an cancel event on sigint
    """
    event = mp.Event()
    signal(SIGINT, lambda *a: event.set())

    return event


class ProcessIterator(BackgroundIterator, mp.Process):
    """
    Runs an iterator in a separate process.
    """
    QueueClass = mp.Queue


def process_iter(iterator, maxsize=1):
    """
    Take an iterator and run it on another process.
    """
    return iter(ProcessIterator(iterator, maxsize=maxsize))


def process_map(func, iterator, num_procs=None, maxsize=None, starmap=False):
    """
    Take an `iterator` of key, value pairs and apply `func` to all values using `num_procs` processes.
    """
    if num_procs is None:
        num_procs = get_cpu_cores()

    elif 0 < num_procs < 1:
        num_procs = int(get_cpu_cores() * num_procs)

    if num_procs == 0:
        if starmap:
            return ((k, func(*v)) for k, v in iterator)
        else:
            return ((k, func(v)) for k, v in iterator)

    maxsize = maxsize or 2 * num_procs
    return iter(ProcessMap(func,
                           iterator,
                           num_procs,
                           starmap=starmap,
                           output_queue=mp.Queue(maxsize)))


class MapWorker(mp.Process):
    """
    Process that reads items from an input_queue, applies a
    func to them and puts them on an output_queue.
    """

    def __init__(self, func, input_queue, output_queue,
                 starmap=False, send_key=False):
        super().__init__()
        self.func = func
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.starmap = starmap
        self.send_key = send_key

    def run(self):
        while True:
            item = self.input_queue.get()
            if item is StopIteration:
                break

            k, v = item
            if self.starmap:
                if self.send_key:
                    self.output_queue.put((k, self.func(k, *v)))
                else:
                    self.output_queue.put((k, self.func(*v)))
            else:
                if self.send_key:
                    self.output_queue.put((k, self.func(k, v)))
                else:
                    self.output_queue.put((k, self.func(v)))


class ProcessMap(Thread):

    def __init__(self,
                 func,
                 iterator,
                 num_procs,
                 output_queue=None,
                 starmap=False,
                 send_key=False):
        super().__init__()
        self.iterator = iterator
        self.starmap = starmap
        self.send_key = send_key
        self.work_queue = mp.Queue(num_procs * 2)
        self.output_queue = output_queue or mp.Queue()
        self.processes = [
            MapWorker(func, self.work_queue, self.output_queue, self.starmap, self.send_key)
            for _ in range(num_procs)
        ]

    def start(self):
        for process in self.processes:
            process.start()
        super().start()

    def run(self):
        for k, v in self.iterator:
            self.work_queue.put((k, v))

        for _ in self.processes:
            self.work_queue.put(StopIteration)

        for process in self.processes:
            process.join()

        self.output_queue.put(StopIteration)

    def __iter__(self):
        self.start()
        while True:
            item = self.output_queue.get()
            if item is StopIteration:
                break
            yield item