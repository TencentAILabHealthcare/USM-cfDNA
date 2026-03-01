# -*- coding: utf-8 -*-
# Copyright (c) 2022, Tencent Inc. All rights reserved.

import queue
from itertools import count
from threading import Thread


class BackgroundIterator:
    """
    Runs an iterator in the background.
    """

    def __init__(self, iterator, maxsize=10):
        super().__init__()
        self.iterator = iterator
        self.queue = self.QueueClass(maxsize)

    def __iter__(self):
        self.start()
        while True:
            item = self.queue.get()
            if item is StopIteration:
                break
            yield item

    def run(self):
        for item in self.iterator:
            self.queue.put(item)
        self.queue.put(StopIteration)

    def stop(self):
        self.join()


class ThreadIterator(BackgroundIterator, Thread):
    """
    Runs an iterator in a separate process.
    """
    QueueClass = queue.Queue


def thread_iter(iterator, maxsize=1):
    """
    Take an iterator and run it on another thread.
    """
    return iter(ThreadIterator(iterator, maxsize=maxsize))


class ThreadMap(Thread):

    def __init__(self, worker_type, iterator, n_thread, maxsize=2):
        super().__init__()
        self.iterator = iterator
        self.n_thread = n_thread
        self.work_queues = [queue.Queue(maxsize) for _ in range(n_thread)]
        self.output_queues = [queue.Queue(maxsize) for _ in range(n_thread)]
        self.workers = [
            worker_type(input_queue=in_q, output_queue=out_q)
            for (in_q, out_q) in zip(self.work_queues, self.output_queues)
        ]

    def start(self):
        for worker in self.workers:
            worker.start()
        super().start()

    def __iter__(self):
        self.start()
        for i in count():
            item = self.output_queues[i % self.n_thread].get()
            if item is StopIteration:
                # do we need to empty output_queues in order to join worker threads?
                for j in range(i + 1, i + self.n_thread):
                    self.output_queues[j % self.n_thread].get()
                break
            yield item

    def run(self):
        for i, (k, v) in enumerate(self.iterator):
            self.work_queues[i % self.n_thread].put((k, v))
        for q in self.work_queues:
            q.put(StopIteration)
        for worker in self.workers:
            worker.join()
