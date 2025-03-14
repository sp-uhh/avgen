import heapq
import random

from functools import partial
from typing import Callable, Iterator, List, Optional, TypeVar

import torch

from torchdata.datapipes import DataChunk, functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

T_co = TypeVar("T_co", covariant=True)

def _default_len_fn(token):
    return len(token)


def _token_len_fn(token, len_fn):
    return len_fn(token), token


def _token_filter_fn(data, *, min_len, max_len):
    length, token = data
    return length >= min_len and length <= max_len


def _heappush(heap, item):
    """Push item onto heap, maintaining the heap invariant."""
    heap.append(item)
    _siftdown(heap, 0, len(heap)-1)


def _siftdown(heap, startpos, pos):
    newitem = heap[pos]
    # Follow the path to the root, moving parents down until finding a place
    # newitem fits.
    while pos > startpos:
        parentpos = (pos - 1) >> 1
        parent = heap[parentpos]
        if newitem[0] < parent[0]: # edited
            heap[pos] = parent
            pos = parentpos
            continue
        break
    heap[pos] = newitem

def _heappop(heap):
    """Pop the smallest item off the heap, maintaining the heap invariant."""
    lastelt = heap.pop()    # raises appropriate IndexError if heap is empty
    if heap:
        returnitem = heap[0]
        heap[0] = lastelt
        _siftup(heap, 0)
        return returnitem
    return lastelt

def _siftup(heap, pos):
    endpos = len(heap)
    startpos = pos
    newitem = heap[pos]
    # Bubble up the smaller child until hitting a leaf.
    childpos = 2*pos + 1    # leftmost child position
    while childpos < endpos:
        # Set childpos to index of smaller child.
        rightpos = childpos + 1
        if rightpos < endpos and not heap[childpos][0] < heap[rightpos][0]: # edited
            childpos = rightpos
        # Move the smaller child up.
        heap[pos] = heap[childpos]
        pos = childpos
        childpos = 2*pos + 1
    # The leaf at pos is empty now.  Put newitem there, and bubble it up
    # to its final resting place (by sifting its parents down).
    heap[pos] = newitem
    _siftdown(heap, startpos, pos)


@functional_datapipe("max_token_bucketize2")
class MaxTokenBucketizerIterDataPipe2(IterDataPipe[DataChunk[T_co]]):
    r"""
    Creates mini-batches of data from a min-heap with limited size, and the total length of samples
    returned by ``len_fn`` within each batch will be limited by ``max_token_count``
    (functional name: ``max_token_bucketize``). If ``min_len`` or ``max_len`` is set, the samples with
    length that is out of ``[min_len, max_len]`` will be filtered out.

    The purpose of this DataPipe is to batch samples with similar length according to ``len_fn``.
    Min-heap is used here to make sure the samples are sorted incrementally based on the length. And,
    the total length of samples in each batch is guaranteed to be smaller than ``max_token_count``.
    For an example in the audio domain, it may be batching samples with similar length. Then, given the
    ``max_token_count``, each batch may be concatenated to a Tensor with the same size and minimum padding.

    If ``include_padding`` is set to ``True``, the token count of each batch includes the padding a succeeding
    DataPipe could add. This guarentees that even after the batch is padded, ``max_token_count`` will not be exceeded.
    This can prevent out-of-memory issues for data with large variations in length.

    Note that batches are bucketized starting from the smallest size in a buffer.
    This can limit the variablity of batches if ``buffer_size`` is large.
    To increase variablity, apply ``torchdata.datapipes.iter.Shuffler`` before and after this DataPipe,
    and keep ``buffer_size`` small.


    Args:
        datapipe: Iterable DataPipe being batched
        max_token_count: Maximum length of total length of data in each batch
        len_fn: Function to be applied to each element to get lengths. ``len(data)`` is used by default.
        min_len: Optional minimum length to be included into each batch
        max_len: Optional maximum length to be included into each batch.
        buffer_size: This restricts how many samples are taken from prior DataPipe to bucketize
        include_padding: If True, the size of each batch includes the extra padding to the largest length in the batch.

    Example:
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> source_dp = IterableWrapper(['1', '11', '1', '1111', '111', '1', '11', '11', '111'])
        >>> # Using default len_fn to sort samples based on length (string length in this case)
        >>> batch_dp = source_dp.max_token_bucketize(max_token_count=5)
        >>> list(batch_dp)
        [['1', '1', '1', '11'], ['11', '11'], ['111'], ['111'], ['1111']]
        >>> batch_dp = source_dp.max_token_bucketize(max_token_count=4, buffer_size=4)
        >>> list(batch_dp)
        [['1', '1', '1'], ['11', '11'], ['11'], ['111'], ['111'], ['1111']]
    """
    datapipe: IterDataPipe[T_co]
    max_token_count: int
    len_fn: Callable
    min_len: int
    max_len: Optional[int]
    buffer_size: int

    def __init__(
        self,
        datapipe: IterDataPipe[T_co],
        max_token_count: int,
        len_fn: Callable = _default_len_fn,
        min_len: int = 0,
        max_len: Optional[int] = None,
        buffer_size: int = 1000,
        include_padding: bool = False,
    ) -> None:
        if max_len is None:
            max_len = max_token_count

        if min_len < 0 or min_len > max_len:
            raise ValueError("``min_len`` should be larger than 0 and equal to or smaller than ``max_len``.")
        if max_len > max_token_count:
            raise ValueError("``max_token_count`` must be equal to or greater than ``max_len``.")
        datapipe = datapipe.map(partial(_token_len_fn, len_fn=len_fn))
        datapipe = datapipe.filter(partial(_token_filter_fn, min_len=min_len, max_len=max_len))
        if buffer_size <= 0:
            raise ValueError("'buffer_size' is required to be a positive integer.")
        self.datapipe = datapipe
        self.max_token_count = max_token_count
        self.buffer_size = buffer_size
        self.include_padding = include_padding

    def __iter__(self) -> Iterator[DataChunk[T_co]]:
        buffer: List = []
        batch: List = []
        batch_size: int = 0
        max_length: int = 0
        for d in self.datapipe:
            # heapq.heappush(buffer, d)
            _heappush(buffer, d) # use custom heap push
            if len(buffer) == self.buffer_size:
                buffer, batch, batch_size, max_length, data_chunk = self._pop_buffer(
                    buffer, batch, batch_size, max_length
                )
                if data_chunk is not None:
                    yield data_chunk
        while buffer:
            buffer, batch, batch_size, max_length, data_chunk = self._pop_buffer(buffer, batch, batch_size, max_length)
            if data_chunk is not None:
                yield data_chunk
        if batch:
            yield DataChunk(batch)

    def _pop_buffer(self, buffer: List, batch: List, batch_size: int, max_length: int):
        data_chunk_to_yield = None
        # length, token = heapq.heappop(buffer)
        length, token = _heappop(buffer) # use custom heap pop

        if self.include_padding:
            max_length = max(length, max_length)
            new_batch_size = (len(batch) + 1) * max_length
        else:
            new_batch_size = batch_size + length

        if new_batch_size > self.max_token_count:
            data_chunk_to_yield = DataChunk(batch)
            batch = [token]
            batch_size = length
            max_length = length
        else:
            batch.append(token)
            batch_size = new_batch_size

        return buffer, batch, batch_size, max_length, data_chunk_to_yield
