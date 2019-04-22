"""
Given a non-empty 2D matrix matrix and an integer k, find the max sum
of a rectangle in the matrix such that its sum is no larger than k.

Example:

Input: matrix = [[1,0,1],[0,-2,3]], k = 2
Output: 2
Explanation: Because the sum of rectangle [[0, 1], [-2, 3]] is 2,
             and 2 is the max number no larger than k (k = 2).
Note:

The rectangle inside the matrix must have an area > 0.
What if the number of rows is much larger than the number of columns?
"""


# The idea here is that we need to convert back to the count of range sum
# i.e prob 327.
class Solution(object):
    def maxSumSubmatrix(self, matrix, k):
        """
        :type matrix: List[List[int]]
        :type k: int
        :rtype: int
        """
        if not matrix:
            return 0
        num_row = len(matrix)
        num_col = len(matrix[0])
        small_dim = min(num_row, num_col)
        big_dim = max(num_row, num_col)
        max_sum = float('-inf')
        for x0 in range(small_dim):
            sums = [0] * big_dim
            for x1 in range(x0, small_dim):
                for y in range(0, big_dim):
                    num = matrix[x1][y] if small_dim == num_row else matrix[y][
                        x1]
                    sums[y] += num
                max_sum = max(max_sum, self.max_sum_array(sums, k))
        return max_sum

    def max_sum_array(self, nums, k):
        """
        :type matrix: List[List[int]]
        :type k: int
        :rtype: int
        """
        max_sum = float('-inf')
        sums = [0]
        cum_sum = 0
        for i, n in enumerate(nums, 1):
            cum_sum += n
            index = bisect.bisect_left(sums, cum_sum - k)
            if index != i:
                max_sum = max(max_sum, cum_sum - sums[index])
            bisect.insort(sums, cum_sum)
        return max_sum


# Another solution (similar idea)
from bisect import bisect_left, insort


class Solution:
    def maxSumSubmatrix(self, matrix, k):
        dim1 = len(matrix)
        dim2 = len(matrix[0])
        # determine which dimension is larger
        small_dim = min(dim1, dim2)
        large_dim = max(dim1, dim2)
        max_sum = float('-inf')
        # iterate through the small dimension
        for x0 in range(small_dim):
            # keep track of cumulative sums over the smaller dimension
            small_sums = [0]*large_dim
            for x1 in range(x0, small_dim):
                # this will hold the cummulative sum across the small
                # dimension as we traverse the large dimension
                cumm_sum = 0
                # this holds the previous cummulative sums and 0;
                #we maintain it in sorted order for efficiency
                prev_sums = [0]
                for y in range(large_dim):
                    # as we traverse the larger dimension, accumlate
                    #the sums over the small dimension
                    small_sums[y] += matrix[x1][
                        y] if dim1 == small_dim else matrix[y][x1]
                    # running total of the small_sums
                    cumm_sum += small_sums[y]
                    # we are looking for a previous cummulative
                    # sum (prev_sums[i]) such that
                    #     cumm_sum - prev_sums[i] <= k
                    # which is equivalent to finding i where
                    #     cumm_sum - k <= prev_sums[i]
                    #
                    # If the i's were in order, the difference cumm_sum
                    # - prev_sums[i] would be the sum of the rectangle
                    # from i + 1 through y on the larger dimension.
                    # We are not required to find that hypothetical i
                    # value, but only the sum of the rectangle.
                    # Accordingly, we don't worry about the position
                    # order in the matrix, but rather just keep the
                    # cummulative sums in sorted order so we can quickly
                    # find one that meets our criteria, that is, some
                    # rectangle's area is less than or equal to k.
                    #
                    # We started with a "dummy" previous sum of 0.
                    # If we find that value in our search, it means that
                    # the rectangle from x0 to x1 inclusive and up to y
                    # inclusive has the sum we are looking for.
                    #
                    # bisect_left does a binary search over the prev_sums list
                    i = bisect_left(prev_sums, cumm_sum - k)
                    # if our search would go past the end of the list,
                    # then there is no i such that prev_sums[i]
                    # satisfies our criteria
                    if i < len(prev_sums):
                        max_sum = max(max_sum, cumm_sum - prev_sums[i])
                        # optimization:  if we have a rectange that sums
                        # to k, we are done
                        if max_sum == k:
                            return max_sum
                    # add the cummulative sum to prev_sums, maintaining
                    # the order of prev_sums
                    insort(prev_sums, cumm_sum)
        return max_sum
