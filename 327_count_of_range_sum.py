"""
Given an integer array nums, return the number of range sums that
lie in [lower, upper] inclusive. Range sum S(i, j) is defined as the
sum of the elements in nums between indices i and j (i ≤ j), inclusive.

Note:
A naive algorithm of O(n2) is trivial. You MUST do better than that.

Example:

Input: nums = [-2,5,-1], lower = -2, upper = 2,
Output: 3
Explanation: The three ranges are : [0,0], [2,2], [0,2] and their
respective sums are: -2, -1, 2.
"""
# Solution 1: using bisect
class Solution(object):
    def countRangeSum(self, nums, lower, upper):
        """
        :type nums: List[int]
        :type lower: int
        :type upper: int
        :rtype: int
        """
        sums, sm, count = [0], 0, 0
        for num in nums:
            sm += num
            count += bisect.bisect_right(sums, sm-lower) - bisect.bisect_left(sums, sm-upper)
            bisect.insort(sums, sm)
        return count

# Solution 2: Merge-sort and count
class Solution(object):
    def countRangeSum(self, nums, lower, upper):
        sums = [0]
        for num in nums:
            sums.append(sums[-1] + num)
        def sort(lo, hi):
            mid = (lo + hi) / 2
            if mid == lo:
                return 0
            count = sort(lo, mid) + sort(mid, hi)
            i = j = mid
            for left in sums[lo:mid]:
                while i < hi and sums[i] - left <  lower: i += 1
                while j < hi and sums[j] - left <= upper: j += 1
                count += j - i
            sums[lo:hi] = sorted(sums[lo:hi])
            return count
        return sort(0, len(sums))

# Solution 3: Similar to solution 1 but we use Binary Index Tree
# To make sure it is O(NlogN)
class BinaryIndexTree(object):
    def __init__(self, n):
        self.sums = [0] *(n+1)

    def update(self, i, val):
        while i<len(self.sums):
            self.sums[i] += val
            i += i&-i

    def sum(self, i):
        res = 0
        while i>0:
            res += self.sums[i]
            i -= i&-i
        return res

class Solution(object):

    def index(self, nums, x, left=True):
        if not left:
            return bisect.bisect_right(nums, x)
        return bisect.bisect_left(nums, x)

    def countRangeSum(self, nums, lower, upper):
        """
        :type nums: List[int]
        :type lower: int
        :type upper: int
        :rtype: int
        """
        sums = [0]
        for num in nums:
            sums.append(sums[-1]+num)
        n = len(nums)
        binary_index = BinaryIndexTree(n+1)
        sums_sorted = sorted(sums)
        res = 0
        for pre_sum in sums:
            right = self.index(sums_sorted, pre_sum-lower, left=False)
            left = self.index(sums_sorted, pre_sum-upper,left=True)
            res += binary_index.sum(right) - binary_index.sum(left)
            j = self.index(sums_sorted, pre_sum, left=False)
            binary_index.update(j, 1)
        return res

