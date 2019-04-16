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
