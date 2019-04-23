"""
You have a number of envelopes with widths and heights given as a pair
of integers (w, h). One envelope can fit into another if and only if
both the width and height of one envelope is greater than the width and
height of the other envelope.

What is the maximum number of envelopes can you Russian doll? (put one
inside other)

Note:
Rotation is not allowed.

Example:

Input: [[5,4],[6,4],[6,7],[2,3]]
Output: 3
Explanation: The maximum number of envelopes you can Russian doll is 3
([2,3] => [5,4] => [6,7]).
"""


class Solution(object):
    def maxEnvelopes(self, envelopes):
        """
        :type envelopes: List[List[int]]
        :rtype: int
        """
        if not envelopes:
            return 0
        envelopes = sorted(envelopes, key=lambda x: (x[0], -x[1]))
        envelopes_height = [x[1] for x in envelopes]
        return self.LIS(envelopes_height)

    def LIS(self, nums):
        """
        Longest increasing sequences
        """

        def binary_search(l, r, nums, target):
            while r - l > 1:
                m = l + (r - l) // 2
                if nums[m] >= target:
                    r = m
                else:
                    l = m
            return r

        if not nums:
            return 0

        size = len(nums)
        tail_list = [0] * (size + 1)
        LIS = 0
        tail_list[0] = nums[0]
        for i in range(size):
            if nums[i] < tail_list[0]:
                tail_list[0] = nums[i]
            elif nums[i] > tail_list[LIS - 1]:
                tail_list[LIS] = nums[i]
                LIS += 1
            else:
                tail_list[binary_search(-1, LIS - 1, tail_list,
                                        nums[i])] = nums[i]
        return LIS

# Another solution:
class Solution:
    def maxEnvelopes(self, envelopes):
        tails = []
        for w, h in sorted(envelopes, key = lambda x: (x[0], -x[1])):
            i = bisect.bisect_left(tails, h)
            if i == len(tails): tails.append(h)
            else: tails[i] = h
        return len(tails)
