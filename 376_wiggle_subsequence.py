"""
A sequence of numbers is called a wiggle sequence if the differences
between successive numbers strictly alternate between positive and
negative. The first difference (if one exists) may be either positive
or negative. A sequence with fewer than two elements is trivially a
wiggle sequence.

For example, [1,7,4,9,2,5] is a wiggle sequence because the differences
(6,-3,5,-7,3) are alternately positive and negative. In contrast,
[1,4,7,2,5] and [1,7,4,5,5] are not wiggle sequences, the first because
its first two differences are positive and the second because its last
difference is zero.

Given a sequence of integers, return the length of the longest
subsequence that is a wiggle sequence. A subsequence is obtained by
deleting some number of elements (eventually, also zero) from the
original sequence, leaving the remaining elements in their original order.

Example 1:

Input: [1,7,4,9,2,5]
Output: 6
Explanation: The entire sequence is a wiggle sequence.
Example 2:

Input: [1,17,5,10,13,15,10,5,16,8]
Output: 7
Explanation: There are several subsequences that achieve this length.
One is [1,17,10,13,10,16,8].
Example 3:

Input: [1,2,3,4,5,6,7,8,9]
Output: 2
"""
class Solution(object):
    def wiggleMaxLength(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums) < 2:
            return len(nums)
        res = [nums[0]]
        for i in range(1, len(nums)):
            sign = (-1)**(len(res)-1)
            if nums[i]*sign > sign*res[-1]:
                res.append(nums[i])
            elif nums[i]*sign < sign*res[-1]:
                res[-1] = nums[i]
        max_length = len(res)
        res = [nums[0]]
        for i in range(1, len(nums)):
            sign = (-1)**(len(res))
            if nums[i]*sign > sign*res[-1]:
                res.append(nums[i])
            elif nums[i]*sign < sign*res[-1]:
                res[-1] = nums[i]
        max_length = max(max_length, len(res))
        return max_length

class Solution(object):
    def wiggleMaxLength(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums) < 2:
            return len(nums)
        up = [None] * len(nums)
        down = [None] * len(nums)
        up[0] = down[0] = 1
        for i in range(1, len(nums)):
            if nums[i]>nums[i-1]:
                up[i] = down[i-1] + 1
                down[i] = down[i-1]
            elif nums[i] < nums[i-1]:
                up[i] = up[i-1]
                down[i] = up[i-1]+1
            else:
                up[i] = up[i-1]
                down[i] = down[i-1]
        return max(up[-1], down[-1])

class Solution(object):
    def wiggleMaxLength(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums) < 2:
            return len(nums)
        # up = [None] * len(nums)
        #down = [None] * len(nums)
        up = down = 1
        for i in range(1, len(nums)):
            if nums[i]>nums[i-1]:
                up = down + 1
            elif nums[i] < nums[i-1]:
                down = up +1

        return max(up, down)

class Solution(object):
    def wiggleMaxLength(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums) < 2:
            return len(nums)
        # up = [None] * len(nums)
        #down = [None] * len(nums)
        pre_diff = nums[1]-nums[0]
        count = 2 if pre_diff !=0 else 1
        for i in range(2, len(nums)):
            diff = nums[i] - nums[i-1]
            if diff>0 and pre_diff <=0 or diff<0 and pre_diff >=0:
                count += 1
                pre_diff = diff
        return count
