# 315 Counter of smaller numbers after self
"""
You are given an integer array nums and you have to return a new counts
array. The counts array has the property where counts[i] is the number
of smaller elements to the right of nums[i].

Example:

Input: [5,2,6,1]
Output: [2,1,1,0]
Explanation:
To the right of 5 there are 2 smaller elements (2 and 1).
To the right of 2 there is only 1 smaller element (1).
To the right of 6 there is 1 smaller element (1).
To the right of 1 there is 0 smaller element.
"""
class Solutions(object):
    # Solution one is based on a bisect model. We start insert the number
    # from the end of the list to the begining of the list. Everytime
    # the bisect_left of a new number into a sorted list (binary_list)
    # is the answer. In the end we need to reverse it because we are working
    # from the right to the left
    def count_smaller_sol1(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        if not nums: return []
        res = [0]
        binary_list = [nums[-1]]
        for i in reversed(range(len(nums)-1)):
            bisect.insort_left(binary_list, nums[i])
            res += bisect.bisect_left(binary_list, nums[i])
        return res[::-1]

    # Solution two is based on the observation that: the smaller number
    # on the right of a number are exactly those that jump from its right
    # to its left during a Merge sort. So this solution is basically
    # a merge sort with added tracking of those right to left jumps
    def count_smaller_sol2(self, nums):
        def sort(enum):
            half = len(enum)//2
            if half:
                left, right = sort(enum[:half]), sort(enum[half:])
                for i in range(len(enum))[::-1]:
                    if not right or left and left[-1][1]>right[-1][1]:
                        # the smaller is used to keep track of those
                        # that jump from its right to its left.
                        smaller[left[-1][0]] += len(right)
                        enum[i] = left.pop()
                    else:
                        enum[i] = right.pop()
            return enum

        smaller = [0] * len(nums)
        sort(list(enumerate(nums)))
        return smaller
