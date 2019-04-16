"""
Given an unsorted array nums, reorder it such that
nums[0] < nums[1] > nums[2] < nums[3]....

Example 1:

Input: nums = [1, 5, 1, 1, 6, 4]
Output: One possible answer is [1, 4, 1, 5, 1, 6].
Example 2:

Input: nums = [1, 3, 2, 2, 3, 1]
Output: One possible answer is [2, 3, 1, 3, 1, 2].
"""
class Solution:
    def wiggleSort_1(self, nums):
        arr = sorted(nums)
        for i in range(1, len(nums), 2): nums[i] = arr.pop()
        for i in range(0, len(nums), 2): nums[i] = arr.pop()

    def wiggleSort_2(self, nums):
        nums.sort()
        half = len(nums[::2])
        nums[::2], nums[1::2] = nums[:half][::-1], nums[half:][::-1]

# Using the Dutch national flags algorithm
# First we need to find the median values and then sort it based on
# All number in odd index < median and all number in even index > median
# To get median we use quick sort algorithm with is average O(n) but can
# be O(n^2)
class Solution(object):
    def wiggleSort(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        if len(nums)<2:
            return nums
        mid = self.get_k_th_num(nums, 0, len(nums)-1, len(nums)//2)
        print(mid)
        n = len(nums)
        i=j=(n-1)//2*2
        k = 1
        for _ in range(n):
            if nums[j] < mid:
                nums[i], nums[j] = nums[j], nums[i]
                i -= 2
                j -= 2
                if j<0: j=n//2*2-1
            elif nums[j] > mid:
                # print(j, mid)
                nums[j], nums[k] = nums[k], nums[j]
                k += 2
            else:
                j -= 2
                if j<0: j=n//2*2-1
        return nums
    def get_k_th_num(self, nums, start, end, k):
        if start==end:
            return nums[start]
        mid = self.partition(start, end, nums)
        if mid == k:
            return nums[k]
        elif mid > k:
            return self.get_k_th_num(nums, start, mid-1, k)
        else:
            return self.get_k_th_num(nums, mid+1, end, k)

    def partition(self, start, end, nums):
        pivot_index = random.randrange(start, end+1)
        pivot = nums[pivot_index]
        nums[end], nums[pivot_index] = nums[pivot_index], nums[end]
        mid = start
        for i in range(start, end):
            if nums[i] >= pivot:
                nums[mid], nums[i] = nums[i], nums[mid]
                mid += 1
        nums[mid], nums[end] = nums[end], nums[mid]
        return mid


