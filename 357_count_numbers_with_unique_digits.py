"""
Given a non-negative integer n, count all numbers with unique digits, x,
where 0 ≤ x < 10n.

Example:

Input: 2
Output: 91
Explanation: The answer should be the total numbers in the range of 0 ≤ x < 100,
             excluding 11,22,33,44,55,66,77,88,99

"""
class Solution(object):
    def countNumbersWithUniqueDigits(self, n):
        """
        :type n: int
        :rtype: int
        """

        if n==0:
            return 1
        # res is the number of unique digits which has n digits
        # i.e. the first digit must be > 0
        res = 9
        for i in range(n-1):
            res *= 9-i
        return res + self.countNumbersWithUniqueDigits(n-1)
"""
Another solution is using backtracking. It is slow but I wanted to practice
backtracking
"""
class Solution(object):
    def countNumbersWithUniqueDigits(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n==0:
            return 1
        def back_tracking(current_number, visit):
            # print(current_number)
            count = 0
            if current_number > max_num:
                return count
            else:
                count += 1
            for i in range(0, 10):
                if not visit[i]:
                    new = 10*current_number + i
                    visit[i] = True
                    if new<max_num:
                        count += back_tracking(new, visit)
                    visit[i] = False
            return count

        count = 1 # for number 0
        visit = {i: False for i in range(10)}
        max_num = 10**n
        for i in range(1, 10):
            visit[i] = True
            count += back_tracking(i, visit)
            visit[i] = False

        return count
