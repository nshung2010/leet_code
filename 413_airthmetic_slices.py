"""
A sequence of number is called arithmetic if it consists of at least
three elements and if the difference between any two consecutive
elements is the same.

For example, these are arithmetic sequence:

1, 3, 5, 7, 9
7, 7, 7, 7
3, -1, -5, -9
The following sequence is not arithmetic.

1, 1, 2, 5, 7

A zero-indexed array A consisting of N numbers is given. A slice of
that array is any pair of integers (P, Q) such that 0 <= P < Q < N.

A slice (P, Q) of array A is called arithmetic if the sequence:
A[P], A[p + 1], ..., A[Q - 1], A[Q] is arithmetic. In particular,
this means that P + 1 < Q.

The function should return the number of arithmetic slices in the array A.


Example:

A = [1, 2, 3, 4]

return: 3, for 3 arithmetic slices in A: [1, 2, 3], [2, 3, 4] and
[1, 2, 3, 4] itself.
"""
class Solution(object):
    def numberOfArithmeticSlices(self, A):
        """
        :type A: List[int]
        :rtype: int
        """
        def is_arithmetic(nums):
            """
            checking arithmetic for an array of length 3
            """
            return nums[2]-nums[1] == nums[1]-nums[0]
        def count_arithmetic_from_arithmetic_series(N):
            """
            count all of posible sub arithmetic that has length N
            So if first index start fom 0, we will have total N-3
            airthmetic (associcated with second index go from 2 to N-1)
            if first index start from 1, we will have N-2
            ....
            so total = (N-2) + (N-4) + ... + 1
            """
            return (N-2)*(N-3)//2

        if len(A) < 3:
            return 0
        p, q = 0, 2
        count = 0
        N = len(A)
        while p<N and q <N:
            if q-p>=2 and is_arithmetic(A[p:q+1]):
                count += 1
                while q+1<N and A[q+1]-A[q] == A[q]-A[q-1]:
                    q += 1
                    count += 1
                    # print(p, q, count)
                count += count_arithmetic_from_arithmetic_series(q+1-p)
                p = q-2
            if q == N-1 and p < N-3:
                p+=1

            else:
                p += 1
                q += 1
        return count

# Recursion:
"""
Observation: if A[i] - A[i-1] == A[i-1] - A[i-2] then we have additional
1+x arithmetric where x = slice(A, i-1) because (1, i) ...
(i-2, i) can be map to (0, i-1), (1, i-1), ..., (i-3, i-1) which count
equal to slice (A, i-1) = x. 1 more arithmetic is (0, i)
"""
class Solution(object):
    def numberOfArithmeticSlices(self, A):
        """
        :type A: List[int]
        :rtype: int
        """
        self.count = 0
        def slices(A, i):
            """
            count number of arithmetic of matrix from A[0:i+1]

            """
            if i <2:
                return 0
            ap = 0
            if (A[i] - A[i-1] == A[i-1] - A[i-2]):
                ap = 1 + slices(A, i-1)
                self.count += ap
            else:
                slices(A, i-1)
            return ap


        slices(A, len(A)-1)
        return self.count

# Dynamic programing:
class Solution(object):
    def numberOfArithmeticSlices(self, A):
        """
        :type A: List[int]
        :rtype: int
        """
        dp = [0] * len(A)
        count = 0
        for i in range(2, len(A)):
            if A[i]-A[i-1] == A[i-1]-A[i-2]:
                dp[i] = 1+ dp[i-1]
                count += dp[i]
        return count
#
class Solution(object):
    def numberOfArithmeticSlices(self, A):
        """
        :type A: List[int]
        :rtype: int
        """
        dp = 0
        count = 0
        for i in range(2, len(A)):
            if A[i]-A[i-1] == A[i-1]-A[i-2]:
                dp = 1+dp
                count += dp
            else:
                dp=0
        return count
