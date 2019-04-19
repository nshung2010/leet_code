"""
Suppose you have N integers from 1 to N. We define a beautiful
arrangement as an array that is constructed by these N numbers
successfully if one of the following is true for the ith position
(1 <= i <= N) in this array:

The number at the ith position is divisible by i.
i is divisible by the number at the ith position.


Now given N, how many beautiful arrangements can you construct?

Example 1:

Input: 2
Output: 2
Explanation:

The first beautiful arrangement is [1, 2]:

Number at the 1st position (i=1) is 1, and 1 is divisible by i (i=1).

Number at the 2nd position (i=2) is 2, and 2 is divisible by i (i=2).

The second beautiful arrangement is [2, 1]:

Number at the 1st position (i=1) is 2, and 2 is divisible by i (i=1).

Number at the 2nd position (i=2) is 1, and i (i=2) is divisible by 1.
"""
class Solution(object):
    def countArrangement(self, N):
        """
        :type N: int
        :rtype: int
        """
        memo = {}
        def calculate(X):
            if len(X) == 1:
                return 1
            count = 0
            if X in memo:
                return memo[X]
            n = len(X)
            for i, x in enumerate(X):
                if x%n==0 or n%x == 0:
                    new_tuple=X[:i]+X[i+1:]
                    count += calculate(new_tuple)
            memo[X] = count
            return count
        return calculate(tuple(range(1, N+1)))

# Backtracking solution:
"""
The way to understand the algorithm is that calculate (pos, visit)
is a function to add a number (from 1-N) that is current not visited
to a position "pos" so that we have beautiful arragement.
This is not a good solution, I just wanted to practice backtracking
"""
class Solution(object):
    def countArrangement(self, N):
        """
        :type N: int
        :rtype: int
        """
        def calculate(pos, visit):
            if pos>N:
                self.count += 1
                return
            for i in range(1, N+1):
                if not visit[i-1] and (pos%i==0 or i%pos==0):
                    visit[i-1] = True
                    calculate(pos+1, visit)
                    visit[i-1] = False
            return
        visit = [False] * N
        self.count = 0
        calculate(1, visit)
        return self.count
