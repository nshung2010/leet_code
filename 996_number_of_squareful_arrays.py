"""
Given an array A of non-negative integers, the array is squareful if
for every pair of adjacent elements, their sum is a perfect square.

Return the number of permutations of A that are squareful.
Two permutations A1 and A2 differ if and only if there is some index
i such that A1[i] != A2[i].


Example 1:

Input: [1,17,8]
Output: 2
Explanation:
[1,8,17] and [17,8,1] are the valid permutations.
Example 2:

Input: [2,2,2]
Output: 1
"""

# My solution:
class Solution(object):
    def numSquarefulPerms(self, A):
        """
        :type A: List[int]
        :rtype: int
        """
        def backtrack(index, used, current_permu):
            # print(index, current_permu, memo)
            count = 0
            if index == len(A):
                count += 1
                memo[tuple(current_permu)] = count
                return count
            if tuple(current_permu) in memo:
                count = memo[tuple(current_permu)]
                return count
            val_used = set()
            for i, num in enumerate(sorted(A)):

                temp = current_permu[-1]
                if num not in val_used and not used[i] and is_squareful(temp+A[i]):
                    used[i] = True
                    new_permu = current_permu + [A[i]]
                    if tuple(new_permu) in memo:
                        count += memo[tuple(new_permu)]
                    else:
                        nxt = backtrack(index+1, used, new_permu)
                        count += nxt
                    val_used.add(num)
                    used[i] = False
            memo[tuple(current_permu)]  = count
            return count

        def is_squareful(num):
            return int(num**0.5) == num**0.5
        memo = {}
        A.sort()
        used = [False] * len(A)
        self.ans = 0
        for i, num in enumerate(A):
            if i>0 and A[i] == A[i-1]:
                continue
            used[i] = True
            self.ans += backtrack(1, used, [num])
            used[i] = False
        return self.ans

# Nice solution
class Solution:
    def numSquarefulPerms(self, A):
        res = []
        A.sort()
        self.helper(A, [], res)
        return len(res)

    def helper(self, A, curr, res):
        if not A:
            res.append(curr)
            return
        for i in range(len(A)):
            if i > 0 and A[i] == A[i-1]:
                continue
            if len(curr) == 0 or math.sqrt(curr[-1]+A[i]) % 1 == 0:
                self.helper(A[:i]+A[i+1:], curr+[A[i]], res)

# Another solution using backtrack
class Solution(object):
    def numSquarefulPerms(self, A):
        N = len(A)
        count = collections.Counter(A)

        graph = {x: [] for x in count}
        for x in count:
            for y in count:
                if int((x+y)**.5 + 0.5) ** 2 == x+y:
                    graph[x].append(y)

        def dfs(x, todo):
            count[x] -= 1
            if todo == 0:
                ans = 1
            else:
                ans = 0
                for y in graph[x]:
                    if count[y]:
                        ans += dfs(y, todo - 1)
            count[x] += 1
            return ans

        return sum(dfs(x, len(A) - 1) for x in count)
