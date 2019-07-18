"""
Given an integer matrix, find the length of the longest increasing path.

From each cell, you can either move to four directions: left, right,
up or down. You may NOT move diagonally or move outside of the boundary
(i.e. wrap-around is not allowed).

Example 1:

Input: nums =
[
  [9,9,4],
  [6,6,8],
  [2,1,1]
]
Output: 4
Explanation: The longest increasing path is [1, 2, 6, 9].
Example 2:

Input: nums =
[
  [3,4,5],
  [3,2,6],
  [2,2,1]
]
Output: 4
Explanation: The longest increasing path is [3, 4, 5, 6]. Moving
diagonally is not allowed.
"""


class Solution(object):
    def longestIncreasingPath(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: int
        """
        if not matrix:
            return 0
        graph, incoming_deg = collections.defaultdict(
            set), collections.defaultdict(int)

        for i, row in enumerate(matrix):
            for j, cell in enumerate(row):
                if i > 0 and matrix[i - 1][j] > cell:
                    graph[(i, j)] |= {(i - 1, j)}
                    incoming_deg[(i - 1, j)] += 1
                if i + 1 < len(matrix) and matrix[i + 1][j] > cell:
                    graph[(i, j)] |= {(i + 1, j)}
                    incoming_deg[(i + 1, j)] += 1

                if j > 0 and matrix[i][j - 1] > cell:
                    graph[(i, j)] |= {(i, j - 1)}
                    incoming_deg[(i, j - 1)] += 1
                if j + 1 < len(row) and matrix[i][j + 1] > cell:
                    graph[(i, j)] |= {(i, j + 1)}
                    incoming_deg[(i, j + 1)] += 1

        max_length = 0
        S = collections.deque()
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if not incoming_deg[(i, j)]:
                    S.append(((i, j), 1))

        while S:
            cur_node, cur_length = S.popleft()
            max_length = max(max_length, cur_length)
            for adj_node in graph[cur_node]:
                incoming_deg[adj_node] -= 1
                if incoming_deg[adj_node] == 0:
                    S.append((adj_node, cur_length + 1))
        return max_length


# DP solutions
class Solution:
    def longestIncreasingPath(self, matrix):
        if not matrix:
            return 0
        res = 0
        m = len(matrix)
        n = len(matrix[0])
        dp = [[0 for i in range(n)] for j in range(m)]
        for i in range(m):
            for j in range(n):
                tem = self.dfs(matrix, dp, i, j, n, m)
                res = max(res, tem)
        return res

    def dfs(self, matrix, dp, i, j, n, m):
        if not dp[i][j]:
            dp[i][j] = 1 + max(
                self.dfs(matrix, dp, i - 1, j, n, m)
                if i - 1 >= 0 and matrix[i - 1][j] < matrix[i][j] else 0,
                self.dfs(matrix, dp, i + 1, j, n, m)
                if i + 1 < m and matrix[i + 1][j] < matrix[i][j] else 0,
                self.dfs(matrix, dp, i, j + 1, n, m)
                if j + 1 < n and matrix[i][j + 1] < matrix[i][j] else 0,
                self.dfs(matrix, dp, i, j - 1, n, m)
                if j - 1 >= 0 and matrix[i][j - 1] < matrix[i][j] else 0)
        return dp[i][j]


class Solution(object):
    def longestIncreasingPath(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: int
        """
        if not matrix or not matrix[0]:
            return 0

        row, col = len(matrix), len(matrix[0])
        num_coordinate = set(
            [(matrix[i][j], i, j) for i in xrange(row) for j in xrange(col)])
        nums_tuple = sorted(list(num_coordinate), key=lambda x: x[0])

        dp = [[1 for _ in xrange(col)] for _ in xrange(row)]
        steps = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        res = 1
        for i in xrange(len(nums_tuple)):
            cur_num, x, y = nums_tuple[i]
            for dx, dy in steps:
                new_x, new_y = x + dx, y + dy
                if new_x < 0 or new_x >= row or new_y < 0 or new_y >= col:
                    continue
                if matrix[new_x][new_y] >= cur_num:
                    continue
                dp[x][y] = max(dp[x][y], dp[new_x][new_y] + 1)
                res = max(res, dp[x][y])
        return res


# Depth First Search + Memoization
class Solution(object):
    def longestIncreasingPath(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: int
        """
        if not matrix or not matrix[0]:
            return 0

        res = float('-inf')
        row, col = len(matrix), len(matrix[0])
        memo = {}
        for i in xrange(row):
            for j in xrange(col):
                temp_len = self.check(
                    matrix, i, j,
                    memo)  # if already (i,j) in memo, directly return
                res = max(res, temp_len)
        return res

    def check(self, matrix, i, j, memo):
        if (i, j) in memo:
            return memo[(i, j)]
        max_len = 0
        row, col = len(matrix), len(matrix[0])
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            x, y = i + dx, j + dy
            if x < 0 or x >= row or y < 0 or y >= col or matrix[i][
                    j] <= matrix[x][y]:
                continue
            tmp_len = self.check(matrix, x, y, memo)
            max_len = max(tmp_len, max_len)
        memo[(i, j)] = max_len + 1
        return max_len + 1

# concise solutions
from functools import lru_cache
from itertools import product
class Solution:
    def longestIncreasingPath(self, matrix):
        if not matrix or not matrix[0]: return 0
        m, n = len(matrix), len(matrix[0])

        @lru_cache(None)
        def query(i, j):
            x = matrix[i][j]
            return 1 + max([query(ni, nj) for (ni, nj) in \
                           [(i, j+1), (i, j-1), (i+1, j), (i-1, j)] \
                           if ni >= 0 and ni < m and nj >= 0 and nj < n and matrix[ni][nj] > x], \
                           default = 0)

        return max(query(i, j) for (i, j) in product(range(m), range(n)))


