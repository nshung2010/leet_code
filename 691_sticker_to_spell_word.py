"""
691. Stickers to Spell Word
Hard

226

23

Favorite

Share
We are given N different types of stickers. Each sticker has a lowercase
English word on it.

You would like to spell out the given target string by cutting individual
letters from your collection of stickers and rearranging them.

You can use each sticker more than once if you want, and you have infinite
quantities of each sticker.

What is the minimum number of stickers that you need to spell out the
target? If the task is impossible, return -1.

Example 1:

Input:

["with", "example", "science"], "thehat"
Output:

3
Explanation:

We can use 2 "with" stickers, and 1 "example" sticker.
After cutting and rearrange the letters of those stickers, we can form
the target "thehat". Also, this is the minimum number of stickers
necessary to form the target string.

Example 2:

Input:

["notice", "possible"], "basicbasic"
Output:

-1
Explanation:

We can't form the target "basicbasic" from cutting letters from the given stickers.
"""


class Solution(object):
    def minStickers(self, stickers, target):
        """
        :type stickers: List[str]
        :type target: str
        :rtype: int
        """
        t_count = collections.Counter(target)
        s_count = [
            collections.Counter(sticker) & t_count for sticker in stickers
        ]
        n = len(stickers)
        for i in range(len(s_count) - 1, -1, -1):
            if any(s_count[i] == s_count[i] & s_count[j]
                   for j in range(len(s_count)) if i != j):
                s_count.pop(i)

        self.best = len(target) + 1

        def search(ans):
            # print(ans, s_count)
            if ans >= self.best:
                return
            if not s_count:
                if all(t_count[letter] <= 0 for letter in t_count):
                    self.best = ans
                return
            sticker = s_count.pop()
            used = max((t_count[letter] - 1) // sticker[letter] + 1
                       for letter in sticker)
            used = max(used, 0)
            for letter in sticker:
                t_count[letter] -= used * sticker[letter]
            search(ans + used)
            for i in range(used - 1, -1, -1):
                for letter in sticker:
                    t_count[letter] += sticker[letter]
                search(ans + i)
            s_count.append(sticker)

        search(0)
        return self.best if self.best <= len(target) else -1


# Solution 2: Backtracking: LTE
class Solution(object):
    def minStickers(self, stickers, target):
        """
        :type stickers: List[str]
        :type target: str
        :rtype: int
        """

        def backtrack(index, used):
            if used >= self.ans:
                return
            if index == N:
                self.ans = min(used, self.ans)

            if count[target[index]] <= 0:
                backtrack(index + 1, used)
            else:
                for sticker in stickers:
                    if target[index] in stick:
                        for s in stick:
                            count[s] -= 1
                        backtrack(index + 1, used + 1)
                        for s in stick:
                            count[s] += 1

        N = len(target)
        count = colelctions.Counter(target)
        self.ans = N + 1
        backtrack(0, 0)
        return self.ans if self.ans < N else -1


# Solution 3: DFS
class Solution(object):
    def minStickers(self, stickers, target):
        stickers, self.map = [
            collections.Counter(s) for s in stickers if set(s) & set(target)
        ], {}

        def dfs(target):
            if not target:
                return 0
            if target in self.map:
                return self.map[target]
            count, res = collections.Counter(target), float('inf')
            for c in stickers:
                if c[target[0]] == 0: continue  # we can make sure the 1st
                # letter will be removed to reduce the time complexity
                nxt = dfs(''.join([s * t for (s, t) in (count - c).items()]))
                if nxt != -1:
                    res = min(res, 1 + nxt)
            self.map[target] = -1 if res == float('inf') else res
            return self.map[target]

        return dfs(target)

# Solution 4: Dynamic programing:
class Solution(object):
    def minStickers(self, stickers, target):
        t_count = collections.Counter(target)
        A = [collections.Counter(sticker) & t_count
             for sticker in stickers]

        for i in range(len(A) - 1, -1, -1):
            if any(A[i] == A[i] & A[j] for j in range(len(A)) if i != j):
                A.pop(i)

        stickers = ["".join(s_count.elements()) for s_count in A]
        dp = [-1] * (1 << len(target))
        dp[0] = 0
        for state in range(1 << len(target)):
            if dp[state] == -1: continue
            for sticker in stickers:
                now = state
                for letter in sticker:
                    for i, c in enumerate(target):
                        if (now >> i) & 1: continue
                        if c == letter:
                            now |= 1 << i
                            break
                if dp[now] == -1 or dp[now] > dp[state] + 1:
                    dp[now] = dp[state] + 1

        return dp[-1]
