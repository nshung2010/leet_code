"""
In the "100 game," two players take turns adding, to a running total,
any integer from 1..10. The player who first causes the running total
to reach or exceed 100 wins.

What if we change the game so that players cannot re-use integers?

For example, two players might take turns drawing from a common pool
of numbers of 1..15 without replacement until they reach a total >= 100.

Given an integer maxChoosableInteger and another integer desiredTotal,
determine if the first player to move can force a win, assuming both
players play optimally.

You can always assume that maxChoosableInteger will not be larger than
20 and desiredTotal will not be larger than 300.

Example

Input:
maxChoosableInteger = 10
desiredTotal = 11

Output:
false

Explanation:
No matter which integer the first player choose, the first player will lose.
The first player can choose an integer from 1 up to 10.
If the first player choose 1, the second player can only choose integers from 2 up to 10.
The second player will win by choosing 10 and get a total = 11, which is >= desiredTotal.
Same with other integers chosen by the first player, the second player will always win.
"""
class Solution(object):
    """
    The variable "cur" represents which numbers have been chosen
    and which are not. For example, if maxChoosableInteger = 8, we have all
    the numbers we can choose from: 1, 2, 3, 4, 5, 6, 7, 8.

    The "cur" variable lets us know the state of our number stock.
    Originally, cur = 0, or cur = 0000 0000 (in 8-bit system). This means,
    all number from 1 to 8 have not been chosen.
    At some point, let's say cur = 0000 0001, that means number 1 have been
    picked by some player. Or when cur = 0000 0101, this means number 1 and
    3 have been picked.

    And so if (cur >> i) & 1 == 0 is checking if number (i+1) has been
    picked yet or not, so we don't pick what has been picked (since the
    problem said that we pick without replacement, which means you cannot
    reuse a number). You shouldn't do what I did, just use hash set.
    Looking back at this, I don't understand why I did this and not use a
    hash set. @@
    """
    def canWin(self, maxChoosableInteger, desiredTotal, cur, d):
        if cur in d: return d[cur]
        if desiredTotal <= 0:
            d[cur] = False
            return d[cur]
        for i in range(maxChoosableInteger):
            if (cur >> i) & 1 == 0:
                if not self.canWin(maxChoosableInteger, desiredTotal - (i+1), cur + (1 << i), d):
                    d[cur] = True
                    return d[cur]
        d[cur] = False
        return d[cur]

    def canIWin(self, maxChoosableInteger, desiredTotal):
        if desiredTotal <= 0: return True
        if (maxChoosableInteger+1)*maxChoosableInteger/2 < desiredTotal: return False
        return self.canWin(maxChoosableInteger, desiredTotal, 0, {})

class Solution(object):
    def canIWin(self, max, desiredTotal):
        options = tuple(range(1,max+1))
        dp = {}
        def dfs(options, remain):
            if options[-1] >= remain or options in dp:
                return dp.setdefault(options,True)
            for i,j in enumerate(options):
                if not dfs(options[:i] + options[i+1:], remain-j):
                    return dp.setdefault(options,True)
            return dp.setdefault(options,False)
        return sum(options) >= desiredTotal and dfs(options, desiredTotal)

# This solution is more readable:
class Solution(object):
    def helper(self, allowed, target, so_far, cache):
        if len(allowed) == 0:
            return False
        state = tuple(allowed)
        if state in cache:
            return cache[state]
        else:
            cache[state] = False
            if max(allowed) + so_far >= target:
                cache[state] = True
            else:
                for x in allowed:
                    new_allowed = [y for y in allowed if x!=y]
                    if self.helper(new_allowed, target, so_far+x, cache) ==  False:
                        cache[state] = True
                        break
            return cache[state]

    def canIWin(self, maxChoosableInteger, desiredTotal):
        """
        :type maxChoosableInteger: int
        :type desiredTotal: int
        :rtype: bool
        """
        allowed = [x for x in range(1, maxChoosableInteger+1)]
        if sum(allowed) < desiredTotal:
            return False
        return self.helper(allowed, desiredTotal, 0, {})
