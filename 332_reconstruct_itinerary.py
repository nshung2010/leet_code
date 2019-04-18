"""
Given a list of airline tickets represented by pairs of departure and
arrival airports [from, to], reconstruct the itinerary in order. All
of the tickets belong to a man who departs from JFK. Thus, the
itinerary must begin with JFK.

Note:

If there are multiple valid itineraries, you should return the itinerary
that has the smallest lexical order when read as a single string.
 For example, the itinerary ["JFK", "LGA"] has a smaller lexical
 order than ["JFK", "LGB"].
All airports are represented by three capital letters (IATA code).
You may assume all tickets form at least one valid itinerary.
Example 1:

Input: [["MUC", "LHR"], ["JFK", "MUC"], ["SFO", "SJC"], ["LHR", "SFO"]]
Output: ["JFK", "MUC", "LHR", "SFO", "SJC"]
Example 2:

Input: [["JFK","SFO"],["JFK","ATL"],["SFO","ATL"],["ATL","JFK"],["ATL","SFO"]]
Output: ["JFK","ATL","JFK","SFO","ATL","SFO"]
Explanation: Another possible reconstruction is ["JFK","SFO","ATL","JFK","ATL","SFO"].
             But it is larger in lexical order.
"""
class Solution(object):
    def findItinerary(self, tickets):
        """
        :type tickets: List[List[str]]
        :rtype: List[str]
        """
        def get_the_itinerary(begin):
            """
            Get the itinerary from
            """
            res.append(begin)
            if len(res) == total_trips:
                return True
            if begin in graph:
                # print(graph[begin])
                n = len(graph[begin])
                i = 0
                while i<n:
                    neighbor = graph[begin].pop(0)
                    if get_the_itinerary(neighbor):
                        return True
                    graph[begin].append(neighbor)
                    i += 1
            res.pop()
            return False

        graph = collections.defaultdict(list)
        for beg, end in tickets:
            graph[beg].append(end)
        for airport in graph:
            graph[airport].sort()

        total_trips = len(tickets)+1
        res = []
        get_the_itinerary('JFK')
        return res
# Solution 2 is based on Hierholzer’s Algorithm
"""
void dfs(s) {
    for all the neighbors of s:
        dfs(n)
    route.add(s)
}
"""
class Solution(object):
    def findItinerary(self, tickets):
        graph = collections.defaultdict(list)
        for beg, end in tickets:
            graph[beg] += [end]
        for airport in graph:
            graph[airport] = sorted(graph[airport])[::-1]
        def visit(airport):
            while graph[airport]:
                neighbor = graph[airport].pop()
                visit(neighbor)
            route.append(airport)
        route = []
        visit('JFK')
        return route[::-1]
