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
        graph = collections.defaultdict(list)
        for beg, end in tickets:
            graph[beg].append(end)
        for airports in graph:
            graph[airports].sort()

        total_trips = len(tickets)
        self.ans = []
        res = ['JFK']
        num_trips = 0
        self.get_the_itinerary('JFK')
        return res
        def get_the_itinerary(begin):
            """
            Get the itinerary from
            """
            if not graph[begin]:
                return
            # print(res, begin, count, num_trips, n)

                # self.ans.append(res)
            for neighbor in graph[begin]:
                graph[begin].remove(neighbor)
                res.append(neighbor)
                num_trips += 1
                get_the_itinerary(neighbor)
                if num_trips == total_trips:
                    return
                graph[begin].add(neighbor)
                route.pop()
                num_trips -= 1


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
        self.ans = []
        res = []
        get_the_itinerary('JFK')
        return res



class Solution(object):
    def findItinerary(self, tickets):
        """
        :type tickets: List[List[str]]
        :rtype: List[str]
        """
        graph = {}
        def build_graph(tickets):
            for t in tickets:
                start, end = t
                graph[start] = graph.get(start, []) + [end]
            for A in graph:
                graph[A] = sorted(graph[A])

        def dfs(S):
            trip.append(S)
            if len(trip) == length:
                return True
            if S in graph:
                n, i = len(graph[S]), 0
                while i < n:
                    next_city = graph[S].pop(0)
                    if dfs(next_city):
                        return True
                    graph[S].append(next_city)
                    i += 1
            trip.pop()
            return False
        build_graph(tickets)
        trip, length = [], len(tickets) + 1
        dfs("JFK")
        return trip
