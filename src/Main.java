import java.util.*;

public class Main {
    public static void main(String[] args){
    }
}
class GraphTheory {
    static class Dijkstras {
        static class AdjacencyMatrix {
            public static int[] dijkstra(int[][] map, int src) {
                int[] distances = new int[map.length];
                HashSet<Integer> visited = new HashSet<>();
                for (int i = 0; i < distances.length; i++)
                    distances[i] = Integer.MAX_VALUE;
                PriorityQueue<Integer> queue = new PriorityQueue<>((i1, i2) -> Integer.compare(distances[i1], distances[i2]));
                queue.add(0);
                distances[src] = 0;
                for (int count = 0; count < distances.length - 1; count++) {
                    int from = queue.poll();
                    if (distances[from] == Integer.MAX_VALUE || visited.contains(from))
                        continue;
                    visited.add(from);
                    for (int to = 0; to < map[from].length; to++)
                        if (map[from][to] != 0 && !visited.contains(to) && distances[to] > distances[from] + map[from][to]) {
                            distances[to] = distances[from] + map[from][to];
                            queue.add(to);
                        }
                }
                return distances;
            }
        }
        static class AdjacencyList {
            public static int[] dijkstra(HashMap<Integer, HashMap<Integer, Integer>> map, int src) {
                int[] distances = new int[map.size()];
                HashSet<Integer> visited = new HashSet<>();
                for (int i = 0; i < distances.length; i++)
                    distances[i] = Integer.MAX_VALUE;
                PriorityQueue<Integer> queue = new PriorityQueue<>((i1, i2) -> Integer.compare(distances[i1], distances[i2]));
                queue.add(0);
                distances[src] = 0;
                for (int count = 0; count < distances.length - 1; count++){
                    int from = queue.poll();
                    if(distances[from] == Integer.MAX_VALUE || visited.contains(from))
                        continue;
                    visited.add(from);
                    for (int to : map.get(from).keySet())
                        if (!visited.contains(to) && distances[to] > distances[from] + map.get(from).get(to)) {
                            distances[to] = distances[from] + map.get(from).get(to);
                            queue.add(to);
                        }
                }
                return distances;
            }
        }
    }
    static class BellmanFord {
        static class AdjacencyMatrix {
            public static int[] bellmanFord(int[][] map, int src) {
                int[] distances = new int[map.length];
                for (int i = 0; i < distances.length; i++)
                    distances[i] = Integer.MAX_VALUE;
                distances[src] = 0;
                for (int count = 0; count < distances.length - 1; count++) {
                    for (int start = 0; start < map.length; start++) {
                        for (int end = 0; end < map.length; end++) {
                            if (map[start][end] != 0 && distances[start] != Integer.MAX_VALUE && distances[end] > distances[start] + map[start][end])
                                distances[end] = distances[start] + map[start][end];
                        }
                    }
                }
                return distances;
            }
            public static boolean checkNegativeCycle(int[] distances, int[][] map){
                //If it can get better after another iteration when it's already had n - 1 iterations, then there's a negative loop!
                for (int start = 0; start < map.length; start++) {
                    for (int end = 0; end < map.length; end++) {
                        if (map[start][end] != 0 && distances[start] != Integer.MAX_VALUE && distances[end] > distances[start] + map[start][end])
                            return true;
                    }
                }
                return false;
            }
        }
        static class AdjacencyList {
            //Mmmmm not quite bellman ford... I went ahead and did a pseudo-breadth first search
            //to optimize it a bit. Still got the same underlying algorithm and worst case runtime
            //I just told it to only try spreading a node's weight if it's been updated.
            public static int[] bellmanFord(HashMap<Integer, HashMap<Integer, Integer>> map, int src) {
                int[] distances = new int[map.size()];
                for (int i = 0; i < distances.length; i++)
                    distances[i] = Integer.MAX_VALUE;
                LinkedHashSet<Integer> queue = new LinkedHashSet<>();
                queue.add(0);
                distances[src] = 0;
                for (int count = 0; count < distances.length - 1; count++) {
                    LinkedHashSet<Integer> nextQueue = new LinkedHashSet<>();
                    for (Integer start : queue) {
                        for (int end : map.get(start).keySet()) {
                            if (distances[end] > distances[start] + map.get(start).get(end)) {
                                distances[end] = distances[start] + map.get(start).get(end);
                                nextQueue.add(end);
                            }
                        }
                    }
                }
                return distances;
            }
            public static boolean checkNegativeCycle(int[] distances, HashMap<Integer, HashMap<Integer, Integer>> map){
                //If it can get better after another iteration when it's already had n - 1 iterations, then there's a negative loop!
                for (int start : map.keySet()) {
                    for (int end : map.get(start).keySet()) {
                        if (distances[start] != Integer.MAX_VALUE && distances[end] > distances[start] + map.get(start).get(end))
                            return true;
                    }
                }
                return false;
            }
        }
    }
    static class FloydWarshall {
        public static int[][] solve(int[][] graph){
            int dist[][] = new int[graph.length][graph.length];
            int i, j, k;
            for(i = 0; i < graph.length; i++)
                for(j = 0; j < graph.length; j++)
                    dist[i][j] = graph[i][j];
            for(k = 0; k < graph.length; k++){
                for(i = 0; i < graph.length; i++){
                    for(j = 0; j < graph.length; j++){
                        if(dist[i][k] + dist[k][j] < dist[i][j])
                            dist[i][j] = dist[i][k] + dist[k][j];
                    }
                }
            }
            return dist;
        }
    }
    static class Prims {
        static class AdjacencyMatrix {
            static class Edge {
                int from, to, cost;
                public Edge(int from, int to, int cost){
                    this.from = from;
                    this.to = to;
                    this.cost = cost;
                }
            }
            public static Edge[] prims(int[][] graph){
                int[] mst = new int[graph.length];
                Edge[] edges = new Edge[graph.length - 1];
                Arrays.fill(mst, -1);
                mst[0] = 0;
                PriorityQueue<Edge> queue = new PriorityQueue<>((c1, c2) -> Integer.compare(c1.cost, c2.cost));
                int from = 0;
                for(int itrs = 0; itrs < graph.length - 1; itrs++){
                    for(int to = 0; to < graph[from].length; to++)
                        if(graph[from][to] != 0)
                            queue.add(new Edge(from, to, graph[from][to]));
                    Edge e;
                    while(mst[(e = queue.poll()).to] != -1);
                    mst[e.to] = from;
                    from = e.to;
                    edges[itrs] = e;
                }
                return edges;
            }
        }
        static class AdjacencyList {
            static class Edge {
                int from, to, cost;
                public Edge(int from, int to, int cost){
                    this.from = from;
                    this.to = to;
                    this.cost = cost;
                }
            }
            public static Edge[] prims(HashMap<Integer, HashMap<Integer, Integer>> graph){
                int[] mst = new int[graph.size()];
                Edge[] edges = new Edge[graph.size() - 1];
                Arrays.fill(mst, -1);
                mst[0] = 0;
                PriorityQueue<Edge> queue = new PriorityQueue<>((c1, c2) -> Integer.compare(c1.cost, c2.cost));
                int from = 0;
                for(int itrs = 0; itrs < mst.length - 1; itrs++){
                    for(int to : graph.get(from).keySet())
                        queue.add(new Edge(from, to, graph.get(from).get(to)));
                    Edge e;
                    while(mst[(e = queue.poll()).to] != -1);
                    mst[e.to] = from;
                    from = e.to;
                    edges[itrs] = e;
                }
                return edges;
            }
        }
    }
    static class Kruskals {
        static class AdjacencyMatrix {
            static class Edge {
                int from, to, cost;
                public Edge(int from, int to, int cost) {
                    this.from = from;
                    this.to = to;
                    this.cost = cost;
                }
            }
            public static Edge[] kruskals(int[][] graph) {
                Other.UnionFind sets = new Other.UnionFind(graph.length);
                Edge[] mst = new Edge[graph.length - 1];
                PriorityQueue<Edge> edges = new PriorityQueue<>((e1, e2) -> Integer.compare(e1.cost, e2.cost));
                for(int i = 0; i < graph.length; i++){
                    for(int j = 0; j < graph.length; j++){
                        if(graph[i][j] > 0)
                            edges.add(new Edge(i, j, graph[i][j]));
                    }
                }
                int edgeNum = 0;
                while(edgeNum < mst.length){
                    Edge e = edges.poll();
                    if(sets.find(e.from) == sets.find(e.to))
                        continue;
                    mst[edgeNum++] = e;
                    sets.union(e.from, e.to);
                }
                return mst;
            }
        }
        static class AdjacencyList {
            static class Edge {
                int from, to, cost;
                public Edge(int from, int to, int cost) {
                    this.from = from;
                    this.to = to;
                    this.cost = cost;
                }
            }
            public static Edge[] kruskals(HashMap<Integer, HashMap<Integer, Integer>> graph) {
                Other.UnionFind sets = new Other.UnionFind(graph.size());
                Edge[] mst = new Edge[graph.size() - 1];
                PriorityQueue<Edge> edges = new PriorityQueue<>((e1, e2) -> Integer.compare(e1.cost, e2.cost));
                for(int i : graph.keySet()){
                    for(int j : graph.get(i).keySet()){
                        edges.add(new Edge(i, j, graph.get(i).get(j)));
                    }
                }
                int edgeNum = 0;
                while(edgeNum < mst.length){
                    Edge e = edges.poll();
                    if(sets.find(e.from) == sets.find(e.to))
                        continue;
                    mst[edgeNum++] = e;
                    sets.union(e.from, e.to);
                }
                return mst;
            }
        }
    }
    static class FordFulkerson {
        static class AdjacencyMatrix {
            public static boolean bfs(int[][] rGraph, int s, int t, int[] parent) {
                boolean[] visited = new boolean[rGraph.length];
                for(int i = 0; i < rGraph.length; i++)
                    visited[i] = false;
                LinkedList<Integer> queue = new LinkedList<>();
                queue.add(s);
                visited[s] = true;
                parent[s] = -1;
                while (queue.size() != 0) {
                    int from = queue.poll();
                    for (int to = 0; to < rGraph.length; to++) {
                        if (!visited[to] && rGraph[from][to] > 0) {
                            queue.add(to);
                            parent[to] = from;
                            visited[to] = true;
                        }
                    }
                }
                return visited[t];
            }
            public static int fordFulkerson(int graph[][], int s, int t) {
                int from, to;
                int[][] rGraph = new int[graph.length][graph.length];
                for (from = 0; from < graph.length; from++)
                    for (to = 0; to < graph.length; to++)
                        rGraph[from][to] = graph[from][to];
                int[] parent = new int[graph.length];
                int maxFlow = 0;
                while (bfs(rGraph, s, t, parent)) {
                    int pathFlow = Integer.MAX_VALUE;
                    for (to = t; to != s; to = parent[to]) {
                        from = parent[to];
                        pathFlow = Math.min(pathFlow, rGraph[from][to]);
                    }
                    for (to = t; to != s; to=parent[to]) {
                        from = parent[to];
                        rGraph[from][to] -= pathFlow;
                        rGraph[to][from] += pathFlow;
                    }
                    maxFlow += pathFlow;
                }
                return maxFlow;
            }
        }
    }
}
class Other {
    static class UnionFind {
        private int[] parent;
        private int[] rank;
        public int find(int i) {
            int p = parent[i];
            return (i == p) ? i : (parent[i] = find(p));
        }
        public void union(int i, int j) {
            int root1 = find(i);
            int root2 = find(j);
            if (root2 == root1) return;
            if (rank[root1] > rank[root2]) {
                parent[root2] = root1;
            } else if (rank[root2] > rank[root1]) {
                parent[root1] = root2;
            } else {
                parent[root2] = root1;
                rank[root1]++;
            }
        }
        public UnionFind(int max) {
            parent = new int[max];
            rank = new int[max];
            for (int i = 0; i < max; i++) {
                parent[i] = i;
            }
        }
    }
}