import cudf
import cugraph

# Create a graph using cuGraph
G = cugraph.Graph()
df = cudf.DataFrame({'src': [0, 1, 2, 0], 'dst': [1, 2, 3, 3]})
G.add_edges_from(df)

# Perform BFS on the graph
bfs_result = cugraph.bfs(G, 0)
print(bfs_result)