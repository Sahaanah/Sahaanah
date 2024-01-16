# Importing the necessary packages
using DataFrames, CSV

# Loading the LinkedIn dataset
file_path = "dataset_updated.csv"
f = CSV.read(file_path, DataFrame) 
f


using CSV
using DataFrames
using Graphs
using GraphPlot

# Read the CSV file into a DataFrame
f = CSV.read("dataset_updated.csv", DataFrame)

# Create a unique list of names from the 'Source' and 'Target' columns
h = unique(vcat(f[!, 1], f[!, 2]))

# Create a dictionary to map names to integers
ed = Dict(name => count for (count, name) in enumerate(h))

# Create a list of edges using the integer mapping
edges = [(ed[row.source], ed[row.Target]) for row in eachrow(f)]

# Create a graph
G = SimpleGraph(length(h))

# Add edges to the graph
for edge in edges
    add_edge!(G, edge[1], edge[2])
end

# Plot the graph
gplot(G, nodelabel=1:length(h))


# Create a DataFrame to show the mapping between names and ID numbers
id_mapping_df = DataFrame(ID = 1:length(h), Name = h)

# Display the DataFrame
println(id_mapping_df)

# Example: Filter data for the year 2023
data_2020 = filter(row -> row["Connected On"] == 2020, f)

# Filter data for both 2020 and 2021
data_2020_2021 = filter(row -> row["Connected On"] in [2020, 2021], f)

# Filter data for 2020,2021,2022
data_2020_2021_2022 = filter(row -> row["Connected On"] in [2020, 2021,2022], f)

# Filter data for 2020,2021,2022,2023
data_2020_2021_2022_2023 = filter(row -> row["Connected On"] in [2020, 2021,2022,2023], f)

# Function to create a graph from filtered data
function create_graph(data)
    h = unique(vcat(data[!, 1], data[!, 2]))
    ed = Dict(name => count for (count, name) in enumerate(h))
    edges = [(ed[row.source], ed[row.Target]) for row in eachrow(data)]
    G = SimpleGraph(length(h))

    for edge in edges
        add_edge!(G, edge[1], edge[2])
    end
    return G, h
end

function plotgraph(G,h)
    gplot(G, nodelabel=1:length(h))
end


# Function to calculate degree centrality of a graph
function degree_centrality(graph)
    num_nodes = nv(graph)
    centrality = Dict{Int, Float64}()

    for node in 1:num_nodes
        centrality[node] = degree(graph, node)
    end

    return centrality
end



# Assuming you already have data_2023 defined
G_2020, labels_2020 = create_graph(data_2020)
graph2023 = plotgraph(G_2020, labels_2020)


using DataFrames

# Assuming degree_centrality_2023 is a dictionary
degree_centrality_2020 = degree_centrality(G_2020)

# Flatten the dictionary
flattened_data = [(node, centrality) for (nodes, centrality) in zip(keys(degree_centrality_2020), values(degree_centrality_2020)) for node in nodes]

# Create a DataFrame
df_degree_centrality_2020 = DataFrame(flattened_data, [:Node, :DegreeCentrality])

# Print the DataFrame
println(df_degree_centrality_2020)


# Assuming you already have data_2023 defined
G_2020_2021, labels_2020_2021 = create_graph(data_2020_2021)
graph2023 = plotgraph(G_2020_2021, labels_2020_2021)


using DataFrames

# Assuming degree_centrality_2023 is a dictionary
degree_centrality_2020_2021 = degree_centrality(G_2020_2021)

# Flatten the dictionary
flattened_data = [(node, centrality) for (nodes, centrality) in zip(keys(degree_centrality_2020_2021), values(degree_centrality_2020_2021)) for node in nodes]

# Create a DataFrame
df_degree_centrality_2020_2021 = DataFrame(flattened_data, [:Node, :DegreeCentrality])

# Print the DataFrame
println(df_degree_centrality_2020_2021)

# Assuming you already have data_2023 defined
G_2020_2021_2022, labels_2020_2021_2022 = create_graph(data_2020_2021_2022)
graph2023 = plotgraph(G_2020_2021_2022, labels_2020_2021_2022)


using DataFrames

# Assuming degree_centrality_2023 is a dictionary
degree_centrality_2020_2021_2022 = degree_centrality(G_2020_2021_2022)

# Flatten the dictionary
flattened_data = [(node, centrality) for (nodes, centrality) in zip(keys(degree_centrality_2020_2021_2022), values(degree_centrality_2020_2021_2022)) for node in nodes]

# Create a DataFrame
df_degree_centrality_2020_2021_2022 = DataFrame(flattened_data, [:Node, :DegreeCentrality])

# Print the DataFrame
println(df_degree_centrality_2020_2021_2022)

# Assuming you already have data_2023 defined
G_2020_2021_2022_2023, labels_2020_2021_2022_2023 = create_graph(data_2020_2021_2022_2023)
graph2023 = plotgraph(G_2020_2021_2022_2023, labels_2020_2021_2022_2023)

using DataFrames

# Assuming degree_centrality_2023 is a dictionary
degree_centrality_2020_2021_2022_2023 = degree_centrality(G_2020_2021_2022_2023)

# Flatten the dictionary
flattened_data = [(node, centrality) for (nodes, centrality) in zip(keys(degree_centrality_2020_2021_2022_2023), values(degree_centrality_2020_2021_2022_2023)) for node in nodes]

# Create a DataFrame
df_degree_centrality_2020_2021_2022_2023 = DataFrame(flattened_data, [:Node, :DegreeCentrality])

# Print the DataFrame
println(df_degree_centrality_2020_2021_2022_2023)

# Extract degree centrality values for nodes 1, 2, and 3
nodes_to_compare = [1, 2, 3]

# Define a function to extract degree centrality values for specific nodes
function extract_node_values(df, nodes)
    filtered_data = filter(row -> row.Node in nodes, df)
    return Dict(filtered_data[:, :Node] .=> filtered_data[:, :DegreeCentrality])
end

# Extract degree centrality values for each year
values_2021_nodes = extract_node_values(df_degree_centrality_2020_2021, nodes_to_compare)
values_2022_nodes = extract_node_values(df_degree_centrality_2020_2021_2022, nodes_to_compare)
values_2023_nodes = extract_node_values(df_degree_centrality_2020_2021_2022_2023, nodes_to_compare)

# Convert dictionary values to an array
get_array(dict, keys) = get.(Ref(dict), keys, missing)

# Create a DataFrame to compare values
comparison_df = DataFrame(
    Node = nodes_to_compare,
    #DegreeCentrality_2020 = get_array(values_2020_nodes, nodes_to_compare),
    DegreeCentrality_20_21 = get_array(values_2021_nodes, nodes_to_compare),
    DegreeCentrality_20_21_22 = get_array(values_2022_nodes, nodes_to_compare),
    DegreeCentrality_20_21_22_23 = get_array(values_2023_nodes, nodes_to_compare)
)

# Print the comparison DataFrame
println(comparison_df)


using DataFrames
using Plots

# Sample DataFrame
df = DataFrame(
    Node = [1, 2, 3],
    DegreeCentrality_20 = [9.0, 0, 15.0],
    DegreeCentrality_20_21 = [18.0, 13.0, 25.0],
    DegreeCentrality_20_21_22 = [30.0, 34.0, 27.0],
    DegreeCentrality_20_21_22_23 = [39.0, 42.0, 41.0]
)

# Extract relevant columns for plotting
nodes = df.Node
degree_centralities = select(df, Not(:Node))

# Create a new DataFrame for plotting
df_plot = DataFrame(
    Node = repeat(nodes, inner=size(degree_centralities, 2)),
    DegreeCentrality = collect(Iterators.flatten(eachrow(degree_centralities))),
    Year = repeat(["20", "20_21", "20_21_22", "20_21_22_23"], outer=size(degree_centralities, 1))
)

# Manually set colors for each node
node_colors = Dict(
    1 => :blue,
    2 => :orange,
    3 => :green
)

# Plot degree centrality trends over time with consistent colors for each node
plot()

for node in nodes
    plot!(df_plot[df_plot.Node .== node, :Year], df_plot[df_plot.Node .== node, :DegreeCentrality],
          label="Node $node", color=node_colors[node], marker=:auto, linewidth=2)
end

xlabel!("Year")
ylabel!("Degree Centrality")
title!("Degree Centrality Over Time")



using CSV
using DataFrames


f=CSV.read("dataset_updated.csv",DataFrame)


f.Index = 1:nrow(f)

f[!,2]


h=unique(f[!,2])

ed = Dict()


count=0
for i in h
    count+=1
    ed[i]=count
end


tu=()

l=[]

for i in eachrow(f)
    #println((ed[i[:1]],ed[i[:2]]))
    #println(i[:1])
    push!(l, (ed[i[:1]],ed[i[:2]]))
end



using Graphs, GraphPlot

g = let 
    edges = Graphs.Edge.(l)
    Graph(edges)
end

gplot(g, nodelabel=1:count)

import Pkg
Pkg.add("DataStructures")
using DataStructures

using DataFrames


function counts(obs::AbstractVector{T})::Dict{T,Int} where T
    out = Dict{T,Int}()
    count=DataStructures.counter(obs)
    return count
    out
end


#Computing frequency values
function frequencies(suma::Dict)
    total=0
    for (key, value) in suma
        total=total+value
    end
    Dict(c => v / total for (c, v) in pairs(suma))
    end

function frequencies(vals)
    counts = DataStructures.counter(vals)
    total = length(vals)
    Dict(c => v / total for (c, v) in pairs(counts))
end

#Computing the chance homophily for the specified attributes
function chance_homophily(sumq::Dict{T,Float64})::Float64 where T
    tot=0
    for i in values(sumq)
        tot=tot+i^2
    end
    return tot
end      


function chance_homophily(suma::Dict{T,Int})::Float64 where T
    f=frequencies(suma)
    chance_homophily(f)
    #Dict(c => v / total for (c, v) in pairs(suma))
    
end

function chance_homophily(x::Vector)::Float64
    chance_homophily(counts(x))
end


y=f[!,"University"]
o=[]
for i in y
    #println(i)
    push!(o,i)
end
chu=chance_homophily(o)

y=f[!,"Company"]
o=[]
for i in y
    #println(i)
    push!(o,i)
end
chc=chance_homophily(o)

y=f[!,"Skills"]
o=[]
for i in y
    #println(i)
    push!(o,i)
end
chs=chance_homophily(o)

y=f[!,"Position"]
o=[]
for i in y
    #println(i)
    push!(o,i)
end
chp=chance_homophily(o)

y=f[!,"City"]
o=[]
for i in y
    #println(i)
    push!(o,i)
end
chcity=chance_homophily(o)

#Computing the observed homophily for the specified attributes
function observed_homophily(G::Graphs.SimpleGraph,characteristics::Dict{Int,Any})
    num_ties=0
    num_same_ties=0
    for edge in Graphs.edges(G)
        n1, n2 = Graphs.src(edge), Graphs.dst(edge)
        if haskey(characteristics, n1) && haskey(characteristics, n2)
            
            num_ties+=1

            if characteristics[n1] == characteristics[n2]
                num_same_ties+=1

                
            end
        end
    end
    return (num_same_ties / num_ties)
end

function extract_characteristic_dict(df::DataFrame, characteristic::String)::Dict{Int,Any}
    Dict(x.Index => x[characteristic] for x in eachrow(df))
end

function observed_homophily(G::Graphs.SimpleGraph,observed::DataFrame,characteristic::String)
    y=extract_characteristic_dict(observed,characteristic)
    #print(y)
    observed_homophily(G,y)

    # TODO: your code here
end

ohc=observed_homophily(g,f,"Company")

ohp=observed_homophily(g,f,"Position")

ohu=observed_homophily(g,f,"University")

ohs=observed_homophily(g,f,"Skills")

ohcity=observed_homophily(g,f,"City")

using DataFrames
df = DataFrame()
data = Dict("1Col" => ["Chance Homophily", "Observed Homophily","Result"],
            "Company" => [chc,ohc,"Inverse"],
            "University" => [chu,ohu,"Inverse "],
            "Position" => [chp,ohp,"Inverse "],
            "City" => [chcity,ohcity,"No Homophily"],
            "Skills" => [chs,ohs,"Homophily"])
df = DataFrame(data)
df

using CSV
using DataFrames


f=CSV.read("dataset_updated.csv",DataFrame)


f.Index = 1:nrow(f)

f[!,2]


h=unique(f[!,2])

ed = Dict()


count=0
for i in h
    count+=1
    ed[i]=count
end


tu=()

l=[]

for i in eachrow(f)
    #println((ed[i[:1]],ed[i[:2]]))
    #println(i[:1])
    push!(l, (ed[i[:1]],ed[i[:2]]))
end



using Graphs, GraphPlot

g = let 
    edges = Edge.(l)
    Graph(edges)
end

gplot(g, nodelabel=1:count)

sw=Dict()
str=[]
function strongweak(df)
    for i in eachrow(df)
        name1=i[:1]
        company="s"
        university="s"
        for i in eachrow(df)
            if name1==i[:2]
                company=i[:3]
                university=i[:6]
            end
        end
     
        if (i[:3]==company)|(i[:6]==university)
            println(i[:1],"-",i[:2])
            println(ed[name1],"-",ed[i[:2]])
            sw[( ed[name1] , ed[i[:2]]    )]=2
            sw[( ed[i[:2]] , ed[name1]    )]=2
            push!(str,(ed[name1] , ed[i[:2]]))
        else
            sw[(ed[name1],ed[i[:2]])]=1
            sw[( ed[i[:2]] , ed[name1] )]=1
            
        end   
        
    end
end

strongweak(f)

sw

adj_matrix = adjacency_matrix(g)

for (i, edge) in enumerate(edges(g))
    src_node, dst_node = src(edge), dst(edge)
    #println(src_node)
    adj_matrix[src_node, dst_node]=sw[(src_node,dst_node)]
    println(sw[(src_node,dst_node)])
end

#If the above code snippet throws an error, please restart the kernel and re-run the code from Network Structure 

for i in adj_matrix
    println(i)
end

using Graphs, GraphPlot, SimpleWeightedGraphs


G3 = SimpleWeightedDiGraph(adj_matrix)

gplot(G3,nodelabel=1:count, edgelabel=weight.(edges(G3)))

arr=[]
for i in eachrow(f)
    #println(     sw[ (ed[i[:1]],ed[i[:2]] )]   )
    push!(arr,sw[ (ed[i[:1]],ed[i[:2]] )])
end
arr

f.ties=arr

f

strongties = filter(row -> row["ties"] in [2], f)

# Get the triangle counts for each node
triangle_counts = triangles(g)

# Create a DataFrame with node names and triangle counts
result_df = DataFrame(Node = h, Triangle_Count = triangle_counts)

# Display the result DataFrame
display(result_df)

clustering_coeff = local_clustering_coefficient(g)

# Create a DataFrame with node names and clustering coeff
coeff_df = DataFrame(Node = h, LocalClusteringCoefficient = clustering_coeff)

# Display the result DataFrame
display(coeff_df)


using Plots
using GraphPlot

# Visualize the graph, highlighting nodes with high triangle counts and clustering coefficients
function visualize_network(G, triangle_counts, clustering_coefficients; threshold_triangles=0, threshold_clustering=0)
    # Highlight nodes 1, 2, and 3 with a different color (e.g., green)
    node_colors = [i in [1, 2, 3] ? RGB(0, 1, 0) :
                   ((triangle_counts[i] > threshold_triangles) && (clustering_coefficients[i] > threshold_clustering)) ? RGB(1, 0, 0) : RGB(0, 0, 1)
                   for i in 1:length(triangle_counts)]

    gplot(G, nodelabel=1:length(h), nodefillc=node_colors, nodesize=5)
end

# Set threshold values to highlight nodes
threshold_triangles = 10
threshold_clustering = 0.5

# Visualize the network
visualize_network(g, triangle_counts, clustering_coeff, threshold_triangles=threshold_triangles, threshold_clustering=threshold_clustering)


using Graphs
using GraphPlot
using DataFrames
using Plots

# Function for community detection using connected components
function detect_communities(graph)
    components = connected_components(graph)
    community_labels = zeros(Int, nv(graph))

    for (i, component) in enumerate(components)
        community_labels[component] .= i
    end

    return community_labels
end

# Function to visualize the graph with communities
function visualize_communities(graph, communities, labels)
    node_colors = [RGB(rand(), rand(), rand()) for _ in 1:nv(graph)]
    community_colors = node_colors[communities]

    gplot(graph, nodelabel=labels, nodefillc=community_colors)
end


communities = detect_communities(G_2020_2021)
visualize_communities(G_2020_2021, communities, labels_2020_2021)


using CSV
using DataFrames


f=CSV.read("dataset_updated.csv",DataFrame)

data_2020_2021 = filter(row -> row["Connected On"] in [2020,2021,2022,2023], f)

h=unique(data_2020_2021[!,2])
y=unique(data_2020_2021[!,1])


s=Set()

for i in h
    push!(s,i)
end

for i in y
    push!(s,i)
end

ed = Dict()

count=0

for i in s
    count+=1
    ed[i]=count
end


tu=()


l=[]

for i in eachrow(data_2020_2021)
    push!(l, (ed[i[:1]],ed[i[:2]]))
end


for i in eachrow(data_2020_2021)
    push!(l, (ed[i[:1]],ed[i[:2]]))
end



using Graphs, GraphPlot

g = let 
    edges = Edge.(l)
    Graph(edges)
end

gplot(g, nodelabel=1:count)


f

sw=Dict()
str=[]

function similar(name,df)
    name1=name
    company="s"
    university="s"
    year=0
    city="s"
    for i in eachrow(df)
        if name1==i[:2]
            company=i[:3]
            university=i[:6]
            city=i[:8]
            year=i[:5]
        end
    end
    
    for i in eachrow(df)
        
        if i[:2]!=name1
            count=0

            if i[:3]==company
                count+=1
            end

            if i[:6]==university
                count+=1
            end

            if i[:5]==year
                count+=1
            end

            if i[:8]==city
                count+=1
            end

            sw[( ed[name1] , ed[i[:2]]    )]=count
            sw[( ed[i[:2]] , ed[name1]    )]=count

            if count>=2
                println(i[:2])
            end
        end

        
    end
end

similar("Gowtham",f)

similar("Sahaanah",f)

similar("Snega",f)
