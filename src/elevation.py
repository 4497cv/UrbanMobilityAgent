import osmnx as ox
import workspace
import urban_mobility

def run(G, user):
    # mapping node elevations to graph
    print("Adding Elevations to Graph of %s using OSMX" % user.place)
    ox.add_node_elevations_google(G, api_key=workspace.gg_key())
    urban_mobility.save_shp_files_from_graph(G) 
    print("Saving shape files")


    if(user.elevation_active == True):
        urban_mobility.set_elevation_weight(G)
        workspace.set_elevation_flag(True)
    else:
        workspace.set_elevation_flag(False)