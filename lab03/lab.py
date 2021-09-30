#!/usr/bin/env python3

import pickle
# NO ADDITIONAL IMPORTS ALLOWED!

# Note that part of your checkoff grade for this lab will be based on the
# style/clarity of your code.  As you are working through the lab, be on the
# lookout for things that would be made clearer by comments/docstrings, and for
# opportunities to rearrange aspects of your code to avoid repetition (for
# example, by introducing helper functions).  See the following page for more
# information: https://py.mit.edu/fall21/notes/style

'''
My implementation of transform_data returns a tuple (DICT_ONE, DICT_TWO) where:

DICT_ONE is effectively a reversed version of the provided names.pickle, where IDs map to names
DICT_TWO is an adjacency list, where the keys are actor ids, and the elements are tuples of (actor_id, film_that_they_acted_in_together)
'''

def transform_data(smdb, do_film = False):
    with open("resources/names.pickle", 'rb') as nm:
        namedb = pickle.load(nm)
        newdb = {}
        for j in namedb.keys():
            newdb[namedb[j]]=j
        graph_structure = {}
        # loop over the 3-tuple in the data, and we are creating an adjacency list from the data
        for (idx1, idx2, idx3) in smdb:
            if(idx1 == idx2):
                continue
            if(idx1 not in graph_structure.keys()):#if it is not in the dict, then add the key (note that the graph will be bidirectional)
                graph_structure[idx1] = []
                graph_structure[idx1].append((idx2,idx3))
            else:
                graph_structure[idx1].append((idx2,idx3))
            if(idx2 not in graph_structure.keys()):
                graph_structure[idx2] = []
                graph_structure[idx2].append((idx1,idx3))
            else:
                graph_structure[idx2].append((idx1,idx3))
        return (newdb, graph_structure)
def convert_id_to_names(lis, mp):
    return [mp[i] for i in lis]

'''
Returns True if actor_id_1 and actor_id_2 act together (i.e. if actor_id_2 occurs in the list with key actor_id_1)
'''
def acted_together(transformed_data, actor_id_1, actor_id_2):
    (actor_dict, actor_graph) = transformed_data #separate transformed_data into the two structures it contains
    if(actor_id_1 == actor_id_2):
        return True
    '''
    Iterate over the adjacent elements to actor_id_1 (note that these are tuples, not integers)
    '''
    for i in actor_graph[actor_id_1]:
        (a,b) = i
        if(a == actor_id_2):
            print(a,b)
            return True
    return False

'''
Returns a set representing the actors which are a unweighted distance of N away from Kevin Bacon
'''
def actors_with_bacon_number(transformed_data, n):
    return bfs_bacon_number(transformed_data,n)

'''
Returns the shortest unweighted path from Kevin Bacon to any specified actor (i.e. the Bacon path)
'''
def bacon_path(transformed_data, actor_id):
    return bfs_bacon_path(4724, actor_id, transformed_data)

'''
Helper method for actors_with_bacon_number

Performs a BFS(breadth-first-search) using a list for a queue, and returns the set of actors that are N away from Kevin Bacon
'''
def bfs_bacon_number(transformed_data, n):
    vis = set()
    ans = set()
    q = []
    cur_idx = 0
    min_dict = {} #this dictionary will maintain intermediate optimal results for the distance away from Kevin Bacon 
    (actor_dict, actor_graph) = transformed_data
    
    with open("resources/names.pickle", 'rb') as np:
        ld = pickle.load(np)
        q.append( (ld['Kevin Bacon'], 0) )
        while(cur_idx < len(q)):
            (val, amt) = q[cur_idx] #the queue implementation here is a list with a pointer (i.e. when something is "popped" off, the pointer is incremented )
            vis.add(val)
            #print(q, cur_idx)
            if(val not in min_dict.keys()):
                min_dict[val] = amt
            else:
                min_dict[val] = min(min_dict[val],amt)
            lis = actor_graph[val]
            for (j,k) in lis:
                if(j not in vis):
                    if(amt + 1 <= n):
                        q.append( (j, amt + 1) )
            cur_idx += 1
    #print(min_dict)
    for j in min_dict.keys():
        if(min_dict[j] == n):
            ans.add(j)
    return ans
'''
Helper method for bacon_path

Performs a BFS(breadth first search) while also maintaing parent pointers in order to find the actual shortest path between actor_id, and actor_goal
Uses a separate backtracking function to construct the path
'''
def bfs_bacon_path(actor_id, actor_goal, transformed_data, do_film = False):
    vis = set()
    q = []
    cur_idx = 0
    parent_pointers = {}
    min_dict = dict()
    (actor_dict, actor_graph) = transformed_data
    ans = []

    #print(transformed_data[1])

    with open("resources/names.pickle", 'rb') as np:
        ld = pickle.load(np)
        q.append(actor_id)
        vis.add(actor_id)
        path_exists = False
        while(cur_idx < len(q)):
            val = q[cur_idx]
            #print(val)
            adj = transformed_data[1][val]
            if(val == actor_goal):
                path_exists = True #maintains if the path actually exists (so we can return None if it doesn't later)
            for elem in adj:
                if(elem[0] not in vis):
                    parent_pointers[elem[0]] = (val,elem[1]) #parent pointers maintain a tuple of the actor before it, and their common film
                    q.append(elem[0])
                    vis.add(elem[0])
            cur_idx += 1 #after each operation, we increment cur_idx to simulate popping off the queue
    if(not path_exists): 
        return None
    return backtrack(actor_goal,actor_id,parent_pointers)[::-1]

'''
Uses parent pointers to return a list of the path from actor_id to actor_goal

'''
def backtrack(actor_goal, actor_id, parent_pointers):
    ans = []
    #print("PARENTS: {}".format(parent_pointers))
    while(actor_goal != actor_id):
        ans.append(actor_goal)
        actor_goal = parent_pointers[actor_goal][0]
    ans.append(actor_goal)
    return ans
'''
Returns the shortest unweighted path between actor_id_1 and actor_id_2
'''
def actor_to_actor_path(transformed_data, actor_id_1, actor_id_2):
    return bfs_bacon_path(actor_id_1,actor_id_2,transformed_data)

'''
Runs a BFS from a source to multiple 'sinks'(?) based on some goal_test_function (i.e. if it returns true, its a valid sink)
'''
def multisink_bfs(transformed_data, actor_id_1, goal_test_function):
    vis = set() #visited set will maintain nodes the BFS has seen
    q = []
    cur_idx = 0
    parent_pointers = {}
    (actor_dict, actor_graph) = transformed_data
    ans = []
    running_min = 1000000007
    
    #check case where our start node is a valid end node, in which case do nothing
    if(goal_test_function(actor_id_1)):
        return [actor_id_1]
    
    opt_end_point = -1 #this will maintain the current optimal ending node, and will bactrack from it at the end
    with open("resources/names.pickle", 'rb') as np:
        ld = pickle.load(np)
        q.append( (actor_id_1,0) )
        vis.add(actor_id_1)
        path_exists = False
        while(cur_idx < len(q)):
            (val,amt) = q[cur_idx]
            #print(val)
            adj = transformed_data[1][val]
            if(goal_test_function(val)): #check if val is a valid sink
                running_min = min(running_min,amt)
                if(running_min == amt):
                    opt_end_point = val #update opt_end_point if the path is shortened
            for elem in adj:
                if(elem[0] not in vis):
                    parent_pointers[elem[0]] = (val,elem[1])
                    q.append( (elem[0], amt+1) )
                    vis.add(elem[0])
            cur_idx += 1
    #print(parent_pointers)
    if(opt_end_point == -1):
        return None
    return backtrack(opt_end_point,actor_id_1,parent_pointers)[::-1]

'''
Uses multisink_bfs to find the shortest unweighted path from actor_id to any end node which returns True from goal_test_function
'''
def actor_path(transformed_data, actor_id_1, goal_test_function):
    return multisink_bfs(transformed_data,actor_id_1,goal_test_function)




def actors_connecting_films(transformed_data, film1, film2):
    actors_one = (actors_within_movie(film1, transformed_data))
    actors_two = (actors_within_movie(film2, transformed_data))
    comb_set = set()
    for j in actors_two:
        comb_set.add(j)
    ret = []
    for elem in actors_one:
        ans = actor_path(transformed_data, elem, lambda x : (x in comb_set) )
        if(len(ret)==0 or len(ret) > len(ans)):
            ret = ans
    return ret
def actors_within_movie(m1, transformed_data):
    s = set()
    for e in transformed_data[1].keys():
        for (j,k) in transformed_data[1][e]:
            if(k == m1):
                s.add(j)
                s.add(e)
    return s


def goal_function(x):
    return x in {}

if __name__ == '__main__':
    with open('resources/tiny.pickle', 'rb') as f:
        g = open("resources/names.pickle","rb")
        h = open("resources/movies.pickle","rb")
        mdb = pickle.load(h) #database of movies
        rev_movies = {}
        for k in mdb.keys():
            rev_movies[mdb[k]] = k
        ndb = pickle.load(g) #database of names
        smdb = pickle.load(f) #database for graph
        trdata = transform_data(smdb)
        #print(trdata[1])
        #print(trdata[1])
        #print(transform_data(smdb)[1])
        #actor_set = actors_with_bacon_number(transform_data(smdb),2)
        #print(actor_set, len(actor_set))
        '''
        print(ndb['Rod Browning'], ndb['Ellen Page'])
        a = actor_to_actor_path(trdata, ndb['Rod Browning'], ndb['Ellen Page']) + [ndb['Ellen Page']]
        nms = []
        for actor in a:
            nms.append(trdata[0][actor])
        print(a)
        print(nms)
        '''
        #constrained_path = actor_path(trdata, 4724, goal_function)
        #print(convert_id_to_names(constrained_path,trdata[0]))
        #ret = actors_connecting_films(617,31932, test_data)
        #print(ret)

        
    pass
