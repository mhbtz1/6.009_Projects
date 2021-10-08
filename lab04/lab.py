#!/usr/bin/env python3
"""6.009 Lab -- Six Double-Oh Mines"""

# NO IMPORTS ALLOWED!


def dump(game):
    """
    Prints a human-readable version of a game (provided as a dictionary)
    """
    for key, val in sorted(game.items()):
        if isinstance(val, list) and val and isinstance(val[0], list):
            print(f'{key}:')
            for inner in val:
                print(f'    {inner}')
        else:
            print(f'{key}:', val)


# 2-D IMPLEMENTATION


def new_game_2d(num_rows, num_cols, bombs):
    """
    Start a new game.

    Return a game state dictionary, with the 'dimensions', 'state', 'board' and
    'mask' fields adequately initialized.

    Parameters:
       num_rows (int): Number of rows
       num_cols (int): Number of columns
       bombs (list): List of bombs, given in (row, column) pairs, which are
                     tuples

    Returns:
       A game state dictionary

    NOTE: Refactored to use the n-dimensional functions, but specified for 2d

    >>> dump(new_game_2d(2, 4, [(0, 0), (1, 0), (1, 1)]))
    board:
        ['.', 3, 1, 0]
        ['.', '.', 1, 0]
    dimensions: (2, 4)
    mask:
        [False, False, False, False]
        [False, False, False, False]
    state: ongoing
    """
    return new_game_nd( (num_rows,num_cols), bombs)
    

def dig_2d(game, row, col):
    """
    Reveal the cell at (row, col), and, in some cases, recursively reveal its
    neighboring squares.

    Update game['mask'] to reveal (row, col).  Then, if (row, col) has no
    adjacent bombs (including diagonally), then recursively reveal (dig up) its
    eight neighbors.  Return an integer indicating how many new squares were
    revealed in total, including neighbors, and neighbors of neighbors, and so
    on.

    The state of the game should be changed to 'defeat' when at least one bomb
    is visible on the board after digging (i.e. game['mask'][bomb_location] ==
    True), 'victory' when all safe squares (squares that do not contain a bomb)
    and no bombs are visible, and 'ongoing' otherwise.

    Parameters:
       game (dict): Game state
       row (int): Where to start digging (row)
       col (int): Where to start digging (col)

    Returns:
       int: the number of new squares revealed

    NOTE: Refactored to use the n-dimensional function, but specified for 2 dimensions
    
    >>> game = {'dimensions': (2, 4),
    ...         'board': [['.', 3, 1, 0],
    ...                   ['.', '.', 1, 0]],
    ...         'mask': [[False, True, False, False],
    ...                  [False, False, False, False]],
    ...         'state': 'ongoing'}
    >>> dig_2d(game, 0, 3)
    4
    >>> dump(game)
    board:
        ['.', 3, 1, 0]
        ['.', '.', 1, 0]
    dimensions: (2, 4)
    mask:
        [False, True, True, True]
        [False, False, True, True]
    state: victory

    >>> game = {'dimensions': [2, 4],
    ...         'board': [['.', 3, 1, 0],a
    ...                   ['.', '.', 1, 0]],
    ...         'mask': [[False, True, False, False],
    ...                  [False, False, False, False]],
    ...         'state': 'ongoing'}
    >>> dig_2d(game, 0, 0)
    1
    >>> dump(game)
    board:
        ['.', 3, 1, 0]
        ['.', '.', 1, 0]
    dimensions: [2, 4]
    mask:
        [True, True, False, False]
        [False, False, False, False]
    state: defeat
    """
    return dig_nd(game, (row,col) )

def chr_add(game, j, k, xray):
    '''
    Determines the character value to return while visualizing the 2d board
    '''
    if(xray):
        if game['board'][j][k] == 0: 
            return ' '
        else:
            return str(game['board'][j][k])
    else:
        if(game['mask'][j][k]):
            if(game['board'][j][k] == 0):
                return ' '
            else:
                return str(game['board'][j][k])
        else:
            return '_'



def render_2d_locations(game, xray=False):
    """
    Prepare a game for display.

    Returns a two-dimensional array (list of lists) of '_' (hidden squares),
    '.' (bombs), ' ' (empty squares), or '1', '2', etc. (squares neighboring
    bombs).  game['mask'] indicates which squares should be visible.  If xray
    is True (the default is False), game['mask'] is ignored and all cells are
    shown.

    Parameters:
       game (dict): Game state
       xray (bool): Whether to reveal all tiles or just the ones allowed by
                    game['mask']

    Returns:
       A 2D array (list of lists)

    >>> render_2d_locations({'dimensions': (2, 4),
    ...         'state': 'ongoing',
    ...         'board': [['.', 3, 1, 0],
    ...                   ['.', '.', 1, 0]],
    ...         'mask':  [[False, True, True, False],
    ...                   [False, False, True, False]]}, False)
    [['_', '3', '1', '_'], ['_', '_', '1', '_']]

    >>> render_2d_locations({'dimensions': (2, 4),
    ...         'state': 'ongoing',
    ...         'board': [['.', 3, 1, 0],
    ...                   ['.', '.', 1, 0]],
    ...         'mask':  [[False, True, False, True],
    ...                   [False, False, False, True]]}, True)
    [['.', '3', '1', ' '], ['.', '.', '1', ' ']]
    """

    ret = []
    for j in range(len(game['board'])):
        app = []
        for k in range(len(game['board'][j])):
            app.append(chr_add(game,j,k,xray))
        ret.append(app)
    return ret


def render_2d_board(game, xray=False):
    """
    Render a game as ASCII art.

    Returns a string-based representation of argument 'game'.  Each tile of the
    game board should be rendered as in the function
        render_2d_locations(game)

    Parameters:
       game (dict): Game state
       xray (bool): Whether to reveal all tiles or just the ones allowed by
                    game['mask']

    Returns:
       A string-based representation of game

    >>> render_2d_board({'dimensions': (2, 4),
    ...                  'state': 'ongoing',
    ...                  'board': [['.', 3, 1, 0],
    ...                            ['.', '.', 1, 0]],
    ...                  'mask':  [[True, True, True, False],
    ...                            [False, False, True, False]]})
    '.31_\\n__1_'
    """
    s = ''
    for j in range(len(game['board'])):
        add_string = ''
        for k in range(len(game['board'][j])):
            add_string += chr_add(game, j, k, xray)
        s += add_string
        if(j != len(game['board']) - 1):
            s += '\n'
    #print(s)
    return s


# N-D IMPLEMENTATION


def generate_nd_neighbors(pos, dim, ptr, s = set()):
    '''
    Generates all the n dimensional neighbors from some specified position by generating a set of the 3^n - 1 possible deviations from some potential position
    '''
    if(ptr == 0):
        q = set()
        if(pos[0] + 1 < dim[0]):
            q.add( (pos[0]+1,) )
        q.add( (pos[0],) )
        if(pos[0] - 1 >= 0):
            q.add( (pos[0]-1,) )
        return q
    ans = set()
    res = generate_nd_neighbors(pos,dim, ptr - 1, s)
    for j in res:
        if( (1+pos[ptr]) < dim[ptr] and j + (1+pos[ptr],) not in s):
            ans.add( j + (1+pos[ptr],))
        if( j + (0+pos[ptr],) not in s):
            ans.add( j + (0+pos[ptr],))
        if( (-1+pos[ptr] >= 0) and j + (-1+pos[ptr],) not in s):
            ans.add( j + (-1+pos[ptr],))
    return ans

def setter(game, position, value, ptr=0, flag=False):
    '''
    Increments/sets some position within game by some given value
    '''
    #print(flag)
    if(ptr == len(position)-1):
        if(not flag and game[position[ptr]] != '.'):
            game[position[ptr]] += value
        elif(flag):
            game[position[ptr]] = value
    else:
        setter(game[position[ptr]], position, value, ptr + 1, flag)
def get(game, position, ptr=0):
    '''
    Gets the value stored in game at some n-dimensional position
    '''
    if(ptr == len(position) - 1):
        return game[position[ptr]]
    return get(game[position[ptr]], position, ptr+1)

def optimize_storage(game, coordinates):
    '''
    Optimizes the algorithm by converting the structures from ndarrays to dictionaries; useful for dig_nd since we have to iterate through all the nodes to know how many safe spaces exist
    '''
    needed = 0
    bomb_set = set()
    hashed_position_coordinates = dict()
    hashed_boolean_coordinates = dict()
    hashed_position_coordinates[coordinates] = get(game['board'], coordinates,0)
    hashed_boolean_coordinates[coordinates] = get(game['mask'], coordinates, 0)

    all_nodes = generate_nd_cells(game)

    for element in all_nodes:
        elem_get = get(game['board'], element,0)
        elem_bool = get(game['mask'], element, 0)
        hashed_position_coordinates[element] = elem_get
        hashed_boolean_coordinates[element] = elem_bool
        if(not elem_bool and elem_get != '.'):
            needed += 1
        if(elem_get == '.'):
            bomb_set.add(element)
    return (hashed_position_coordinates, hashed_boolean_coordinates, bomb_set, needed)



def generate_nd_cells(game):
    ''' 
    Generates a list of all the cells that exist within the game using BFS (make it a generator in order to optimize overall runtime, potentially?)
    '''
    start_node = tuple([0 for i in range(len(game['dimensions']))])
    dummy_mask = set()
    q = []
    cur_ptr = 0
    q.append(start_node)
    all_nodes = []
    
    while(cur_ptr != len(q)):
        tqueue = q[cur_ptr]
        #print(tqueue)
        if(tqueue in dummy_mask):
            cur_ptr += 1
            continue
        else:
            dummy_mask.add(tqueue)
            all_nodes.append(tqueue)
            neighbors = generate_nd_neighbors(tqueue, game['dimensions'], len(game['dimensions']) - 1, dummy_mask)
            for j in neighbors:
                q.append(j)
            cur_ptr += 1
    #print("NODES: {}".format(all_nodes))
    return all_nodes


def construct_nd_grid(dimensions, ptr, default_value = 0):
    '''
    Constructs the n-dimensional mask for game initialization, using some specified default value
    '''
    a = []
    if(ptr == len(dimensions)-1):
        return [default_value for j in range(dimensions[ptr])]
    for j in range(dimensions[ptr]):
        a += [construct_nd_grid(dimensions, ptr + 1, default_value)]
    return a
def new_game_nd(dimensions, bombs):
    """
    Start a new game.

    Return a game state dictionary, with the 'dimensions', 'state', 'board' and
    'mask' fields adequately initialized.


    Args:
       dimensions (tuple): Dimensions of the board
       bombs (list): Bomb locations as a list of lists, each an
                     N-dimensional coordinate

    Returns:
       A game state dictionary

    >>> g = new_game_nd((2, 4, 2), [(0, 0, 1), (1, 0, 0), (1, 1, 1)])
    >>> dump(g)
    board:
        [[3, '.'], [3, 3], [1, 1], [0, 0]]
        [['.', 3], [3, '.'], [1, 1], [0, 0]]
    dimensions: (2, 4, 2)
    mask:
        [[False, False], [False, False], [False, False], [False, False]]
        [[False, False], [False, False], [False, False], [False, False]]
    state: ongoing
    """


    game = {'board':[], 'dimensions':dimensions, 'mask':[], 'state':'ongoing'}
    mask_val = []
    usable_dimensions = list(dimensions)
    game['mask'] = construct_nd_grid(dimensions,0,False)
    game['board'] = construct_nd_grid(dimensions,0,0)
    #print("GRID: {}".format(game['board']))
    for j in bombs:
        setter(game['board'], j, '.', 0, True)
        neighbors = generate_nd_neighbors(j, game['dimensions'], len(game['dimensions']) - 1 )
        for neighbor in neighbors:
            setter(game['board'], neighbor, 1, 0, False)
    return game

    
def dig_nd_helper(game, coordinates, needed, s, bomb_set, hashed_position_coordinates, hashed_boolean_coordinates):
    '''
    Helper method for dig_nd by maintaining a count of iterations to handle some edge cases
    '''
    #print("NEEDED: {}, {}".format(needed,len(s)))
    if(game['state'] != 'ongoing'):
        #print("exit via one")
        return 0

    if(hashed_position_coordinates[coordinates]=='.'):
        setter(game['mask'], coordinates, True, 0,True)
        game['state'] = 'defeat'
        #print("exit via two")
        return 1
    s.add(coordinates)
    coord_neighbors = generate_nd_neighbors(coordinates, game['dimensions'], len(game['dimensions']) - 1, s)
    set_prior = hashed_boolean_coordinates[coordinates]
    hashed_boolean_coordinates[coordinates]=True
    setter(game['mask'], coordinates,True,0,True)
    value = hashed_position_coordinates[coordinates]
    
    if(value == 0):
        ans = 0
        if(not set_prior):
            ans = 1
        for n in coord_neighbors:
            if(hashed_boolean_coordinates[n]):
                continue
            if(n in bomb_set):
                continue
            ans += dig_nd_helper(game, n, needed, s, bomb_set, hashed_position_coordinates, hashed_boolean_coordinates)
        return ans
    else:
        if not set_prior:
            return 1
        return 0

def dig_nd(game, coordinates):
    """
    Recursively dig up square at coords and neighboring squares.

    Update the mask to reveal square at coords; then recursively reveal its
    neighbors, as long as coords does not contain and is not adjacent to a
    bomb.  Return a number indicating how many squares were revealed.  No
    action should be taken and 0 returned if the incoming state of the game
    is not 'ongoing'.

    The updated state is 'defeat' when at least one bomb is visible on the
    board after digging, 'victory' when all safe squares (squares that do
    not contain a bomb) and no bombs are visible, and 'ongoing' otherwise.

    Args:
       coordinates (tuple): Where to start digging

    Returns:
       int: number of squares revealed

    >>> g = {'dimensions': (2, 4, 2),
    ...      'board': [[[3, '.'], [3, 3], [1, 1], [0, 0]],
    ...                [['.', 3], [3, '.'], [1, 1], [0, 0]]],
    ...      'mask': [[[False, False], [False, True], [False, False],
    ...                [False, False]],
    ...               [[False, False], [False, False], [False, False],
    ...                [False, False]]],
    ...      'state': 'ongoing'}
    >>> dig_nd(g, (0, 3, 0))
    8
    >>> dump(g)
    board:
        [[3, '.'], [3, 3], [1, 1], [0, 0]]
        [['.', 3], [3, '.'], [1, 1], [0, 0]]
    dimensions: (2, 4, 2)
    mask:
        [[False, False], [False, True], [True, True], [True, True]]
        [[False, False], [False, False], [True, True], [True, True]]
    state: ongoing
    >>> g = {'dimensions': (2, 4, 2),
    ...      'board': [[[3, '.'], [3, 3], [1, 1], [0, 0]],
    ...                [['.', 3], [3, '.'], [1, 1], [0, 0]]],
    ...      'mask': [[[False, False], [False, True], [False, False],
    ...                [False, False]],
    ...               [[False, False], [False, False], [False, False],
    ...                [False, False]]],
    ...      'state': 'ongoing'}
    >>> dig_nd(g, (0, 0, 1))
    1
    >>> dump(g)
    board:
        [[3, '.'], [3, 3], [1, 1], [0, 0]]
        [['.', 3], [3, '.'], [1, 1], [0, 0]]
    dimensions: (2, 4, 2)
    mask:
        [[False, True], [False, True], [False, False], [False, False]]
        [[False, False], [False, False], [False, False], [False, False]]
    state: defeat
    
    #>>> g = {'dimensions': (2, 4), 'board': [ [0, 0, 0, 0], [0, 0, 0, 0] ], 'mask': [ [False, False, False, False], [False, False, False, False]], 'state':'ongoing'}
    #>>> dig_nd(g, (0,1) )
    #8
    #>>> dump(g)
    #board:
    #    [0, 0, 0, 0]
    #    [0, 0, 0, 0]
    #dimensions: (2,4)
    #mask: 
    #    [True, True, True, True] 
    #    [True, True, True, True]
    #state: victory

    """
    #print(coordinates
    (hashed_position_coordinates, hashed_boolean_coordinates, bomb_set, needed) = optimize_storage(game,coordinates)
    result = dig_nd_helper(game, coordinates, needed, set(), bomb_set, hashed_position_coordinates, hashed_boolean_coordinates)
    #print(result, needed)
    if(result == needed):
        game['state'] = 'victory'
    return result


def render_nd(game, xray=False):
    """
    Prepare the game for display.

    Returns an N-dimensional array (nested lists) of '_' (hidden squares),
    '.' (bombs), ' ' (empty squares), or '1', '2', etc. (squares
    neighboring bombs).  The mask indicates which squares should be
    visible.  If xray is True (the default is False), the mask is ignored
    and all cells are shown.

    Args:
       xray (bool): Whether to reveal all tiles or just the ones allowed by
                    the mask

    Returns:
       An n-dimensional array of strings (nested lists)
        [(0, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 0), (1, 0, 0), (0, 1, 1), (0, 2, 0), (1, 2, 0), (0, 2, 1), (1, 2, 1), (0, 3, 0), (1, 3, 0), (0, 3, 1), (1, 3, 1)]
    
    >>> g = {'dimensions': (2, 4, 2),
    ...      'board': [[[3, '.'], [3, 3], [1, 1], [0, 0]],
    ...                [['.', 3], [3, '.'], [1, 1], [0, 0]]],
    ...      'mask': [[[False, False], [False, True], [True, True],
    ...                [True, True]],
    ...               [[False, False], [False, False], [True, True],
    ...                [True, True]]],
    ...      'state': 'ongoing'}
    >>> render_nd(g, False)
    [[['_', '_'], ['_', '3'], ['1', '1'], [' ', ' ']],
     [['_', '_'], ['_', '_'], ['1', '1'], [' ', ' ']]]

    >>> render_nd(g, True)
    [[['3', '.'], ['3', '3'], ['1', '1'], [' ', ' ']],
     [['.', '3'], ['3', '.'], ['1', '1'], [' ', ' ']]]
    """
    start_node = tuple([0 for i in range(len(game['dimensions']))])
    dummy_mask = construct_nd_grid(game['dimensions'], 0, False)
    render = construct_nd_grid(game['dimensions'],0)
    all_nodes = generate_nd_cells(game)
    for element in all_nodes:
        if(get(game['mask'], element,0) or xray):
            el = get(game['board'], element,0)
            if(el != 0):
                setter(render, element, str(el), 0,True)
            else:
                setter(render, element, ' ', 0,True)
        else:
            setter(render, element, '_', 0,True)
    #print(render)
    return render
    
    
    

if __name__ == "__main__":
    # Test with doctests. Helpful to debug individual lab.py functions.
    import doctest
    doctest_flags = doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS
    doctest.testmod(optionflags=doctest_flags)  # runs ALL doctests
    #s = generate_nd_neighbors( (3,7,6), (10,20,10), 2, set() )
    #print(s)
    #g = {'dimensions': (2, 4), 'board': [ [0, 0, 0, 0], [0, 0, 0, 0] ], 'mask': [ [False, False, False, False], [False, False, False, False]], 'state':'ongoing'}
    #r = dig_nd(g, (1,2) )
    #print("ANS: {}".format(r))
    #print(g)

    #g = new_game_nd((2, 4, 2), [(0,0,1),(1,0,0),(1,1,1)]) #is functional
    #value = dig_nd(g, (0,0,1) ) #is functional
    #h = render_nd(g, False)
    
    #print(generate_nd_neighbors( (0,0,0) ) )
    #print(render_2d_board({'dimensions': (2, 4),'state': 'ongoing','board': [['.', 3, 1, 0],['.', '.', 1, 0]],'mask':  [[True, True, True, False],[False, False, True, False]]}))
    # Alternatively, can run the doctests JUST for specified function/methods,
    # e.g., for render_2d_locations or any other function you might want.  To
    # do so, comment out the above line, and uncomment the below line of code.
    # This may be useful as you write/debug individual doctests or functions.
    # Also, the verbose flag can be set to True to see all test results,
    # including those that pass.
    #
    # doctest.run_docstring_examples(
    #    render_2d_locations,
    #    globals(),
    #    optionflags=_doctest_flags,
    #    verbose=False
    # )
    pass
