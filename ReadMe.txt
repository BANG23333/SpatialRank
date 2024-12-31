Specify paths and hyperparameters in data loading part in sparank.py

# the orginal study area is parititioned into a grid as a rectagle with wide len_x, len_y.
# The shape of the X would be like [samples, sequence_length, len_x, len_y, features]
# To formulate a ranking problem, we reformulate current X and Y to a new list of locations by a mask map

# mask [len_x, len_y] is the map of study area(rectangle shape), where mask[x][y] = 1 means this location is considered in the ranking, 0 # # means it is ignored.

# By using mask map, we only consider some locations and reshape the X and Y into:

# The shape of X: [samples, sequence_length, length_of_list(number of locations), features]
# The shape of Y: [samples, length_of_list(number of locations)]

# the below X, Y, Xv, Yv, Xt, Yt are reshaped matrix (list of locations)

# hyperoparamter num_temporal, num_spatial, num_spatial_tempoal, are the numebr of temporal feature diemnsions, spatial feature dims, and # spatialtemporal feature dims in your X[:, :, :, !] (feature dim)

