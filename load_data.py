import h5py

def load_hdf_data(path, network_name='network', feature_name='features'):
    """Load a GCN input HDF5 container and return its content.

    This funtion reads an already preprocessed data set containing all the
    data needed for training a GCN model in a medical application.
    It extracts a network, features for all of the nodes, the names of the
    nodes (genes) and training, testing and validation splits.

    Parameters:
    ---------
    path:               Path to the container
    network_name:       Sometimes, there might be different networks in the
                        same HDF5 container. This name specifies one of those.
                        Default is: 'network'
    feature_name:       The name of the features of the nodes. Default is: 'features'

    Returns:
    A tuple with all of the data in the order: network, features, y_train, y_val,
    y_test, train_mask, val_mask, test_mask, node names.
    """
    with h5py.File(path, 'r') as f:
        network = f[network_name][:]
        features = f[feature_name][:]
        node_names = f['gene_names'][:]
        y_train = f['y_train'][:]
        y_test = f['y_test'][:]
        if 'y_val' in f:
            y_val = f['y_val'][:]
        else:
            y_val = None
        train_mask = f['mask_train'][:]
        test_mask = f['mask_test'][:]
        if 'mask_val' in f:
            val_mask = f['mask_val'][:]
        else:
            val_mask = None
        if 'feature_names' in f:
            feature_names = f['feature_names'][:]
        else:
            feature_names = None
    return network, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, node_names, feature_names

if __name__ == '__main__':
    path = "./CPDB_multiomics.h5"
    data = load_hdf_data(path, feature_name='features')
    print(data)