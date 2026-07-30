import pandas as pd
from itertools import accumulate
from tables import open_file


def read_simulation_config(file_name: str, key: str) -> pd.DataFrame:
    """
    Read MC simulation configuration from HDF5 files.

    Parameters
    ----------
    file_name: str
        HDF5 file to read.
    key: str
        HDF key to read the configuration table from.
    """
    cfg = pd.read_hdf(file_name, key=key)
    with open_file(file_name) as file:
        cfg.attrs = {
            name: getattr(file.root[key].attrs, name)
            for name in file.root[key].attrs._f_list()
        }

    return cfg


def write_simulation_config(cfg: pd.DataFrame, file_name: str, key: str) -> None:
    """
    Write simulation configuration table to
    the specified key of the HDF5 file.

    lstchain expects the resulting table structure to be compliant with PyTables.
    Ensuring the correct format with DataFrame.to_hdf(..., key=some_key, format='table')
    however, stores the resulting table under the additional 'some_key/table' key.
    This function bypasses the issue employing the `tables` module instead.

    If provided data frame defines a dictionary-like "attrs" attribute,
    it will be used update the written table attributes.

    Parameters
    ----------
    cfg: pd.DataFrame
        Simulation configuration data frame to be written down
    file_name: str
        HDF5 file to write to.
    key: str
        HDF key to write the table to.
    """
    with open_file(file_name, mode="a") as file:
        structured_array = cfg.to_records(index=False)

        split = key.split('/')
        table_name = split[-1]
        group_names = split[:-1]
        group_names = list(filter(lambda string: string, group_names))
        group_names = ['/'] + group_names

        roots = list(
            accumulate(group_names[:-1], lambda a, b: a.rstrip('/') + '/' + b.lstrip('/'))
        )
        group_names = group_names[1:]
        for where, name in zip(roots, group_names):
            group = file.create_group(where, name)

        table = file.create_table(group, table_name, structured_array.dtype, "Simulation run config")
        if hasattr(cfg, 'attrs'):
            for name in cfg.attrs:
                table.attrs[name] = cfg.attrs[name]

        table.append(structured_array)
        table.flush()
