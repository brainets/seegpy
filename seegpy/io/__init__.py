"""I/O functions."""
from .load import (get_data_path, load_marsatlas, load_ma_mesh,  # noqa
                   load_ma_labmap, load_ma_table, load_fs_mesh, load_fs_labmap,
                   load_fs_table)
from .read import (read_contacts_trc, read_trm, read_3dslicer_fiducial)  # noqa
from .syslog import (set_log_level)  # noqa
